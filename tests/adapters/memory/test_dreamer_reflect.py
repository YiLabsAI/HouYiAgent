from __future__ import annotations

from houyi.adapters.memory.dreamer_reflect import (
    LLMReExtractor,
    MemoryReflector,
    RecallAnchoredSourceSampler,
    ReflectedFact,
    SelfRetrievabilityJudge,
    TokenOverlapGroundingVerifier,
    _parse_reflected_facts,
)
from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RawTurn,
)


def _turn(text: str, *, turn_id: str = "t1", anchor: str = "a1") -> RawTurn:
    return RawTurn(
        turn_id=turn_id,
        namespace="ns",
        session_id="s1",
        role="user",
        content=text,
        metadata={"source_anchor": anchor},
    )


class _FakeCandidate:
    """Minimal stand-in for RecallCandidate with a .fact attribute."""

    def __init__(self, subject: str, obj: str) -> None:
        self.fact = type("F", (), {"subject": subject, "object": obj})()


class _FakeRecall:
    """Fake RecallProbe returning a fixed candidate list per call."""

    def __init__(self, candidates: list[_FakeCandidate]) -> None:
        self._candidates = candidates

    def recall(self, query: str, *, namespace: str, top_k: int = 10):
        return list(self._candidates)


class _FakeSourceReader:
    def __init__(self, turns: list[RawTurn]) -> None:
        self._turns = turns

    def list_turns(self, namespace: str):
        return list(self._turns)


class _FakePromoter:
    """Fake FactPromoter returning a MemoryRecord for each fact."""

    def promote(self, turn: RawTurn, fact: AtomicFact) -> MemoryRecord | None:
        return MemoryRecord(
            scope=MemoryScope.USER,
            key=f"{fact.subject}.{fact.predicate}",
            content=f"{fact.subject} {fact.predicate} {fact.object}",
            memory_type=MemoryType.FACT,
            confidence=0.9,
            valid_from=1.0,
        )


class _RecordingStore:
    """Fake store that records put_record calls so retract is observable."""

    def __init__(self) -> None:
        self.puts: list[MemoryRecord] = []

    def put_record(self, record: MemoryRecord) -> None:
        self.puts.append(record)


class TestRecallAnchoredSampler:
    def test_finds_source_turns(self) -> None:
        """recall top-k fact objects are matched to source turns by tokens."""
        recall = _FakeRecall([_FakeCandidate("Andrew", "wine tasting")])
        reader = _FakeSourceReader(
            [
                _turn("My girlfriend and I went wine tasting", anchor="D25"),
                _turn("I own a dog named Toby", anchor="D1"),
            ]
        )
        sampler = RecallAnchoredSourceSampler(max_turns=5)
        turns = sampler.sample(
            "indoor activities", recall=recall, source_reader=reader, namespace="ns"
        )
        assert len(turns) == 1
        assert turns[0].metadata["source_anchor"] == "D25"

    def test_empty_when_no_recall(self) -> None:
        sampler = RecallAnchoredSourceSampler()
        turns = sampler.sample(
            "q", recall=_FakeRecall([]), source_reader=_FakeSourceReader([]), namespace="ns"
        )
        assert turns == []


class TestGroundingVerifier:
    def test_accepts_supported(self) -> None:
        turns = [_turn("Andrew went to wine tasting with girlfriend", anchor="D25")]
        fact = ReflectedFact("Andrew", "went_to", "wine tasting")
        grounded = TokenOverlapGroundingVerifier().verify(fact, turns)
        assert grounded is not None
        assert grounded.source_anchor == "D25"
        assert grounded.subject == "Andrew"

    def test_rejects_hallucination(self) -> None:
        turns = [_turn("Andrew went to wine tasting", anchor="D25")]
        fact = ReflectedFact("Andrew", "did", "skydiving")  # not in any source turn
        assert TokenOverlapGroundingVerifier().verify(fact, turns) is None


class TestParseReflectedFacts:
    def test_parses_json(self) -> None:
        raw = '{"facts": [{"subject": "Andrew", "predicate": "went to", "object": "wine tasting", "event_time": "2023-10-21"}]}'
        facts = _parse_reflected_facts(raw)
        assert len(facts) == 1
        assert facts[0].subject == "Andrew"
        assert facts[0].predicate == "went_to"
        assert facts[0].object == "wine tasting"
        assert facts[0].event_time == "2023-10-21"

    def test_empty_on_garbage(self) -> None:
        assert _parse_reflected_facts("not json") == []
        assert _parse_reflected_facts('{"facts": []}') == []


class TestLLMReExtractor:
    def test_recovers_coparticipation(self) -> None:
        """The reflector preserves co-participation in the source turn."""

        class _Adapter:
            async def chat(self, messages, *, temperature=0.0, max_tokens=512):
                return type(
                    "R",
                    (),
                    {
                        "content": '{"facts": [{"subject": "Andrew", '
                        '"predicate": "went to", "object": "wine tasting with girlfriend"}]}'
                    },
                )()

        turns = [_turn("My girlfriend and I went to this awesome wine tasting", anchor="D25")]
        facts = LLMReExtractor().reflect(
            "indoor activities with girlfriend?", turns, llm=_Adapter()
        )
        assert len(facts) == 1
        assert "girlfriend" in facts[0].object

    def test_no_invention(self) -> None:
        """Garbage LLM output degrades to an empty list, never raises."""

        class _BadAdapter:
            async def chat(self, messages, *, temperature=0.0, max_tokens=512):
                raise RuntimeError("model down")

        facts = LLMReExtractor().reflect("q", [_turn("src")], llm=_BadAdapter())
        assert facts == []


class TestSelfRetrievabilityJudge:
    def test_keeps_retrievable(self) -> None:
        store = _RecordingStore()
        promoter = _FakePromoter()
        judge = SelfRetrievabilityJudge(promoter, store)
        fact = AtomicFact(
            subject="Andrew",
            predicate="went_to",
            object="wine tasting",
            certainty=Certainty.CERTAIN,
            source_anchor="D25",
        )
        # Recall returns a candidate whose object overlaps the fact's object.
        recall = _FakeRecall([_FakeCandidate("Andrew", "wine tasting")])
        record = judge.judge("q", fact, _turn("src"), recall=recall, namespace="ns")
        assert record is not None
        assert all(r.valid_to is None for r in store.puts)  # no retraction

    def test_retracts_when_absent(self) -> None:
        store = _RecordingStore()
        judge = SelfRetrievabilityJudge(_FakePromoter(), store)
        fact = AtomicFact(
            subject="Andrew",
            predicate="went_to",
            object="wine tasting",
            certainty=Certainty.CERTAIN,
            source_anchor="D25",
        )
        # Recall returns nothing overlapping -> retract.
        recall = _FakeRecall([_FakeCandidate("Bob", "hiking")])
        record = judge.judge("q", fact, _turn("src"), recall=recall, namespace="ns")
        assert record is None
        # The retraction re-puts the record with valid_to set.
        retracted = [r for r in store.puts if r.valid_to is not None]
        assert len(retracted) == 1


class TestMemoryReflector:
    def test_end_to_end(self) -> None:
        """sample -> reflect -> ground -> judge keeps the retrievable fact."""

        class _Adapter:
            async def chat(self, messages, *, temperature=0.0, max_tokens=512):
                return type(
                    "R",
                    (),
                    {
                        "content": '{"facts": [{"subject": "Andrew", '
                        '"predicate": "went to", "object": "wine tasting with girlfriend"}]}'
                    },
                )()

        store = _RecordingStore()
        recall = _FakeRecall([_FakeCandidate("Andrew", "wine tasting")])
        reflector = MemoryReflector(
            sampler=RecallAnchoredSourceSampler(),
            reextractor=LLMReExtractor(),
            verifier=TokenOverlapGroundingVerifier(),
            judge=SelfRetrievabilityJudge(_FakePromoter(), store),
            recall=recall,
            source_reader=_FakeSourceReader(
                [_turn("My girlfriend and I went wine tasting", anchor="D25")]
            ),
            llm=_Adapter(),
        )
        report = reflector.reflect(["indoor activities?"], namespace="ns")
        assert report.facts_extracted == 1
        assert report.facts_grounded == 1
        assert report.facts_kept == 1
        assert len(report.kept_records) == 1

    def test_skips_when_no_failures(self) -> None:
        reflector = MemoryReflector(
            sampler=RecallAnchoredSourceSampler(),
            reextractor=LLMReExtractor(),
            verifier=TokenOverlapGroundingVerifier(),
            judge=SelfRetrievabilityJudge(_FakePromoter(), _RecordingStore()),
            recall=_FakeRecall([]),
            source_reader=_FakeSourceReader([]),
            llm=None,
        )
        report = reflector.reflect([], namespace="ns")
        assert report.facts_kept == 0
