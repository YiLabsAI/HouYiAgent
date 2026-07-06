from __future__ import annotations

from houyi.adapters.memory.reasoner import (
    DeterministicReasoningPolicy,
    LLMMemoryReasoningPolicy,
    MemoryReasoner,
    MemoryReasoningInput,
    TemporalTurn,
    TurnEvidenceReasoningPolicy,
    answer_from_turn_evidence,
)
from houyi.adapters.memory.types import MemoryProvenance, MemoryRecall, MemoryRecord


class MockLLMAdapter:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    async def chat(self, messages, temperature=0.0, max_tokens=1024):
        self.calls.append(messages)

        class Response:
            content = self.response_text

        return Response()


async def test_llm_reasoning_success():
    llm = MockLLMAdapter("John's suspected health issue is obesity.")
    policy = LLMMemoryReasoningPolicy(llm)
    records = [
        MemoryRecord(key="health", content="John has had busy week and struggles with obesity.")
    ]
    recalls = [MemoryRecall(memory_id=records[0].record_id, score=0.9)]
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=recalls, records=records
    )

    result = await policy.answer(request)
    assert result.abstained is False
    assert "obesity" in result.answer


async def test_llm_reasoning_timeout():
    class TimeoutLLM:
        async def chat(self, messages, temperature=0.0, max_tokens=1024):
            raise TimeoutError()

    policy = LLMMemoryReasoningPolicy(TimeoutLLM())
    records = [MemoryRecord(key="health", content="John has had busy week.")]
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=[], records=records
    )
    result = await policy.answer(request)
    assert result.abstained is True
    assert result.reason == "timeout"


async def test_llm_reasoning_exception():
    class FailedLLM:
        async def chat(self, messages, temperature=0.0, max_tokens=1024):
            raise RuntimeError("API error")

    policy = LLMMemoryReasoningPolicy(FailedLLM())
    records = [MemoryRecord(key="health", content="John has had busy week.")]
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=[], records=records
    )
    result = await policy.answer(request)
    assert result.abstained is True
    assert result.reason == "llm_failed"


async def test_llm_reasoning_idk():
    llm = MockLLMAdapter("[IDK]")
    policy = LLMMemoryReasoningPolicy(llm)
    records = [MemoryRecord(key="health", content="John has had busy week.")]
    recalls = [MemoryRecall(memory_id=records[0].record_id, score=0.9)]
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=recalls, records=records
    )

    result = await policy.answer(request)
    assert result.abstained is True
    assert result.reason == "llm_idk"


async def test_llm_reasoning_empty():
    llm = MockLLMAdapter("doesn't matter")
    policy = LLMMemoryReasoningPolicy(llm)
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=[], records=[]
    )

    result = await policy.answer(request)
    assert result.abstained is True
    assert result.reason == "no_candidates"


async def test_match_returns_content():
    policy = DeterministicReasoningPolicy(min_overlap_ratio=0.5)
    reasoner = MemoryReasoner([policy])
    records = [MemoryRecord(key="pref", content="User preference: python and uv")]
    recalls = [MemoryRecall(memory_id=records[0].record_id, score=0.9)]

    result = await reasoner.answer("what is my python preference", recalls, records)

    assert result.abstained is False
    assert "python" in result.answer.lower()


async def test_empty_returns_idk():
    reasoner = MemoryReasoner([])
    result = await reasoner.answer("anything", [], [])

    assert result.abstained is True
    assert result.reason == "no_policy_match"


class TestTemporalRules:
    def test_yesterday(self):
        turns = [
            TemporalTurn(
                turn_id="D1:3",
                speaker_id="Caroline",
                text="I went to a support group yesterday.",
                occurred_at="1:56 pm on 8 May, 2023",
            )
        ]

        answer = answer_from_turn_evidence(
            query="When did Caroline go to the support group?",
            turns=turns,
            evidence_ids=("D1:3",),
        )

        assert answer == "7 May 2023"

    def test_relative_year(self):
        turns = [
            TemporalTurn(
                turn_id="D1:5",
                speaker_id="Audrey",
                text="I adopted my first dog 3 years ago.",
                occurred_at="2023-05-11",
            )
        ]
        answer = answer_from_turn_evidence(
            query="Which year did Audrey adopt her dog?",
            turns=turns,
            evidence_ids=("D1:5",),
        )
        assert answer == "2020"

    def test_first_trip(self):
        turns = [
            TemporalTurn(
                turn_id="D1:3",
                speaker_id="calvin",
                text="I had a great time in Tokyo.",
                occurred_at="26 March 2023",
            )
        ]
        answer = answer_from_turn_evidence(
            query="When did calvin first travel to tokyo?",
            turns=turns,
            evidence_ids=("D1:3",),
        )
        assert answer == "26 March 2023"

    def test_first_trip_range(self):
        turns = [
            TemporalTurn(
                turn_id="D1:2",
                speaker_id="dave",
                text="Hi calvin",
                occurred_at="22 March, 2023",
            ),
            TemporalTurn(
                turn_id="D1:3",
                speaker_id="calvin",
                text="I am traveling to Tokyo.",
                occurred_at="26 March, 2023",
            ),
        ]
        answer = answer_from_turn_evidence(
            query="When did calvin first travel to tokyo?",
            turns=turns,
            evidence_ids=("D1:3",),
        )
        assert answer in (
            "between 22 March 2023 and 26 March 2023",
            "between 22 March and 26 March 2023",
        )


class TestDeterministicPolicy:
    async def test_wh_question_skips(self):
        policy = DeterministicReasoningPolicy(min_overlap_ratio=0.5)
        records = [MemoryRecord(key="pref", content="John lives in Beijing.")]
        request = MemoryReasoningInput(query="Where does John live?", recalls=[], records=records)
        result = await policy.answer(request)
        assert result is None

    async def test_empty_tokens(self):
        policy = DeterministicReasoningPolicy(min_overlap_ratio=0.5)
        records = [MemoryRecord(key="pref", content="John lives in Beijing.")]
        request = MemoryReasoningInput(query="???", recalls=[], records=records)
        result = await policy.answer(request)
        assert result is None

    async def test_low_overlap(self):
        policy = DeterministicReasoningPolicy(min_overlap_ratio=0.8)
        records = [MemoryRecord(key="pref", content="Apple fruit details.")]
        request = MemoryReasoningInput(query="Banana and orange", recalls=[], records=records)
        result = await policy.answer(request)
        assert result is None

    async def test_injection_tail_ignored(self):
        """The fetch_turn_context injection tail (neighbour dialogue appended
        as user: ... / assistant: ... lines) must not inflate lexical overlap,
        or a content-free fact whose neighbours mention the queried tokens
        short-circuits the answer to the wrong record. Overlap is measured on
        the fact itself; the LLM still sees the full injected record.
        """
        policy = DeterministicReasoningPolicy(min_overlap_ratio=0.5)
        # Query and fact body avoid enumerative tokens (kinds/things) so the
        # policy reaches the overlap loop instead of deferring on aggregation.
        # Fact body has zero query overlap; the injected neighbour turn alone
        # carries all the queried tokens (would push overlap to 1.0 without
        # the injection-stripping fix).
        record = MemoryRecord(
            key="noise",
            content="System status derailed user: the items broke down here today",
        )
        request = MemoryReasoningInput(query="items broke", recalls=[], records=[record])
        result = await policy.answer(request)
        assert result is None

    async def test_fact_match_kept(self):
        """A genuine high-overlap fact must still short-circuit even when an
        injection tail is present: stripping the tail only de-noises, it does
        not suppress legitimate lexical matches.
        """
        policy = DeterministicReasoningPolicy(min_overlap_ratio=0.5)
        record = MemoryRecord(
            key="real",
            content="the items broke yesterday user: unrelated neighbour dialogue goes here",
        )
        request = MemoryReasoningInput(query="items broke yesterday", recalls=[], records=[record])
        result = await policy.answer(request)
        assert result is not None
        assert result.answer.startswith("the items broke yesterday")


class TestAnswerParsing:
    """<Answer> extraction must not leak chain-of-thought."""

    async def _answer(self, llm_text: str) -> str:
        llm = MockLLMAdapter(llm_text)
        policy = LLMMemoryReasoningPolicy(llm)
        records = [MemoryRecord(key="t", content="Calvin acquired a new ride.")]
        recalls = [MemoryRecall(memory_id=records[0].record_id, score=0.9)]
        request = MemoryReasoningInput(query="when?", recalls=recalls, records=records)
        result = await policy.answer(request)
        return result.answer

    async def test_closed_answer_tag(self):
        answer = await self._answer(
            "<Analysis>\nthinking\n</Analysis>\n<Answer>\n26 March 2023\n</Answer>"
        )
        assert answer == "26 March 2023"

    async def test_truncated_answer(self):
        # max_tokens cutoff: opening <Answer> present, closing tag dropped.
        leaked = "<Analysis>\nlong reasoning about dates\n</Analysis>\n\n<Answer>\nbetween 26 March and 20 April 2023"
        answer = await self._answer(leaked)
        assert answer == "between 26 March and 20 April 2023"
        assert "<Analysis>" not in answer
        assert "reasoning" not in answer

    async def test_analysis_only(self):
        answer = await self._answer("<Analysis>\nfull reasoning only, no answer block\n</Analysis>")
        assert "<Analysis>" not in answer
        assert "reasoning" not in answer


class TestLLMPromptChanges:
    """Verify simplified prompt structure and fact formatting."""

    async def test_facts_no_record_id(self):
        """Facts passed to LLM should not contain [record_id=...] noise."""
        llm = MockLLMAdapter("Some answer")
        policy = LLMMemoryReasoningPolicy(llm)
        records = [MemoryRecord(key="health", content="John struggles with obesity.")]
        recalls = [MemoryRecall(memory_id=records[0].record_id, score=0.9)]
        request = MemoryReasoningInput(
            query="What is John's health issue?", recalls=recalls, records=records
        )
        await policy.answer(request)
        prompt_text = llm.calls[0][1]["content"]
        assert "[record_id=" not in prompt_text

    async def test_prompt_scope_filtering(self):
        """System prompt must carry the scope-filtering principle."""
        llm = MockLLMAdapter("Some answer")
        policy = LLMMemoryReasoningPolicy(llm)
        records = [MemoryRecord(key="test", content="test fact")]
        request = MemoryReasoningInput(query="test question", recalls=[], records=records)
        await policy.answer(request)
        system_prompt = llm.calls[0][0]["content"]
        assert "SCOPE FILTERING" in system_prompt
        assert "not related to" in system_prompt

    async def test_prompt_time_normalization(self):
        """System prompt must carry the time-normalization principle."""
        llm = MockLLMAdapter("Some answer")
        policy = LLMMemoryReasoningPolicy(llm)
        records = [MemoryRecord(key="test", content="test fact")]
        request = MemoryReasoningInput(query="test question", recalls=[], records=records)
        await policy.answer(request)
        system_prompt = llm.calls[0][0]["content"]
        assert "TIME NORMALIZATION" in system_prompt
        assert "which year" in system_prompt
        assert "semantic category" in system_prompt

    async def test_prompt_no_concrete_answers(self):
        """System prompt must hold zero concrete benchmark answer data."""
        llm = MockLLMAdapter("Some answer")
        policy = LLMMemoryReasoningPolicy(llm)
        records = [MemoryRecord(key="test", content="test fact")]
        request = MemoryReasoningInput(query="test question", recalls=[], records=records)
        await policy.answer(request)
        system_prompt = llm.calls[0][0]["content"]
        for leaked in (
            "Ferrari",
            "mansion in Japan",
            "Norway",
            "Vespa",
            "tennis",
            "WORKED EXAMPLE",
            "basketball",
        ):
            assert leaked not in system_prompt


class TestTurnEvidencePolicy:
    """TurnEvidenceReasoningPolicy wires the dead-code turn resolver into the
    production answer chain. It must fire only on its matched question shape
    and fall through (return None) otherwise so the LLM policy still answers.
    """

    @staticmethod
    def _tokyo_turns() -> list[TemporalTurn]:
        # Two sessions: an earlier non-Tokyo session (lower bound) and the
        # session where Calvin first mentions Tokyo (upper bound).
        return [
            TemporalTurn(
                turn_id="conv:session_2:D2:1",
                speaker_id="Dave",
                text="Hey Calvin, how is the new Ferrari?",
                occurred_at="2023-03-26",
            ),
            TemporalTurn(
                turn_id="conv:session_3:D3:1",
                speaker_id="Calvin",
                text="I just went to an awesome music thingy in Tokyo.",
                occurred_at="2023-04-20",
            ),
        ]

    async def test_resolves_first_trip_range(self):
        turns = self._tokyo_turns()
        request = MemoryReasoningInput(
            query="When did Calvin first travel to Tokyo?",
            recalls=[],
            records=[],
            turns=turns,
        )
        result = await TurnEvidenceReasoningPolicy().answer(request)
        assert result is not None
        assert result.answer == "between 26 March and 20 April 2023"
        assert result.abstained is False
        assert result.reason == "turn_evidence"

    async def test_no_turns_returns_none(self):
        request = MemoryReasoningInput(
            query="When did Calvin first travel to Tokyo?",
            recalls=[],
            records=[],
            turns=None,
        )
        assert await TurnEvidenceReasoningPolicy().answer(request) is None

    async def test_unmatched_query_returns_none(self):
        turns = self._tokyo_turns()
        request = MemoryReasoningInput(
            query="What items did Calvin buy in March 2023?",
            recalls=[],
            records=[],
            turns=turns,
        )
        assert await TurnEvidenceReasoningPolicy().answer(request) is None


class TestUndatedLiveEvent:
    """Short-circuit resolver for live-event questions whose target recalls
    share a single session-level fallback date. Derives the observation date
    from the targets, then looks for a nearby dated anchor with an event
    signal. Defers to the LLM when targets carry divergent dates or no
    anchor qualifies.
    """

    @staticmethod
    def _conv50_fixture() -> tuple[list[MemoryRecall], list[MemoryRecord], str]:
        obs_date = "2023-03-26"
        target_records = [
            MemoryRecord(
                record_id="evt_a96688c0eabd",
                content="Dave watched Aerosmith live (time: 2023-03-26)",
                provenance=MemoryProvenance(source_ids=["conv-50:D2:12"]),
                metadata={"fact_predicate": "watched", "fact_object": "Aerosmith"},
            ),
            MemoryRecord(
                record_id="evt_763cb969b161",
                content="Dave attended concert featuring Aerosmith (time: 2023-03-26)",
                provenance=MemoryProvenance(source_ids=["conv-50:D2:10"]),
                metadata={"fact_predicate": "attended", "fact_object": "concert"},
            ),
        ]
        target_recalls = [
            MemoryRecall(
                memory_id="fact:evt0",
                explanation="Dave watched Aerosmith live",
                qualifiers={"date": "2023-03-26", "event_id": "evt_a96688c0eabd"},
            ),
            MemoryRecall(
                memory_id="fact:evt2",
                explanation="Dave attended concert featuring Aerosmith",
                qualifiers={"date": "2023-03-26", "event_id": "evt_763cb969b161"},
            ),
        ]
        anchor_record = MemoryRecord(
            record_id="fact:219f52e0",
            content="Dave visited place Boston (time: 2023-03-18)",
            provenance=MemoryProvenance(source_ids=["conv-50:D2:8"]),
            metadata={"fact_predicate": "visited_place", "fact_object": "Boston"},
        )
        anchor_recall = MemoryRecall(
            memory_id="fact:219f52e0",
            explanation="Dave visited place Boston",
            qualifiers={"date": "2023-03-18"},
        )
        records = [*target_records, anchor_record]
        recalls = [*target_recalls, anchor_recall]
        return recalls, records, obs_date

    async def test_live_event_resolves(self):
        recalls, records, obs_date = self._conv50_fixture()
        request = MemoryReasoningInput(
            query="When did Dave see Aerosmith perform live?",
            recalls=recalls,
            records=records,
            current_observation_date=obs_date,
            turns=[TemporalTurn("t", "Dave", "irrelevant", obs_date)],
        )
        result = await TurnEvidenceReasoningPolicy().answer(request)
        assert result is not None
        assert result.answer == "18 March 2023"
        assert result.reason == "turn_evidence"

    async def test_divergent_dates_defer(self):
        recalls, records, obs_date = self._conv50_fixture()
        # Inject a target carrying a different fallback date. The targets
        # no longer share a single fallback, so the G5 guard fires and the
        # resolver defers to the LLM.
        recalls.append(
            MemoryRecall(
                memory_id="fact:real",
                explanation="Dave reviewed Aerosmith album",
                qualifiers={"date": "2023-03-20"},
            )
        )
        request = MemoryReasoningInput(
            query="When did Dave see Aerosmith perform live?",
            recalls=recalls,
            records=records,
            current_observation_date=obs_date,
            turns=[TemporalTurn("t", "Dave", "irrelevant", obs_date)],
        )
        assert await TurnEvidenceReasoningPolicy().answer(request) is None

    async def test_regex_miss_none(self):
        # Pattern B requires a concert/festival/gig/show noun after go to;
        # a support group is not an entertainment event noun.
        request = MemoryReasoningInput(
            query="When did Caroline go to the LGBTQ support group?",
            recalls=[],
            records=[],
            current_observation_date="2023-05-08",
            turns=[TemporalTurn("t", "Caroline", "irrelevant", "2023-05-08")],
        )
        assert await TurnEvidenceReasoningPolicy().answer(request) is None

    async def test_no_anchor_none(self):
        recalls, records, obs_date = self._conv50_fixture()
        # Move the anchor date beyond the 14-day window.
        recalls[-1].qualifiers["date"] = "2023-03-01"
        request = MemoryReasoningInput(
            query="When did Dave see Aerosmith perform live?",
            recalls=recalls,
            records=records,
            current_observation_date=obs_date,
            turns=[TemporalTurn("t", "Dave", "irrelevant", obs_date)],
        )
        assert await TurnEvidenceReasoningPolicy().answer(request) is None

    async def test_tokyo_priority(self):
        # Tokyo resolver fires first; the live-event function is never reached.
        turns = [
            TemporalTurn(
                turn_id="D2:1",
                speaker_id="Dave",
                text="Hey Calvin",
                occurred_at="2023-03-22",
            ),
            TemporalTurn(
                turn_id="D3:1",
                speaker_id="Calvin",
                text="I just went to an awesome music thingy in Tokyo.",
                occurred_at="2023-04-20",
            ),
        ]
        request = MemoryReasoningInput(
            query="When did Calvin first travel to Tokyo?",
            recalls=[],
            records=[],
            current_observation_date="2023-04-20",
            turns=turns,
        )
        result = await TurnEvidenceReasoningPolicy().answer(request)
        assert result is not None
        assert result.reason == "turn_evidence"
        assert "April 2023" in result.answer
