from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.retrievers.entity_state import EntityStateRetriever
from houyi.adapters.memory.recall.retrievers.iterative import (
    GapAnalysis,
    IterativeMultiHopRetriever,
)
from houyi.adapters.memory.recall.retrievers.raw_turn import RawTurnLogRetriever
from houyi.adapters.memory.recall.retrievers.timeline import TimelineRetriever
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, Certainty, EntityStateRecord


class FakeView:
    def __init__(self) -> None:
        self.active_calls: list[tuple[str, str, str | None]] = []
        self.as_of_calls: list[tuple[str, str, float, str | None]] = []
        self.history_calls: list[tuple[str, str, str | None]] = []
        self.rows = [
            EntityStateRecord(
                namespace="n",
                entity="Martin",
                attribute="phone",
                value="123",
                certainty=Certainty.CERTAIN,
                valid_from=10.0,
                source_unit_id="u1",
            ),
            EntityStateRecord(
                namespace="n",
                entity="Martin",
                attribute="manager",
                value="Alice",
                certainty=Certainty.CERTAIN,
                valid_from=20.0,
                source_unit_id="u2",
            ),
            EntityStateRecord(
                namespace="n",
                entity="Alice",
                attribute="email",
                value="a@example.com",
                certainty=Certainty.CERTAIN,
                valid_from=30.0,
                source_unit_id="u3",
            ),
            EntityStateRecord(
                namespace="n",
                entity="Martin",
                attribute="identity",
                value="Martin",
                certainty=Certainty.CERTAIN,
                valid_from=0.0,
                source_unit_id="u5",
            ),
            EntityStateRecord(
                namespace="n",
                entity="Martin",
                attribute="city",
                value="Paris",
                certainty=Certainty.CERTAIN,
                valid_from=1.0,
                valid_to=5.0,
                source_unit_id="u4",
            ),
        ]

    def get_active(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        self.active_calls.append((namespace, entity, attribute))
        return [
            r
            for r in self.rows
            if r.namespace == namespace
            and r.entity.casefold() == entity.casefold()
            and r.valid_to is None
            and (attribute is None or r.attribute.casefold() == attribute.casefold())
        ]

    def get_as_of(
        self,
        namespace: str,
        entity: str,
        ts: float,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        self.as_of_calls.append((namespace, entity, ts, attribute))
        return [
            r
            for r in self.rows
            if r.namespace == namespace
            and r.entity.casefold() == entity.casefold()
            and r.valid_from <= ts
            and (r.valid_to is None or r.valid_to > ts)
            and (attribute is None or r.attribute.casefold() == attribute.casefold())
        ]

    def get_history(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        self.history_calls.append((namespace, entity, attribute))
        return [
            r
            for r in self.rows
            if r.namespace == namespace
            and r.entity.casefold() == entity.casefold()
            and (attribute is None or r.attribute.casefold() == attribute.casefold())
        ]


class DelegateRetriever(Retriever):
    def __init__(self) -> None:
        self.calls: list[RecallQuery] = []

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        self.calls.append(query)
        entity = (query.entity_hint or "").casefold()
        if entity == "martin" and query.attribute_hint is None:
            return [
                _recall_candidate("Martin", "manager", "Alice", "u2"),
                _recall_candidate("Martin", "phone", "123", "u1"),
            ]
        if entity == "alice" and query.attribute_hint == "email":
            return [_recall_candidate("Alice", "email", "a@example.com", "u3")]
        return []


class ManyAnalyzer:
    def analyze(
        self,
        query: RecallQuery,
        candidates: list[RecallCandidate],
        *,
        max_subqueries: int,
    ) -> GapAnalysis:
        return GapAnalysis(
            subqueries=tuple(
                RecallQuery(
                    text=f"attr{i} of Entity{i}",
                    namespace=query.namespace,
                    entity_hint=f"Entity{i}",
                    attribute_hint=f"attr{i}",
                )
                for i in range(max_subqueries + 2)
            )
        )


def _recall_candidate(
    subject: str,
    predicate: str,
    obj: str,
    source_anchor: str,
) -> RecallCandidate:
    return RecallCandidate(
        fact=AtomicFact(
            subject=subject,
            predicate=predicate,
            object=obj,
            certainty=Certainty.CERTAIN,
            source_anchor=source_anchor,
        ),
        score=1.0,
        matched_by=RetrieverKind.ENTITY_STATE,
    )


@pytest.mark.asyncio
async def test_entity_lookup_hits() -> None:
    retriever = EntityStateRetriever(FakeView())

    hits = await retriever.retrieve(
        RecallQuery(text="phone of Martin", namespace="n"),
        RetrieverContext(),
    )

    assert len(hits) == 1
    assert hits[0].matched_by == RetrieverKind.ENTITY_STATE
    assert hits[0].fact.object == "123"
    assert hits[0].signals["exact_attribute"] is True


@pytest.mark.asyncio
async def test_entity_uses_hints() -> None:
    view = FakeView()
    retriever = EntityStateRetriever(view)

    hits = await retriever.retrieve(
        RecallQuery(
            text="anything",
            namespace="n",
            entity_hint="Martin",
            attribute_hint="manager",
        ),
        RetrieverContext(),
    )

    assert [hit.fact.object for hit in hits] == ["Alice"]
    assert view.active_calls == [("n", "Martin", "manager")]
    assert hits[0].signals["hint_source"] == "caller_hint"


@pytest.mark.asyncio
async def test_entity_broad_lookup() -> None:
    retriever = EntityStateRetriever(FakeView())

    hits = await retriever.retrieve(
        RecallQuery(text="who is Martin", namespace="n"),
        RetrieverContext(),
    )

    assert {hit.fact.predicate for hit in hits} == {"phone", "manager"}
    assert all(hit.score == 5.0 for hit in hits)


@pytest.mark.asyncio
async def test_entity_trim_greedy() -> None:
    view = FakeView()
    retriever = EntityStateRetriever(view)

    # Martin is the entity, followed by a verb break.
    hits = await retriever.retrieve(
        RecallQuery(text="When did Martin lose his key?", namespace="n"),
        RetrieverContext(),
    )

    # Should match Martin in FakeView even with the trailing lose phrase.
    assert len(hits) == 2
    assert view.active_calls[-1][1] == "Martin"


@pytest.mark.asyncio
async def test_entity_no_hint() -> None:
    view = FakeView()
    retriever = EntityStateRetriever(view)

    hits = await retriever.retrieve(RecallQuery(text="tell me stories"), RetrieverContext())

    assert hits == []
    assert view.active_calls == []


def test_entity_requires_view() -> None:
    with pytest.raises(ValueError):
        EntityStateRetriever(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_entity_filters_identity_anchor() -> None:
    retriever = EntityStateRetriever(FakeView())

    hits = await retriever.retrieve(
        RecallQuery(text="phone of Martin", namespace="n"),
        RetrieverContext(),
    )

    predicates = [h.fact.predicate for h in hits]
    assert "identity" not in predicates
    assert "phone" in predicates


@pytest.mark.asyncio
async def test_entity_compound_accumulation() -> None:
    view = FakeView()
    # Add multiple accumulated facts
    view.rows.extend(
        [
            EntityStateRecord(
                namespace="n",
                entity="John",
                attribute="enjoys",
                value="swimming",
                certainty=Certainty.CERTAIN,
                valid_from=10.0,
                source_unit_id="u1",
                qualifiers={"accumulate": "true"},
            ),
            EntityStateRecord(
                namespace="n",
                entity="John",
                attribute="enjoys",
                value="reading",
                certainty=Certainty.CERTAIN,
                valid_from=11.0,
                source_unit_id="u2",
                qualifiers={"accumulate": "true"},
            ),
            EntityStateRecord(
                namespace="n",
                entity="John",
                attribute="enjoys",
                value="running",
                certainty=Certainty.CERTAIN,
                valid_from=12.0,
                source_unit_id="u3",
                qualifiers={"accumulate": "true"},
            ),
        ]
    )

    retriever = EntityStateRetriever(view)
    hits = await retriever.retrieve(
        RecallQuery(text="What does John enjoy?", namespace="n"),
        RetrieverContext(),
    )

    # Retriever should return all 3 individual candidates for Candidate Homogeneity in Fusion
    assert len(hits) == 3

    # Now verify that the post-fusion sibling merge correctly aggregates them
    from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator
    from houyi.adapters.memory.recall.router import CascadingRouter, Tier0RuleRouter

    orchestrator = RecallOrchestrator(
        router=CascadingRouter(Tier0RuleRouter()),
        retrievers={"entity_state": retriever},
    )

    merged_hits = orchestrator._merge_siblings_post_fusion(hits)

    # One candidate should subsume all three enjoys objects
    enjoys_cands = [h for h in merged_hits if h.fact.predicate == "enjoys"]
    assert len(enjoys_cands) == 1

    cand = enjoys_cands[0]
    assert "swimming" in cand.fact.object
    assert "reading" in cand.fact.object
    assert "running" in cand.fact.object

    # source_anchor keeps the best member's original anchor (not comma-joined)
    # so the bench R@10 parser can still extract a single dia_id.
    # Full anchor list is in signals["compound_source_anchors"].
    assert cand.fact.source_anchor == "u1"  # best member's anchor preserved
    assert "compound_source_anchors" in cand.signals
    assert set(cand.signals["compound_source_anchors"]) == {"u1", "u2", "u3"}

    # Ensure it's marked as a compound candidate in signals
    assert "compound_members" in cand.signals
    assert len(cand.signals["compound_members"]) == 3
    assert cand.signals["compound_size"] == 3


@pytest.mark.asyncio
async def test_timeline_returns_history() -> None:
    retriever = TimelineRetriever(FakeView())

    hits = await retriever.retrieve(
        RecallQuery(text="Martin", namespace="n", entity_hint="Martin"),
        RetrieverContext(),
    )

    assert {hit.fact.predicate for hit in hits} == {"phone", "manager", "city"}
    assert all(hit.matched_by == RetrieverKind.TIMELINE for hit in hits)
    assert all(hit.signals["mode"] == "history" for hit in hits)


@pytest.mark.asyncio
async def test_timeline_uses_as_of() -> None:
    view = FakeView()
    retriever = TimelineRetriever(view)

    hits = await retriever.retrieve(
        RecallQuery(
            text="historical lookup",
            namespace="n",
            entity_hint="Martin",
            attribute_hint="city",
            as_of=4.0,
        ),
        RetrieverContext(),
    )

    assert [hit.fact.object for hit in hits] == ["Paris"]
    assert view.as_of_calls == [("n", "Martin", 4.0, "city")]
    assert hits[0].signals["mode"] == "as_of"


@pytest.mark.asyncio
async def test_timeline_no_target() -> None:
    view = FakeView()
    retriever = TimelineRetriever(view)

    hits = await retriever.retrieve(RecallQuery(text="recent work"), RetrieverContext())

    assert hits == []
    assert view.history_calls == []


def test_timeline_requires_view() -> None:
    with pytest.raises(ValueError):
        TimelineRetriever(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_iterative_resolves_chain() -> None:
    retriever = IterativeMultiHopRetriever(FakeView())

    hits = await retriever.retrieve(
        RecallQuery(text="email of manager of Martin", namespace="n"),
        RetrieverContext(),
    )

    assert [hit.fact.object for hit in hits] == ["Alice", "a@example.com"]
    assert all(hit.matched_by == RetrieverKind.ITERATIVE for hit in hits)


@pytest.mark.asyncio
async def test_iterative_partial_chain() -> None:
    retriever = IterativeMultiHopRetriever(FakeView())

    hits = await retriever.retrieve(
        RecallQuery(text="phone of manager of Martin", namespace="n"),
        RetrieverContext(),
    )

    assert [hit.fact.object for hit in hits] == ["Alice"]
    assert hits[0].signals["hop_index"] == 1


@pytest.mark.asyncio
async def test_iterative_caps_hops() -> None:
    view = FakeView()
    retriever = IterativeMultiHopRetriever(view, max_hops=1)

    hits = await retriever.retrieve(
        RecallQuery(text="email of manager of Martin", namespace="n"),
        RetrieverContext(),
    )

    assert [hit.fact.object for hit in hits] == ["Alice"]
    assert view.active_calls == [("n", "martin", "manager")]


@pytest.mark.asyncio
async def test_iterative_rejects_plain() -> None:
    view = FakeView()
    retriever = IterativeMultiHopRetriever(view)

    hits = await retriever.retrieve(
        RecallQuery(text="Martin phone", namespace="n"), RetrieverContext()
    )

    assert hits == []
    assert view.active_calls == []


@pytest.mark.asyncio
async def test_iterative_uses_subqueries() -> None:
    delegate = DelegateRetriever()
    retriever = IterativeMultiHopRetriever(FakeView(), delegate=delegate)

    hits = await retriever.retrieve(
        RecallQuery(text="email of manager of Martin", namespace="n"),
        RetrieverContext(),
    )

    assert [hit.fact.object for hit in hits] == ["Alice", "123", "a@example.com"]
    assert [(q.entity_hint, q.attribute_hint) for q in delegate.calls] == [
        ("martin", None),
        ("Alice", "email"),
    ]
    assert hits[0].signals["iteration_round"] == 1
    assert hits[-1].signals["iteration_round"] == 2


@pytest.mark.asyncio
async def test_iterative_caps_subqueries() -> None:
    delegate = DelegateRetriever()
    retriever = IterativeMultiHopRetriever(
        FakeView(),
        delegate=delegate,
        analyzer=ManyAnalyzer(),
        max_subqueries=2,
    )

    await retriever.retrieve(RecallQuery(text="email of manager of Martin"), RetrieverContext())

    assert len(delegate.calls) == 3


def test_iterative_rejects_max_hops() -> None:
    with pytest.raises(ValueError):
        IterativeMultiHopRetriever(FakeView(), max_hops=0)


def test_iterative_requires_view() -> None:
    with pytest.raises(ValueError):
        IterativeMultiHopRetriever(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_raw_turn_empty() -> None:
    hits = await RawTurnLogRetriever().retrieve(RecallQuery(text="anything"), RetrieverContext())

    assert hits == []


def test_clean_entity_possessive() -> None:
    from houyi.adapters.memory.recall.retrievers.entity_state import _clean_entity

    assert _clean_entity("John's goals with regards to his basketball career") == "John"


def test_clean_entity_regular() -> None:
    from houyi.adapters.memory.recall.retrievers.entity_state import _clean_entity

    assert _clean_entity("Mary's") == "Mary"


class TestInferEntity:
    def test_possessive(self) -> None:
        from houyi.adapters.memory.recall.retrievers.entity_state import _infer_entity_attribute

        query = RecallQuery(text="what are John's goals with regards to his basketball career?")
        hint = _infer_entity_attribute(query)
        assert hint is not None
        assert hint.entity == "John"
        assert hint.attribute == "goals"
        assert hint.source == "possessive"

    def test_kind_of(self) -> None:
        from houyi.adapters.memory.recall.retrievers.entity_state import _infer_entity_attribute

        query = RecallQuery(
            text="What kind of indoor activities has Andrew pursued with his girlfriend?"
        )
        hint = _infer_entity_attribute(query)
        assert hint is not None
        assert hint.entity == "Andrew"
        assert hint.attribute == "indoor activities"
        assert hint.source == "kind_of"

    def test_wh_noun(self) -> None:
        from houyi.adapters.memory.recall.retrievers.entity_state import _infer_entity_attribute

        query = RecallQuery(text="Which places or events have John and James planned to meet at?")
        hint = _infer_entity_attribute(query)
        assert hint is not None
        assert hint.entity == "John"
        assert hint.attribute == "places or events"
        assert hint.source == "wh_noun"

    def test_question_hint(self) -> None:
        from houyi.adapters.memory.recall.retrievers.entity_state import _infer_entity_attribute

        query = RecallQuery(text="who is there", entity_hint="who")
        assert _infer_entity_attribute(query) is None

    def test_zh_attribute(self) -> None:
        from houyi.adapters.memory.recall.retrievers.entity_state import _infer_entity_attribute

        # \u7ea6\u7ff0\u7684\u5e74\u9f84 represents Chinese text for John's age
        query = RecallQuery(text="\u7ea6\u7ff0\u7684\u5e74\u9f84")
        hint = _infer_entity_attribute(query)
        assert hint is not None
        assert hint.entity == "\u7ea6\u7ff0"
        assert hint.attribute == "\u5e74\u9f84"
        assert hint.source == "zh_attribute"


def test_stemish() -> None:
    from houyi.adapters.memory.recall.retrievers.entity_state import _stemish

    assert _stemish("wanted") == "want"
    assert _stemish("dogs") == "dog"
    assert _stemish("cooking") == "cook"
    assert _stemish("cat") == "cat"
