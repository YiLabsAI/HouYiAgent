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
