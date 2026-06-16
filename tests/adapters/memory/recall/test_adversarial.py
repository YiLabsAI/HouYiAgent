from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.idk_guard import IDKGuard, IDKGuardConfig
from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator, RecallPipelineConfig
from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.router import QueryRouter, RouteDecision
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallQuery,
    RecallReason,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, Certainty


class FixedRouter(QueryRouter):
    def __init__(self, query_type: QueryType) -> None:
        self.query_type = query_type

    async def classify(self, query: RecallQuery) -> RouteDecision:
        return RouteDecision(query_type=self.query_type, confidence=1.0)


class FixedRetriever(Retriever):
    def __init__(self, candidates: list[RecallCandidate]) -> None:
        self.candidates = candidates

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        return self.candidates


class EmptyReader:
    def read_source_chunk(self, source_anchor: str) -> str | None:
        return None


def make_candidate(
    obj: str,
    *,
    score: float = 1.0,
    signals: dict[str, object] | None = None,
    kind: RetrieverKind = RetrieverKind.ENTITY_STATE,
) -> RecallCandidate:
    return RecallCandidate(
        fact=AtomicFact(
            subject="user",
            predicate="likes",
            object=obj,
            certainty=Certainty.CERTAIN,
            source_anchor=f"s-{obj}",
        ),
        score=score,
        matched_by=kind,
        retriever_name="fixed",
        signals=signals or {},
    )


@pytest.mark.asyncio
async def test_adversarial_unknown() -> None:
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.FACTUAL_LOOKUP),
        retrievers={"fixed": FixedRetriever([])},
        config=RecallPipelineConfig(route_table={QueryType.FACTUAL_LOOKUP: ("fixed",)}),
        guard=IDKGuard(config=IDKGuardConfig(coverage_threshold=0.3)),
    )

    result = await orchestrator.recall(RecallQuery(text="unknown preference"))

    assert result.reason == RecallReason.NO_CANDIDATES
    assert result.suggested_action == "admit_unknown"


@pytest.mark.asyncio
async def test_adversarial_low_source() -> None:
    # Use empty object -> empty source_anchor -> coverage=0.0 to guarantee LOW_EVIDENCE
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.FACTUAL_LOOKUP),
        retrievers={
            "fixed": FixedRetriever(
                [make_candidate("coffee", score=0.01, kind=RetrieverKind.TIMELINE)]
            )
        },
        config=RecallPipelineConfig(route_table={QueryType.FACTUAL_LOOKUP: ("fixed",)}),
        guard=IDKGuard(config=IDKGuardConfig(coverage_threshold=0.3)),
    )

    result = await orchestrator.recall(
        RecallQuery(text="what does the user like"),
        RetrieverContext(source_reader=EmptyReader()),
    )

    assert result.reason == RecallReason.LOW_EVIDENCE
    assert result.trace["source_reads"] == [{"source_anchor": "s-coffee", "found": False}]


def test_adversarial_contradiction() -> None:
    result = IDKGuard().evaluate(
        query_type=QueryType.FACTUAL_LOOKUP,
        candidates=[
            make_candidate("coffee", signals={"contradicts": True}),
            make_candidate("tea"),
        ],
    )

    assert result.reason == RecallReason.CONTRADICTING_EVIDENCE


@pytest.mark.asyncio
async def test_adversarial_distractor() -> None:
    first = make_candidate(
        "Alice",
        score=0.5,
        signals={"iteration_round": 1},
    )
    chain = make_candidate(
        "a@example.com",
        score=0.5,
        signals={"iteration_round": 2, "source_rehydrated": True},
    )
    distractor = make_candidate("unrelated", score=0.8)
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.RELATIONAL_CHAIN),
        retrievers={"fixed": FixedRetriever([distractor, first, chain])},
        config=RecallPipelineConfig(route_table={QueryType.RELATIONAL_CHAIN: ("fixed",)}),
    )

    result = await orchestrator.recall(RecallQuery(text="email of manager of Martin"))

    assert result.top() is not None
    assert result.top().fact.object == "a@example.com"


@pytest.mark.asyncio
async def test_adversarial_partial_chain() -> None:
    partial = make_candidate("Alice", score=1.0, signals={"iteration_round": 1})
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.RELATIONAL_CHAIN),
        retrievers={"fixed": FixedRetriever([partial])},
        config=RecallPipelineConfig(route_table={QueryType.RELATIONAL_CHAIN: ("fixed",)}),
    )

    result = await orchestrator.recall(RecallQuery(text="email of manager of Martin"))

    assert result.reason == RecallReason.LOW_EVIDENCE
