from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator, RecallPipelineConfig
from houyi.adapters.memory.recall.retrievers.base import Retriever, RetrieverError
from houyi.adapters.memory.recall.retrievers.entity_state import EntityStateRetriever
from houyi.adapters.memory.recall.retrievers.iterative import IterativeMultiHopRetriever
from houyi.adapters.memory.recall.router import (
    CascadingRouter,
    QueryRouter,
    RouteDecision,
    Tier0RuleRouter,
)
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallQuery,
    RecallReason,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, Certainty, EntityStateRecord


class FakeView:
    def get_active(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        if namespace != "n" or entity != "Martin" or attribute != "phone":
            return []
        return [
            EntityStateRecord(
                namespace="n",
                entity="Martin",
                attribute="phone",
                value="123",
                certainty=Certainty.CERTAIN,
                source_unit_id="u1",
            )
        ]


class ChainView:
    def __init__(self, *, complete: bool = True) -> None:
        self.complete = complete

    def get_active(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        if namespace != "n":
            return []
        rows = [
            EntityStateRecord(
                namespace="n",
                entity="martin",
                attribute="manager",
                value="Alice",
                certainty=Certainty.CERTAIN,
                source_unit_id="u-manager",
            )
        ]
        if self.complete:
            rows.append(
                EntityStateRecord(
                    namespace="n",
                    entity="Alice",
                    attribute="email",
                    value="a@example.com",
                    certainty=Certainty.CERTAIN,
                    source_unit_id="u-email",
                )
            )
        return [
            row
            for row in rows
            if row.entity.casefold() == entity.casefold()
            and (attribute is None or row.attribute == attribute)
        ]


def make_candidate(
    score: float,
    *,
    obj: str = "coffee",
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
        retriever_name="fake",
    )


class FixedRouter(QueryRouter):
    def __init__(self, query_type: QueryType) -> None:
        self.query_type = query_type

    async def classify(self, query: RecallQuery) -> RouteDecision:
        return RouteDecision(query_type=self.query_type, confidence=1.0)


class FixedRetriever(Retriever):
    def __init__(self, candidates: list[RecallCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[RecallQuery, RetrieverContext]] = []

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        self.calls.append((query, ctx))
        return self.candidates


class BrokenRetriever(Retriever):
    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        raise RetrieverError("backend unavailable")


@pytest.mark.asyncio
async def test_orchestrator_recalls() -> None:
    orchestrator = RecallOrchestrator(
        router=CascadingRouter(Tier0RuleRouter()),
        retrievers={"entity_state": EntityStateRetriever(FakeView())},
    )

    result = await orchestrator.recall(RecallQuery(text="phone of Martin", namespace="n"))

    assert result.reason == RecallReason.SUFFICIENT
    assert result.top() is not None
    assert result.top().fact.object == "123"


@pytest.mark.asyncio
async def test_orchestrator_traces_error() -> None:
    good = FixedRetriever([make_candidate(1.0, obj="tea")])
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.FACTUAL_LOOKUP),
        retrievers={"bad": BrokenRetriever(), "good": good},
        config=RecallPipelineConfig(route_table={QueryType.FACTUAL_LOOKUP: ("bad", "good")}),
    )

    result = await orchestrator.recall(RecallQuery(text="anything"))

    assert result.reason == RecallReason.SUFFICIENT
    assert result.top() is not None
    assert result.top().fact.object == "tea"
    assert result.trace["errors"] == [
        {"retriever": "BrokenRetriever", "error": "backend unavailable"}
    ]


@pytest.mark.asyncio
async def test_orchestrator_parallel() -> None:
    first = FixedRetriever([make_candidate(0.1, obj="one", kind=RetrieverKind.RAW_TURN)])
    second = FixedRetriever([make_candidate(1.0, obj="two")])
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.FACTUAL_LOOKUP),
        retrievers={"first": first, "second": second},
        config=RecallPipelineConfig(
            route_table={QueryType.FACTUAL_LOOKUP: ("first", "second")},
            parallel_retrieval=True,
        ),
    )

    result = await orchestrator.recall(RecallQuery(text="anything", top_k=1))

    assert result.top() is not None
    assert result.top().fact.object == "two"
    assert len(first.calls) == 1
    assert len(second.calls) == 1


@pytest.mark.asyncio
async def test_orchestrator_missing_route() -> None:
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.THEMATIC_SUMMARY),
        retrievers={},
        config=RecallPipelineConfig(route_table={}),
    )

    result = await orchestrator.recall(RecallQuery(text="anything"))

    assert result.reason == RecallReason.NO_CANDIDATES
    assert result.trace["retrievers"] == []


@pytest.mark.asyncio
async def test_orchestrator_chain_recall() -> None:
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.RELATIONAL_CHAIN),
        retrievers={"iterative": IterativeMultiHopRetriever(ChainView())},
    )

    result = await orchestrator.recall(
        RecallQuery(text="email of manager of Martin", namespace="n")
    )

    assert result.reason == RecallReason.SUFFICIENT
    assert [hit.fact.object for hit in result.candidates] == ["Alice", "a@example.com"]
    assert result.trace["rerank"]["reranker"] == "EvidenceAwareReranker"
    assert all(hit.signals["chain_evidence_complete"] is True for hit in result.candidates)


@pytest.mark.asyncio
async def test_orchestrator_chain_partial() -> None:
    orchestrator = RecallOrchestrator(
        router=FixedRouter(QueryType.RELATIONAL_CHAIN),
        retrievers={"iterative": IterativeMultiHopRetriever(ChainView(complete=False))},
    )

    result = await orchestrator.recall(
        RecallQuery(text="email of manager of Martin", namespace="n")
    )

    assert result.reason == RecallReason.LOW_EVIDENCE
    assert result.trace["guard"]["evidence_coverage"] > 0.0


def test_orchestrator_requires_router() -> None:
    with pytest.raises(ValueError):
        RecallOrchestrator(router=None, retrievers={})  # type: ignore[arg-type]


def _timed_candidate(score: float, *, obj: str, event_time: str | None) -> RecallCandidate:
    return RecallCandidate(
        fact=AtomicFact(
            subject="Audrey",
            predicate="owns_dog" if event_time else "loves",
            object=obj,
            event_time=event_time,
            certainty=Certainty.CERTAIN,
            source_anchor=f"s-{obj}",
        ),
        score=score,
        matched_by=RetrieverKind.ENTITY_STATE,
        retriever_name="fake",
    )


def test_infer_answer_type() -> None:
    """Classifies temporal ('when/which year') and numeric ('how many') asks."""
    from houyi.adapters.memory.recall.orchestrator import _infer_answer_type

    assert "temporal" in _infer_answer_type("Which year did Audrey adopt her dogs?")
    assert "temporal" in _infer_answer_type("When did Calvin travel to Tokyo?")
    assert "numeric" in _infer_answer_type("How many dogs does Audrey own?")
    # A plain what/who lookup asks for neither a date nor a count.
    assert _infer_answer_type("What kind of car does Evan drive?") == frozenset()


def test_answer_type_boost() -> None:
    """Temporal boost adds a small tiebreaker to a dated query-entity fact.

    The boost is a 0-1-scale tiebreaker (not a score-lifter): it nudges a
    dated fact about the queried entity above same-score noise, but does
    not overcome a real score gap -- a generic fact the cross-encoder
    scored higher stays higher. This keeps the cross-encoder's
    discrimination in charge instead of drowning it.
    """
    from houyi.adapters.memory.recall.orchestrator import (
        _ANSWER_TYPE_BOOST,
        _apply_answer_type_boost,
    )

    generic = _timed_candidate(1.0, obj="dogs", event_time=None)
    dated = _timed_candidate(0.8, obj="dog Pepper", event_time="2020")
    _apply_answer_type_boost([generic, dated], frozenset({"temporal"}), ["Audrey"])

    assert dated.signals["rerank_score"] == 0.8 + _ANSWER_TYPE_BOOST
    assert "answer_type_boost" in dated.signals
    assert "answer_type_boost" not in generic.signals


def test_boost_entity_gated() -> None:
    """A dated fact about another person is NOT boosted for the query entity.

    Guards the conv-48 regression: 'when did Deborah's mother pass away'
    must not elevate an unrelated dated 'Jolene lost mother (2022)' over
    the relevant but undated Deborah fact.
    """
    from houyi.adapters.memory.recall.orchestrator import _apply_answer_type_boost

    other = RecallCandidate(
        fact=AtomicFact(
            subject="Jolene",
            predicate="lost_family_member",
            object="mother",
            event_time="2022",
            certainty=Certainty.CERTAIN,
            source_anchor="s-jolene",
        ),
        score=5.0,
        matched_by=RetrieverKind.ENTITY_STATE,
        retriever_name="fake",
    )
    _apply_answer_type_boost([other], frozenset({"temporal"}), ["Deborah"])
    assert "answer_type_boost" not in other.signals


class TestFusionStrategySwitch:
    def test_default_is_rrf(self) -> None:
        from houyi.adapters.memory.recall.fusion import ReciprocalRankFuser

        orchestrator = RecallOrchestrator(
            router=FixedRouter(QueryType.FACTUAL_LOOKUP),
            retrievers={},
        )
        assert orchestrator._config.fusion_strategy == "rrf"
        assert isinstance(orchestrator._fuser_for(QueryType.FACTUAL_LOOKUP), ReciprocalRankFuser)

    def test_weighted_strategy_used(self) -> None:
        from houyi.adapters.memory.recall.fusion import WeightedFuser

        orchestrator = RecallOrchestrator(
            router=FixedRouter(QueryType.FACTUAL_LOOKUP),
            retrievers={},
            config=RecallPipelineConfig(fusion_strategy="weighted"),
        )
        assert isinstance(orchestrator._fuser_for(QueryType.FACTUAL_LOOKUP), WeightedFuser)

    def test_rrf_per_query_weights(self) -> None:
        from houyi.adapters.memory.recall.fusion import ReciprocalRankFuser

        orchestrator = RecallOrchestrator(
            router=FixedRouter(QueryType.FACTUAL_LOOKUP),
            retrievers={},
            config=RecallPipelineConfig(fusion_strategy="rrf"),
        )
        fuser = orchestrator._fuser_for(QueryType.FACTUAL_LOOKUP)
        assert isinstance(fuser, ReciprocalRankFuser)
        # Per-query kind weights flow into the RRF fuser.
        assert fuser._weight(RetrieverKind.GRAPH) == pytest.approx(1.1)

    def test_temporal_stays_weighted(self) -> None:
        from houyi.adapters.memory.recall.fusion import ReciprocalRankFuser, WeightedFuser

        orchestrator = RecallOrchestrator(
            router=FixedRouter(QueryType.TEMPORAL_QUERY),
            retrievers={},
            config=RecallPipelineConfig(fusion_strategy="rrf"),
        )
        # RRF buries single-source dated gold, so temporal keeps WeightedFuser.
        assert isinstance(orchestrator._fuser_for(QueryType.TEMPORAL_QUERY), WeightedFuser)
        assert isinstance(orchestrator._fuser_for(QueryType.FACTUAL_LOOKUP), ReciprocalRankFuser)


class TestTemporalNoTruncation:
    """Temporal queries pass the full candidate pool to the reranker (no
    fusion_k cut) so the cross-encoder can surface dated events buried
    below the weight-monopolised fusion top-k (conv-42 / conv-50 root
    cause). Other query types keep the bounded fusion_k. The carve-out is
    scoped to TEMPORAL only to avoid bloating enumeration pools."""

    @staticmethod
    def _candidates(n: int) -> list[RecallCandidate]:
        return [
            RecallCandidate(
                fact=AtomicFact(
                    subject="s",
                    predicate="p",
                    object=f"o{i}",
                    certainty=Certainty.CERTAIN,
                    source_anchor=f"a{i}",
                ),
                score=1.0,
                matched_by=RetrieverKind.ENTITY_STATE,
                retriever_name="fake",
            )
            for i in range(n)
        ]

    @staticmethod
    def _build_orch(query_type: QueryType, seen: list[int]):
        from houyi.adapters.memory.recall.fusion import Fuser
        from houyi.adapters.memory.recall.rerank import Reranker

        class SpyFuser(Fuser):
            def fuse(self, candidates, *, top_k):
                seen.append(top_k)
                return list(candidates)[:top_k]

        class SpyReranker(Reranker):
            def rerank(self, *, query_type, candidates, top_k, query=None):
                return list(candidates)[:top_k]

        return RecallOrchestrator(
            router=FixedRouter(query_type),
            retrievers={},
            fuser=SpyFuser(),
            reranker=SpyReranker(),
        )

    @pytest.mark.asyncio
    async def test_temporal_full_pool(self) -> None:
        seen: list[int] = []
        orch = self._build_orch(QueryType.TEMPORAL_QUERY, seen)
        await orch._rank_candidates(
            QueryType.TEMPORAL_QUERY, self._candidates(30), top_k=3, trace={}
        )
        assert seen[-1] == 30, "TEMPORAL must pass the full pool (no fusion_k cut)"

    @pytest.mark.asyncio
    async def test_factual_bounded_k(self) -> None:
        seen: list[int] = []
        orch = self._build_orch(QueryType.FACTUAL_LOOKUP, seen)
        await orch._rank_candidates(
            QueryType.FACTUAL_LOOKUP, self._candidates(30), top_k=3, trace={}
        )
        # default rerank_multiplier=2 -> max(3*2, 3) = 6
        assert seen[-1] == 6, "non-temporal must keep the bounded fusion_k"
