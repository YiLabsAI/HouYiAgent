"""Recall pipeline orchestrator.

The orchestrator owns stage order only: route the query, select
retrievers, fuse candidates, de-duplicate, and apply the unknown-answer
guard. Each component remains independently testable and replaceable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from houyi.adapters.memory.event_emitter import MemoryEventEmitter
from houyi.adapters.memory.recall.fusion import MMRDeduplicator, WeightedFuser
from houyi.adapters.memory.recall.idk_guard import IDKGuard
from houyi.adapters.memory.recall.rerank import EvidenceAwareReranker, Reranker
from houyi.adapters.memory.recall.retrievers.base import Retriever, RetrieverError
from houyi.adapters.memory.recall.router import QueryRouter
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallQuery,
    RecallReason,
    RecallResult,
    RetrieverContext,
    RetrieverKind,
)
from houyi.application.evolution.events import EvolutionEventType

# Recall outcomes that warrant a RECALL_FAILURE event for the evolution
# control plane. SUFFICIENT and EXPLICIT_ABSENCE are healthy outcomes;
# everything else is a signal that the retrieval mix could improve.
_FAILURE_REASONS: frozenset[RecallReason] = frozenset(
    {
        RecallReason.NO_CANDIDATES,
        RecallReason.LOW_EVIDENCE,
        RecallReason.CONTRADICTING_EVIDENCE,
    }
)

_SOURCE_FALLBACK_SCORE = 2.0
_SOURCE_SNIPPET_CHARS = 2000

# ROUTING_TABLE — names map 1:1 to the retriever registry keys in the
# RecallOrchestrator constructor. vector is opt-in: it only fires when
# the caller wires a vector retriever into the orchestrator, otherwise
# its slot is a no-op.
_DEFAULT_ROUTE_TABLE: dict[QueryType, tuple[str, ...]] = {
    QueryType.FACTUAL_LOOKUP: ("entity_state", "vector", "raw_turn"),
    QueryType.NEGATION_CHECK: ("entity_state",),
    QueryType.TEMPORAL_QUERY: ("timeline", "vector"),
    QueryType.RELATIONAL_CHAIN: ("iterative", "entity_state", "vector"),
    QueryType.PROCEDURAL_RECALL: ("raw_turn", "vector"),
    QueryType.THEMATIC_SUMMARY: ("vector", "raw_turn", "timeline"),
}

# Per-query-type fusion weight matrix. Vector weights track : it
# is the primary signal for thematic_summary, a moderate boost for
# factual_lookup and relational_chain (where it reranks the
# entity-state hit set), and intentionally low for negation_check
# where positive evidence must come from entity_state alone.
_DEFAULT_FUSION_WEIGHTS: dict[QueryType, dict[RetrieverKind, float]] = {
    QueryType.FACTUAL_LOOKUP: {
        RetrieverKind.ENTITY_STATE: 10.0,
        RetrieverKind.VECTOR: 1.5,
        RetrieverKind.RAW_TURN: 0.6,
        RetrieverKind.ITERATIVE: 1.0,
        RetrieverKind.TIMELINE: 0.4,
    },
    QueryType.NEGATION_CHECK: {
        RetrieverKind.ENTITY_STATE: 10.0,
        RetrieverKind.VECTOR: 0.1,
        RetrieverKind.RAW_TURN: 0.2,
        RetrieverKind.ITERATIVE: 0.5,
        RetrieverKind.TIMELINE: 0.2,
    },
    QueryType.TEMPORAL_QUERY: {
        RetrieverKind.TIMELINE: 3.0,
        RetrieverKind.VECTOR: 0.8,
        RetrieverKind.ENTITY_STATE: 1.0,
        RetrieverKind.RAW_TURN: 0.6,
        RetrieverKind.ITERATIVE: 0.5,
    },
    QueryType.RELATIONAL_CHAIN: {
        RetrieverKind.ITERATIVE: 5.0,
        RetrieverKind.VECTOR: 1.2,
        RetrieverKind.ENTITY_STATE: 1.0,
        RetrieverKind.RAW_TURN: 0.5,
        RetrieverKind.TIMELINE: 0.3,
    },
    QueryType.THEMATIC_SUMMARY: {
        RetrieverKind.VECTOR: 2.5,
        RetrieverKind.RAW_TURN: 1.2,
        RetrieverKind.TIMELINE: 0.8,
        RetrieverKind.ENTITY_STATE: 0.5,
        RetrieverKind.ITERATIVE: 0.3,
    },
    QueryType.PROCEDURAL_RECALL: {
        RetrieverKind.RAW_TURN: 2.0,
        RetrieverKind.VECTOR: 1.0,
        RetrieverKind.ENTITY_STATE: 0.5,
        RetrieverKind.TIMELINE: 0.4,
        RetrieverKind.ITERATIVE: 0.3,
    },
}


@dataclass(frozen=True)
class RecallPipelineConfig:
    """Runtime knobs for the recall orchestrator."""

    route_table: Mapping[QueryType, Sequence[str]] = field(
        default_factory=lambda: dict(_DEFAULT_ROUTE_TABLE)
    )
    fusion_weights: Mapping[QueryType, Mapping[RetrieverKind, float]] = field(
        default_factory=lambda: dict(_DEFAULT_FUSION_WEIGHTS)
    )
    parallel_retrieval: bool = False
    rerank_multiplier: int = 2


class RecallOrchestrator:
    """Coordinate routing, retrieval, fusion, and answer guarding."""

    def __init__(
        self,
        *,
        router: QueryRouter,
        retrievers: Mapping[str, Retriever],
        fuser: WeightedFuser | None = None,
        deduplicator: MMRDeduplicator | None = None,
        reranker: Reranker | None = None,
        guard: IDKGuard | None = None,
        config: RecallPipelineConfig | None = None,
        emitter: MemoryEventEmitter | None = None,
    ) -> None:
        if router is None:
            raise ValueError("router is required")
        self._router = router
        self._retrievers = dict(retrievers)
        self._fuser = fuser or WeightedFuser()
        self._dedupe = deduplicator or MMRDeduplicator()
        self._reranker = reranker or EvidenceAwareReranker()
        self._guard = guard or IDKGuard()
        self._config = config or RecallPipelineConfig()
        # Optional hot-path event emitter. When supplied, the orchestrator
        # publishes one RECALL_FAILURE per recall whose final reason
        # indicates a retrieval miss (post source-fallback). Side-channel
        # only — never blocks or fails the recall path.
        self._emitter = emitter or MemoryEventEmitter()

    async def recall(
        self,
        query: RecallQuery,
        ctx: RetrieverContext | None = None,
    ) -> RecallResult:
        runtime = ctx or RetrieverContext()
        route = await self._router.classify(query)
        retriever_names = list(self._config.route_table.get(route.query_type, ()))
        retrievers = [
            self._retrievers[name] for name in retriever_names if name in self._retrievers
        ]

        trace: dict[str, object] = {
            "route": route.model_dump(),
            "retrievers": retriever_names,
            "errors": [],
        }
        raw_candidates = await self._retrieve_all(query, runtime, retrievers, trace)
        deduped = await self._rank_candidates(
            route.query_type,
            raw_candidates,
            top_k=query.top_k,
            trace=trace,
        )
        result = self._guard.evaluate(
            query_type=route.query_type,
            candidates=deduped,
            trace=trace,
        )
        if result.reason != RecallReason.LOW_EVIDENCE:
            self._emit_outcome(query, route.query_type, result, candidates_seen=len(deduped))
            return result

        source_candidates = await self._source_fallback(deduped, runtime, trace)
        if not source_candidates:
            fallback_result = self._guard.evaluate(
                query_type=route.query_type,
                candidates=deduped,
                trace=trace,
            )
            self._emit_outcome(
                query, route.query_type, fallback_result, candidates_seen=len(deduped)
            )
            return fallback_result

        deduped_with_source = await self._rank_candidates(
            route.query_type,
            [*deduped, *source_candidates],
            top_k=query.top_k,
            trace=trace,
        )
        final_result = self._guard.evaluate(
            query_type=route.query_type,
            candidates=deduped_with_source,
            trace=trace,
        )
        self._emit_outcome(
            query,
            route.query_type,
            final_result,
            candidates_seen=len(deduped_with_source),
        )
        return final_result

    def _emit_outcome(
        self,
        query: RecallQuery,
        query_type: QueryType,
        result: RecallResult,
        *,
        candidates_seen: int,
    ) -> None:
        """Publish RECALL_FAILURE for terminal misses; healthy outcomes are silent.

        We use the final reason (post source-fallback) so the evolution
        signal miner does not over-count transient LOW_EVIDENCE that the
        fallback recovered from.
        """
        if result.reason not in _FAILURE_REASONS:
            return
        self._emitter.emit(
            EvolutionEventType.RECALL_FAILURE,
            target="recall_orchestrator",
            payload={
                "query_type": query_type.value,
                "reason": result.reason.value,
                "namespace": query.namespace,
            },
            metrics={
                "candidates": float(candidates_seen),
                "top_k": float(query.top_k),
            },
            namespace=query.namespace,
        )

    async def _rank_candidates(
        self,
        query_type: QueryType,
        candidates: Sequence[RecallCandidate],
        *,
        top_k: int,
        trace: dict[str, object],
    ) -> list[RecallCandidate]:
        fusion_k = max(top_k * self._config.rerank_multiplier, top_k)
        fuser = self._fuser_for(query_type)
        fused = fuser.fuse(candidates, top_k=fusion_k)
        deduped = self._dedupe.dedupe(fused, top_k=fusion_k)
        ranked = await self._reranker.arerank(
            query_type=query_type,
            candidates=deduped,
            top_k=top_k,
        )
        trace["rerank"] = {
            "reranker": type(self._reranker).__name__,
            "input_count": len(deduped),
            "output_count": len(ranked),
        }
        return ranked

    def _fuser_for(self, query_type: QueryType) -> WeightedFuser:
        if type(self._fuser) is not WeightedFuser:
            return self._fuser
        weights = self._config.fusion_weights.get(query_type)
        if weights is None:
            return self._fuser
        return WeightedFuser(kind_weights=dict(weights))

    async def _retrieve_all(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
        retrievers: Sequence[Retriever],
        trace: dict[str, object],
    ) -> list[RecallCandidate]:
        if self._config.parallel_retrieval:
            groups = await asyncio.gather(
                *(self._safe_retrieve(r, query, ctx, trace) for r in retrievers)
            )
            return [candidate for group in groups for candidate in group]

        candidates: list[RecallCandidate] = []
        for retriever in retrievers:
            candidates.extend(await self._safe_retrieve(retriever, query, ctx, trace))
        return candidates

    async def _safe_retrieve(
        self,
        retriever: Retriever,
        query: RecallQuery,
        ctx: RetrieverContext,
        trace: dict[str, object],
    ) -> list[RecallCandidate]:
        try:
            return await retriever.retrieve(query, ctx)
        except RetrieverError as exc:
            errors = trace.setdefault("errors", [])
            if isinstance(errors, list):
                errors.append({"retriever": retriever.name, "error": str(exc)})
            return []

    async def _source_fallback(
        self,
        candidates: Sequence[RecallCandidate],
        ctx: RetrieverContext,
        trace: dict[str, object],
    ) -> list[RecallCandidate]:
        reader = ctx.source_reader
        if reader is None:
            return []

        reads = trace.setdefault("source_reads", [])
        source_candidates: list[RecallCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if len(seen) >= ctx.max_source_reads:
                break
            anchor = candidate.fact.source_anchor
            if anchor in seen:
                continue
            seen.add(anchor)
            try:
                text = await asyncio.to_thread(reader.read_source_chunk, anchor)
            except Exception as exc:
                if isinstance(reads, list):
                    reads.append({"source_anchor": anchor, "error": str(exc)})
                continue
            if not text or not text.strip():
                if isinstance(reads, list):
                    reads.append({"source_anchor": anchor, "found": False})
                continue
            if isinstance(reads, list):
                reads.append({"source_anchor": anchor, "found": True})
            source_candidates.append(_source_candidate(candidate, text))
        return source_candidates


def _source_candidate(candidate: RecallCandidate, text: str) -> RecallCandidate:
    source_candidate = candidate.model_copy(deep=True)
    source_candidate.score = max(source_candidate.score, _SOURCE_FALLBACK_SCORE)
    source_candidate.matched_by = RetrieverKind.RAW_TURN
    source_candidate.retriever_name = "SourceChunkFallback"
    source_candidate.signals = dict(source_candidate.signals)
    source_candidate.signals["source_rehydrated"] = True
    source_candidate.signals["source_text"] = text[:_SOURCE_SNIPPET_CHARS]
    source_candidate.explanation = "source chunk confirms candidate evidence"
    return source_candidate


__all__ = ["RecallOrchestrator", "RecallPipelineConfig"]
