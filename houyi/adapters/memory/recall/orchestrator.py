"""Recall pipeline orchestrator.

The orchestrator owns stage order only: route the query, select
retrievers, fuse candidates, de-duplicate, and apply the unknown-answer
guard. Each component remains independently testable and replaceable.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from houyi.adapters.memory.event_emitter import MemoryEventEmitter
from houyi.adapters.memory.recall.enumeration import (
    EnumerationBooster,
    detect_enumeration_category,
)
from houyi.adapters.memory.recall.fusion import (
    Fuser,
    MMRDeduplicator,
    ReciprocalRankFuser,
    WeightedFuser,
)
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

# Coverage mode for enumeration queries ("what activities/items/places
# has X"). Such queries score every member fact at the same lexical
# floor, so relevance ranking is uninformative and a normal narrow cut
# drops members by arbitrary tie-order. Coverage mode widens the pre-MMR
# pool (so no member is truncated before diversity selection) and raises
# the MMR diversity weight (so the budget spreads across distinct member
# facts rather than rephrasings of one).
_ENUMERATION_DIVERSITY = 0.7
_ENUMERATION_FUSION_FLOOR = 150

# Answer-type relevance boost. A question that asks for a date ("when did
# X", "which year did X") should prefer facts that actually carry a date,
# and a "how many/how much/how often" question should prefer facts whose
# object contains a number. Without this, multiple facts share one source
# turn and the generic, qualifier-less rephrasings ("X loves dogs") outrank
# the one fact that carries the answer ("X owns dog Pepper (time: 2020)"),
# so the answer-bearing fact is cut before the reasoner ever reads it.
# The boost encodes a universal retrieval prior (match the evidence type to
# the question's answer type); it is not tied to any specific dataset.
_ANSWER_TYPE_BOOST = 0.8
_ANSWER_TYPE_TEMPORAL_RE = re.compile(
    r"\bwhen\b|\b(?:what|which)\s+(?:year|date|month|day|time)\b", re.IGNORECASE
)
_ANSWER_TYPE_NUMERIC_RE = re.compile(r"\bhow\s+(?:many|much|often)\b", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")

# ROUTING_TABLE — names map 1:1 to the retriever registry keys in the
# RecallOrchestrator constructor. vector is opt-in: it only fires when
# the caller wires a vector retriever into the orchestrator, otherwise
# its slot is a no-op.
_DEFAULT_ROUTE_TABLE: dict[QueryType, tuple[str, ...]] = {
    QueryType.FACTUAL_LOOKUP: ("entity_state", "event", "graph", "vector", "raw_turn"),
    QueryType.NEGATION_CHECK: ("entity_state", "vector"),
    QueryType.TEMPORAL_QUERY: ("timeline", "event", "graph", "entity_state", "vector"),
    QueryType.RELATIONAL_CHAIN: ("iterative", "graph", "event", "entity_state", "vector"),
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
        RetrieverKind.ENTITY_STATE: 1.0,
        RetrieverKind.EVENT: 1.0,
        RetrieverKind.GRAPH: 1.1,
        RetrieverKind.VECTOR: 1.0,
        RetrieverKind.RAW_TURN: 0.7,
        RetrieverKind.ITERATIVE: 0.8,
        RetrieverKind.TIMELINE: 0.7,
    },
    QueryType.NEGATION_CHECK: {
        RetrieverKind.ENTITY_STATE: 1.2,
        RetrieverKind.EVENT: 0.7,
        RetrieverKind.GRAPH: 0.7,
        RetrieverKind.VECTOR: 0.7,
        RetrieverKind.RAW_TURN: 0.7,
        RetrieverKind.ITERATIVE: 0.7,
        RetrieverKind.TIMELINE: 0.7,
    },
    QueryType.TEMPORAL_QUERY: {
        RetrieverKind.TIMELINE: 1.3,
        RetrieverKind.EVENT: 1.0,
        RetrieverKind.GRAPH: 1.1,
        RetrieverKind.VECTOR: 0.8,
        RetrieverKind.ENTITY_STATE: 0.9,
        RetrieverKind.RAW_TURN: 0.7,
        RetrieverKind.ITERATIVE: 0.7,
    },
    QueryType.RELATIONAL_CHAIN: {
        RetrieverKind.ITERATIVE: 1.2,
        RetrieverKind.GRAPH: 1.2,
        RetrieverKind.EVENT: 1.0,
        RetrieverKind.VECTOR: 1.0,
        RetrieverKind.ENTITY_STATE: 0.8,
        RetrieverKind.RAW_TURN: 0.7,
        RetrieverKind.TIMELINE: 0.7,
    },
    QueryType.THEMATIC_SUMMARY: {
        RetrieverKind.VECTOR: 1.3,
        RetrieverKind.GRAPH: 1.0,
        RetrieverKind.RAW_TURN: 1.0,
        RetrieverKind.TIMELINE: 0.8,
        RetrieverKind.EVENT: 0.7,
        RetrieverKind.ENTITY_STATE: 0.7,
        RetrieverKind.ITERATIVE: 0.7,
    },
    QueryType.PROCEDURAL_RECALL: {
        RetrieverKind.RAW_TURN: 1.2,
        RetrieverKind.GRAPH: 0.9,
        RetrieverKind.VECTOR: 1.0,
        RetrieverKind.ENTITY_STATE: 0.7,
        RetrieverKind.TIMELINE: 0.7,
        RetrieverKind.ITERATIVE: 0.7,
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
    # Cross-source fusion strategy. "weighted" uses the per-kind min-max
    # WeightedFuser; "rrf" uses the rank-based ReciprocalRankFuser, which is
    # robust to incomparable retriever score scales and is the default.
    fusion_strategy: Literal["weighted", "rrf"] = "rrf"
    # Query types that keep the WeightedFuser even when fusion_strategy="rrf".
    # RRF is score-agnostic and rewards cross-source agreement, which buries
    # single-source high-confidence gold. Temporal answers live in one
    # timeline/event fact whose date is not redundantly attested, so they
    # stay on the magnitude-preserving weighted path.
    rrf_weighted_query_types: frozenset[QueryType] = frozenset({QueryType.TEMPORAL_QUERY})


class RecallOrchestrator:
    """Coordinate routing, retrieval, fusion, and answer guarding."""

    def __init__(
        self,
        *,
        router: QueryRouter,
        retrievers: Mapping[str, Retriever],
        fuser: Fuser | None = None,
        deduplicator: MMRDeduplicator | None = None,
        reranker: Reranker | None = None,
        guard: IDKGuard | None = None,
        config: RecallPipelineConfig | None = None,
        emitter: MemoryEventEmitter | None = None,
        enum_booster: EnumerationBooster | None = None,
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
        # Optional enumeration-family booster. When present, aggregation
        # queries get category-family candidates boosted before fusion so
        # the bounded candidate budget covers the whole family instead of
        # an arbitrary lexical sample.
        self._enum_booster = enum_booster
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
        runtime.query_type = route.query_type
        retriever_names = list(self._config.route_table.get(route.query_type, ()))
        retrievers = [
            self._retrievers[name] for name in retriever_names if name in self._retrievers
        ]

        trace: dict[str, object] = {
            "route": route.model_dump(),
            "retrievers": retriever_names,
            "errors": [],
            "debug_trace": runtime.debug_trace,
        }
        raw_candidates = await self._retrieve_all(query, runtime, retrievers, trace)
        if runtime.debug_trace:
            trace["dbg_raw"] = _dbg_snapshot(raw_candidates)
        if self._enum_booster is not None:
            boosted = await self._enum_booster.apply(query.text, raw_candidates)
            if boosted:
                trace["enumeration_boosted"] = boosted
        query_entities = _infer_query_entities(query)
        is_enumeration = detect_enumeration_category(query.text) is not None
        answer_types = _infer_answer_type(query.text)
        deduped = await self._rank_candidates(
            route.query_type,
            raw_candidates,
            top_k=query.top_k,
            trace=trace,
            query_entities=query_entities,
            is_enumeration=is_enumeration,
            answer_types=answer_types,
            query_text=query.text,
        )
        result = self._guard.evaluate(
            query_type=route.query_type,
            candidates=deduped,
            trace=trace,
        )
        self._emit_outcome(query, route.query_type, result, candidates_seen=len(deduped))
        result.candidates = self._merge_siblings_post_fusion(
            result.candidates, is_enumeration=is_enumeration
        )
        return result

    def _merge_siblings_post_fusion(
        self, candidates: list[RecallCandidate], *, is_enumeration: bool = False
    ) -> list[RecallCandidate]:
        """Perform presentation-layer sibling merge for final candidates.

        This dynamically merges candidates with the same (subject, predicate)
        into a single compound candidate, preserving all individual scores,
        original anchors, and reducing final token space presented to the Answerer.
        """
        import copy
        import hashlib
        from collections import defaultdict

        if not candidates:
            return candidates

        # Group candidates by (subject, predicate)
        groups = defaultdict(list)
        for cand in candidates:
            group_key = (
                cand.fact.subject.strip().casefold(),
                cand.fact.predicate.strip().casefold(),
            )
            groups[group_key].append(cand)

        final_candidates = []
        processed_groups = set()

        for cand in candidates:
            group_key = (
                cand.fact.subject.strip().casefold(),
                cand.fact.predicate.strip().casefold(),
            )
            group_cands = groups[group_key]

            should_merge = len(group_cands) >= 2 and (
                is_enumeration or any(c.fact.accumulate for c in group_cands)
            )

            if should_merge:
                if group_key not in processed_groups:
                    # Select the highest-scoring candidate as representative based on final rerank/fusion scores
                    best_cand = max(
                        group_cands,
                        key=lambda c: float(
                            c.signals.get("rerank_score", c.signals.get("fused_score", c.score))
                        ),
                    )
                    all_objects = []
                    all_anchors = []
                    for c in group_cands:
                        if c.fact.object not in all_objects:
                            all_objects.append(c.fact.object)
                        if c.fact.source_anchor and c.fact.source_anchor not in all_anchors:
                            all_anchors.append(c.fact.source_anchor)

                    representative = RecallCandidate(
                        fact=best_cand.fact.model_copy(
                            update={
                                "object": ", ".join(all_objects),
                            }
                        ),
                        score=best_cand.score,
                        matched_by=best_cand.matched_by,
                        retriever_name=best_cand.retriever_name,
                        signals=copy.deepcopy(best_cand.signals),
                        explanation=best_cand.explanation,
                    )
                    best_anchor = best_cand.fact.source_anchor or ""
                    best_plain = f"{best_cand.fact.subject}|{best_cand.fact.predicate}|{best_cand.fact.object}|{best_anchor}"
                    best_digest = hashlib.sha256(best_plain.encode()).hexdigest()[:24]
                    representative.signals["original_memory_id"] = f"fact:{best_digest}"
                    representative.signals["compound_members"] = all_objects
                    representative.signals["compound_source_anchors"] = all_anchors
                    representative.signals["compound_group_key"] = group_key
                    representative.signals["compound_size"] = len(group_cands)
                    final_candidates.append(representative)
                    processed_groups.add(group_key)
            else:
                # Keep them separate, preserving original order exactly
                # Still tag single-member candidates so orchestrator subsumption works properly
                cand.signals = dict(cand.signals)  # Prevent leakage by shallow copying signals
                cand.signals["compound_members"] = [cand.fact.object]
                cand.signals["compound_group_key"] = group_key
                cand.signals["compound_size"] = 1
                final_candidates.append(cand)

        return final_candidates

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
                # The evolution control plane replays failing queries to judge
                # whether an evolved memory would have made them retrievable, so
                # the query text must travel with the failure signal.
                "query_preview": query.text[:200],
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
        query_entities: list[str] | None = None,
        is_enumeration: bool = False,
        answer_types: frozenset[str] = frozenset(),
        query_text: str | None = None,
    ) -> list[RecallCandidate]:
        # Enumeration coverage needs a wide pre-MMR pool: member facts sit
        # at the same lexical floor score, so a narrow fusion_k would drop
        # them by arbitrary tie-order before diversity selection runs.
        if is_enumeration:
            fusion_k = max(top_k * self._config.rerank_multiplier, _ENUMERATION_FUSION_FLOOR)
        else:
            fusion_k = max(top_k * self._config.rerank_multiplier, top_k)
        fuser = self._fuser_for(query_type)
        fused = fuser.fuse(candidates, top_k=fusion_k)
        ranked = await self._reranker.arerank(
            query_type=query_type,
            candidates=fused,
            top_k=fusion_k,
            query=query_text,
        )
        # Entity relevance boost: candidates whose subject matches ANY
        # entity named in the query get an equal score boost so they
        # survive MMR deduplication. Without it, cross-entity score
        # stacking (e.g. Andrew's facts outranking Audrey's when the
        # question is about Audrey) squeezes out answer-relevant facts.
        # Boosting EVERY query entity equally is essential for multi-
        # entity questions ("what do A and B share"): boosting only the
        # first entity starves the second and zeroes its recall.
        if query_entities:
            _apply_entity_relevance_boost(ranked, query_entities)
        # Answer-type boost: lift facts that carry the evidence type the
        # question asks for (a date for "when/which year", a number for
        # "how many"). Stacks additively with the entity boost so a fact
        # that is both about the right entity AND carries the answer type
        # rises above same-entity rephrasings that lack the qualifier.
        if answer_types:
            _apply_answer_type_boost(ranked, answer_types, query_entities)
        # Precise-date boost: when the question itself names a date (e.g.
        # "on March 16, 2022"), a fact whose fact-time exactly matches that
        # date is the strongest evidence regardless of answer type. Unlike
        # the temporal answer-type boost (which lifts any dated fact for
        # "when" questions), this gates on an exact date match so a fact
        # dated to the asked instant rises above unrelated same-entity facts
        # that merely carry some other date. Applies to any date-bearing
        # question, not just "when".
        # Diversity-aware final cut: rerank scores wide (fusion_k), then let
        # MMR pick the top_k. Enumeration queries use coverage diversity so
        # the budget spreads across distinct member facts.
        if is_enumeration:
            _apply_sibling_boost(ranked)
            final_k = top_k * 2
        else:
            final_k = top_k
        diversity = _ENUMERATION_DIVERSITY if is_enumeration else None
        deduped = self._dedupe.dedupe(ranked, top_k=final_k, diversity=diversity)
        if trace.get("debug_trace"):
            # Per-stage candidate snapshot so callers can trace where a gold
            # fact drops (raw -> fused -> reranked -> final) without poking
            # private methods. Gated on RetrieverContext.debug_trace.
            trace["dbg_fused"] = _dbg_snapshot(fused)
            trace["dbg_reranked"] = _dbg_snapshot(ranked)
            trace["dbg_final"] = _dbg_snapshot(deduped)
            deduped_ids = {id(c) for c in deduped}
            trace["dbg_mmr_dropped"] = _dbg_snapshot(
                [c for c in ranked if id(c) not in deduped_ids]
            )
        trace["rerank"] = {
            "reranker": type(self._reranker).__name__,
            "input_count": len(fused),
            "output_count": len(deduped),
        }
        if query_entities:
            trace["query_entities"] = query_entities
        if is_enumeration:
            trace["enumeration_coverage"] = True
        return deduped

    def _fuser_for(self, query_type: QueryType) -> Fuser:
        # An explicitly injected non-default fuser always wins.
        if not isinstance(self._fuser, WeightedFuser):
            return self._fuser
        weights = self._config.fusion_weights.get(query_type)
        kind_weights = dict(weights) if weights is not None else None
        use_rrf = (
            self._config.fusion_strategy == "rrf"
            and query_type not in self._config.rrf_weighted_query_types
        )
        if use_rrf:
            return ReciprocalRankFuser(kind_weights=kind_weights)
        if kind_weights is None:
            return self._fuser
        return WeightedFuser(kind_weights=kind_weights)

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


def _dbg_snapshot(cands: Sequence[RecallCandidate]) -> list[dict[str, object]]:
    """Compact per-candidate snapshot for debug_trace observability.

    Captures the fact identity, source anchor, and the score signals that
    matter for root-causing (fused_score, rerank_score, retriever). Kept
    small (object truncated) so the trace stays cheap to serialize.
    """
    out: list[dict[str, object]] = []
    for c in cands:
        s = c.signals or {}
        out.append(
            {
                "s": c.fact.subject,
                "p": c.fact.predicate,
                "o": str(c.fact.object)[:40],
                "a": c.fact.source_anchor,
                "fs": s.get("fused_score"),
                "rs": s.get("rerank_score"),
                "rn": c.retriever_name,
            }
        )
    return out


def _apply_entity_relevance_boost(
    candidates: list[RecallCandidate],
    entities: list[str],
    boost: float = 1.5,
) -> None:
    """Add an equal rerank_score boost to every query-entity candidate.

    MMR selects candidates by score, not position. Reordering does not
    change which ones MMR picks — it prefers high-score facts. The root
    fix is to boost candidates whose subject matches ANY query entity so
    they compete with cross-entity score-stacked facts.

    Crucially, every query entity is boosted by the SAME amount. For
    multi-entity questions ("what do A and B share") boosting only the
    first entity starves the second and zeroes its recall; equal boosts
    keep both entities' facts in contention so the diversity selector
    can cover both.

    The boost amount (1.5) overcomes the typical score gap between a
    single-retriever entity-state fact (~1.0) and a multi-retriever
    score-stacked fact, giving entity facts enough margin to
    survive MMR selection.
    """
    targets = {e.lower() for e in entities if e and e.strip()}
    if not targets:
        return
    for candidate in candidates:
        if candidate.fact.subject.lower() in targets:
            candidate.signals = dict(candidate.signals)
            old_score = float(candidate.signals.get("rerank_score", candidate.score))
            candidate.signals["rerank_score"] = old_score + boost
            candidate.signals["entity_relevance_boost"] = boost


def _infer_answer_type(text: str) -> frozenset[str]:
    """Classify the answer type a question asks for.

    Returns a set drawn from {'temporal', 'numeric'}. Empty for questions
    whose answer type is not a date or a count (most 'what/who/where'
    lookups), in which case no answer-type boost is applied and ranking
    falls back to entity relevance plus rerank score.
    """
    types: set[str] = set()
    if _ANSWER_TYPE_TEMPORAL_RE.search(text):
        types.add("temporal")
    if _ANSWER_TYPE_NUMERIC_RE.search(text):
        types.add("numeric")
    return frozenset(types)


def _fact_has_time(fact: object) -> bool:
    """True when a fact carries an explicit time, via event_time or a
    date/time qualifier. These are the facts that can answer a 'when' query.
    """
    if getattr(fact, "event_time", None):
        return True
    quals = getattr(fact, "qualifiers", None) or {}
    return any(key in quals for key in ("date", "time", "when"))


def _fact_has_number(fact: object) -> bool:
    """True when a fact's object contains a digit, the minimal signal that
    it can answer a 'how many/how much' query.
    """
    obj = getattr(fact, "object", "") or ""
    return bool(_DIGIT_RE.search(obj))


def _apply_sibling_boost(candidates: list[RecallCandidate]) -> None:
    from collections import defaultdict

    groups = defaultdict(list)
    for c in candidates:
        key = (c.fact.subject.strip().casefold(), c.fact.predicate.strip().casefold())
        groups[key].append(c)

    for group in groups.values():
        if len(group) > 1:
            # Boost siblings so they stay together in top-k
            for c in group:
                score = float(c.signals.get("rerank_score", c.signals.get("fused_score", c.score)))
                c.signals["rerank_score"] = score + 0.4

    candidates.sort(
        key=lambda c: float(c.signals.get("rerank_score", c.signals.get("fused_score", c.score))),
        reverse=True,
    )


def _apply_answer_type_boost(
    candidates: list[RecallCandidate],
    answer_types: frozenset[str],
    query_entities: list[str] | None,
    boost: float = _ANSWER_TYPE_BOOST,
) -> None:
    """Boost facts that BOTH belong to a query entity AND carry the
    evidence type the question asks for (a date for 'when/which year', a
    number for 'how many').

    The entity gate is essential. An ungated temporal boost lifts every
    dated fact, including ones about other people: for 'when did Deborah's
    mother pass away' it would elevate an unrelated 'Jolene lost mother
    (2022)' over the relevant but undated 'Deborah lost mother', adding
    temporal noise that makes the answerer abstain. Gating to query
    entities keeps the boost a within-entity tiebreaker that favours the
    dated member fact without overriding cross-entity relevance. When no
    query entity is identified (broad/thematic questions) the boost is
    skipped entirely rather than applied blindly.
    """
    if not answer_types:
        return
    targets = {e.lower() for e in (query_entities or []) if e and e.strip()}
    if not targets:
        return
    for candidate in candidates:
        fact = candidate.fact
        if fact.subject.lower() not in targets:
            continue
        matched = ("temporal" in answer_types and _fact_has_time(fact)) or (
            "numeric" in answer_types and _fact_has_number(fact)
        )
        if matched:
            candidate.signals = dict(candidate.signals)
            old_score = float(candidate.signals.get("rerank_score", candidate.score))
            candidate.signals["rerank_score"] = old_score + boost
            candidate.signals["answer_type_boost"] = boost


def _infer_query_entities(query: RecallQuery) -> list[str]:
    """Derive the entities the query is about, if identifiable.

    Uses caller-supplied entity_hint when available, otherwise applies
    lightweight heuristics to extract likely entity names from the
    question text. Returns ALL detected entities (one for single-entity
    questions, two-plus for "what do A and B share"). Returns an empty
    list for broad/thematic questions where no entity dominates.
    """
    if query.entity_hint:
        return [query.entity_hint.strip()]
    text = query.text.strip()
    # Skip thematic/broad questions that have no clear entity anchor.
    broad_words = {"many", "often", "all", "kinds", "types", "list", "field", "fields"}
    query_lower = text.lower()
    if any(w in query_lower for w in broad_words) and not query.entity_hint:
        # These often require cross-entity synthesis with no anchor entity.
        return []

    # Extract capitalized words (likely entity names), excluding common
    # question words, months, and auxiliaries.
    _skip = {
        "what",
        "which",
        "when",
        "where",
        "who",
        "how",
        "why",
        "did",
        "does",
        "has",
        "is",
        "was",
        "were",
        "are",
        "can",
        "the",
        "a",
        "an",
        "this",
        "that",
        "his",
        "her",
        "their",
        "my",
        "your",
        "our",
        "first",
        "last",
        "new",
        "old",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
    words = re.findall(r"[A-Z][a-z]+(?:'[a-z]+)?", text)
    entities: list[str] = []
    seen: set[str] = set()
    for w in words:
        if w.lower() in _skip or len(w) < 2:
            continue
        if w.lower() in seen:
            continue
        seen.add(w.lower())
        entities.append(w)
    return entities


__all__ = ["RecallOrchestrator", "RecallPipelineConfig"]
