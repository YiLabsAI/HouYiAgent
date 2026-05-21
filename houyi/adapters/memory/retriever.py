"""Memory retriever with hybrid scoring.

Retrieval pipeline:
1. Rule-based filter (scope, tag, ttl, validity window)
2. Lexical scoring (FTS5 BM25 when available, else term overlap)
3. Embedding scoring (cosine similarity, if provider available)
4. Recency scoring (exponential time decay)
5. Rule bonus (scope match, tag match, type boost)
6. Weighted merge → final_score
7. Top-K truncation
8. Explanation generation
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import Counter

from houyi.adapters.memory.backends.base import MemoryBackend
from houyi.adapters.memory.embedding import EmbeddingProvider, cosine_similarity
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import (
    MemoryPolicy,
    MemoryRecall,
    MemoryRecord,
    MemoryType,
    RecallMatchMethod,
    RelevanceDetail,
    SessionContext,
)

logger = logging.getLogger(__name__)

_ANN_CAP = 50  # max ANN hits from sqlite-vec
_FTS_CAP = 30  # max BM25 hits from FTS5
_PREFILTER_CAP = 3000  # max rowids considered by ANN after SQL predicate pushdown
_RECENT_WINDOW_SEC = 180 * 86400  # default recency window when query is not historical

_DEFAULT_WEIGHTS = {
    "lexical": 0.25,
    "embedding": 0.35,
    "recency": 0.20,
    "rule": 0.20,
}

_TYPE_BOOST: dict[MemoryType, float] = {
    MemoryType.CONSTRAINT: 0.15,
    MemoryType.PROFILE: 0.10,
    MemoryType.PREFERENCE: 0.05,
}


class MemoryRetriever:
    """Hybrid memory retrieval with scoring and explanation."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
        policy: MemoryPolicy | None = None,
    ):
        self._store = store
        self._embedding = embedding_provider
        self._policy = policy or MemoryPolicy()
        self._backend: MemoryBackend | None = getattr(store, "_backend", None)

    async def retrieve(
        self,
        query: str,
        session_context: SessionContext | None = None,
        top_k: int = 5,
    ) -> list[MemoryRecall]:
        """Retrieve top-K relevant memories. Target p95 < 100ms.

        Two-stage pipeline when both backend and embedding_provider are present:
        1. ANN pre-filter via sqlite-vec (or Python cosine fallback) — narrows
           the candidate set without scanning every record in memory.
        2. Full hybrid scoring (lexical + embedding + recency + rule) over the
           narrowed candidate set.

        When no embedding_provider is configured the pipeline degrades
        gracefully to FTS-only scoring over a bounded candidate set.
        All SQLite IO is dispatched through asyncio.to_thread so the event
        loop is never blocked by synchronous database calls.

        Candidate set size is O(_ANN_CAP + _FTS_CAP) — it does
        NOT grow with total record count, making recall latency independent of
        corpus size.
        """
        ctx = session_context or SessionContext()
        top_k = min(top_k, self._policy.max_recalls_per_turn)

        # --- Step 1: embed the query (IO-bound HTTP, non-blocking) ---
        query_emb: list[float] | None = None
        if self._embedding is not None:
            try:
                embs = await self._embedding.embed([query])
                query_emb = embs[0] if embs else None
            except Exception:
                logger.warning("embedding provider failed during retrieve; falling back to lexical")

        # --- Step 2: bounded candidate assembly (all SQLite IO → thread) ---
        candidates = await self._gather_bounded_candidates(query, query_emb, ctx)
        if not candidates:
            return []

        # --- Step 3: pre-compute FTS BM25 scores (SQLite IO → thread) ---
        # Scores are already available for the FTS channel records; this call
        # fills in scores for ANN-only records that were not in the FTS result.
        fts_scores = await asyncio.to_thread(self._sync_fts_lookup, query)

        # --- Step 4: full hybrid scoring ---
        scored: list[tuple[MemoryRecord, RelevanceDetail]] = []
        for record in candidates:
            detail = self._score(query, record, query_emb, ctx, fts_scores)
            scored.append((record, detail))

        scored.sort(key=lambda x: x[1].final_score, reverse=True)
        top = scored[:top_k]

        return [
            MemoryRecall(
                memory_id=record.record_id,
                score=detail.final_score,
                matched_by=self._determine_match_method(detail),
                explanation=self._explain(record, detail),
                relevance_detail=detail,
            )
            for record, detail in top
            if detail.final_score > 0.01
        ]

    async def _gather_bounded_candidates(
        self,
        query: str,
        query_emb: list[float] | None,
        ctx: SessionContext,
    ) -> list[MemoryRecord]:
        """Build a bounded candidate set without full-table scan.

        When an embedding vector is available two bounded channels run
        concurrently and their results are merged by deduplication:

        1. ANN channel  — sqlite-vec kNN, capped at _ANN_CAP
        2. FTS channel  — BM25 full-text search, capped at _FTS_CAP

        Without an embedding vector only the FTS channel runs.
        Both channels are bounded so candidate count is O(constant)
        regardless of total corpus size.
        """
        if self._backend is None:
            # No backend: fall back to full scope scan (in-memory store path).
            return await asyncio.to_thread(self._sync_scope_scan, ctx)

        rowid_filter = await self._resolve_rowid_filter(query)
        fts_hits = await asyncio.to_thread(self._sync_fts_by_scopes, query, _FTS_CAP)

        ann_hits: list[tuple[MemoryRecord, float]] = []
        if query_emb is not None:
            ann_hits = await asyncio.to_thread(
                self._backend.search_vector,
                query_emb,
                rowid_filter=rowid_filter,
                limit=_ANN_CAP,
            )
        result = self._merge_candidate_hits(ann_hits, fts_hits)

        now = time.time()
        return [r for r in result if not (r.valid_to is not None and now > r.valid_to)]

    async def _resolve_rowid_filter(self, query: str) -> list[int] | None:
        """Resolve optional SQL prefilter rowids for ANN narrowing."""
        if self._backend is None:
            return None
        prefilter_rowids = getattr(self._backend, "prefilter_rowids", None)
        if not callable(prefilter_rowids):
            return None
        updated_after, updated_before = self._resolve_time_bounds(query)
        rowids = await asyncio.to_thread(
            prefilter_rowids,
            scopes=self._policy.scope_priority,
            updated_after=updated_after,
            updated_before=updated_before,
            limit=_PREFILTER_CAP,
        )
        return rowids or None

    @staticmethod
    def _merge_candidate_hits(
        ann_hits: list[tuple[MemoryRecord, float]],
        fts_hits: list[tuple[MemoryRecord, float]],
    ) -> list[MemoryRecord]:
        """Merge ANN/FTS channels while deduplicating by record_id."""
        seen: set[str] = set()
        out: list[MemoryRecord] = []
        for record, _ in ann_hits:
            if record.record_id in seen:
                continue
            seen.add(record.record_id)
            out.append(record)
        for record, _ in fts_hits:
            if record.record_id in seen:
                continue
            seen.add(record.record_id)
            out.append(record)
        return out

    def _sync_scope_scan(self, ctx: SessionContext) -> list[MemoryRecord]:
        """Full scope scan fallback used only when no backend is available."""
        all_records: list[MemoryRecord] = []
        for scope in self._policy.scope_priority:
            all_records.extend(self._store.list_by_scope(scope))
        now = time.time()
        return [r for r in all_records if not (r.valid_to is not None and now > r.valid_to)]

    def _sync_fts_by_scopes(self, query: str, limit: int) -> list[tuple[MemoryRecord, float]]:
        """Run bounded FTS lookups by scope priority and merge deduplicated hits."""
        if self._backend is None:
            return []
        out: list[tuple[MemoryRecord, float]] = []
        seen: set[str] = set()
        per_scope_limit = max(1, limit)
        for scope in self._policy.scope_priority:
            try:
                hits = self._backend.search_fts(query, scope=scope, limit=per_scope_limit)
            except Exception:
                continue
            for rec, score in hits:
                if rec.record_id in seen:
                    continue
                seen.add(rec.record_id)
                out.append((rec, score))
                if len(out) >= limit:
                    return out
        return out

    @staticmethod
    def _resolve_time_bounds(query: str) -> tuple[float | None, float | None]:
        """Return optional (updated_after, updated_before) for SQL prefilter.

        Historical-intent queries are allowed to search across all history
        (still bounded by _PREFILTER_CAP). Other queries default to a recent
        window to reduce ANN search cost.
        """
        q = query.lower()
        historical_markers = (
            "history",
            "historical",
            "earlier",
            "before",
            "old",
            "previous",
            "long ago",
            "past",
        )
        if any(marker in q for marker in historical_markers):
            return None, None
        now = time.time()
        return now - _RECENT_WINDOW_SEC, None

    def _sync_fts_lookup(self, query: str) -> dict[str, float]:
        """Synchronous FTS5 BM25 pre-scores — call via asyncio.to_thread."""
        if self._backend is None:
            return {}
        try:
            hits = self._sync_fts_by_scopes(query, limit=100)
            return {rec.record_id: score for rec, score in hits}
        except Exception:
            return {}

    def _score(
        self,
        query: str,
        record: MemoryRecord,
        query_emb: list[float] | None,
        ctx: SessionContext,
        fts_scores: dict[str, float],
    ) -> RelevanceDetail:
        if record.record_id in fts_scores:
            lex = min(fts_scores[record.record_id] / 10.0, 1.0)
        else:
            lex = self._lexical_score(query, record)

        emb = self._embedding_score(query_emb, record)
        rec = self._recency_score(record)
        rule = self._rule_bonus(record, ctx)

        w = _DEFAULT_WEIGHTS
        has_emb = query_emb is not None and record.embedding is not None
        if has_emb:
            final = (
                w["lexical"] * lex + w["embedding"] * emb + w["recency"] * rec + w["rule"] * rule
            )
        else:
            redistrib = w["embedding"] / 3.0
            final = (
                (w["lexical"] + redistrib) * lex
                + (w["recency"] + redistrib) * rec
                + (w["rule"] + redistrib) * rule
            )

        return RelevanceDetail(
            lexical_score=lex,
            embedding_score=emb,
            recency_score=rec,
            rule_bonus=rule,
            final_score=min(final, 1.0),
        )

    @staticmethod
    def _lexical_score(query: str, record: MemoryRecord) -> float:
        """BM25-lite: term overlap ratio (fallback when FTS5 unavailable)."""
        q_terms = Counter(_tokenize(query))
        r_terms = Counter(_tokenize(record.content + " " + record.key))
        if not q_terms or not r_terms:
            return 0.0
        overlap = sum((q_terms & r_terms).values())
        return min(overlap / max(len(q_terms), 1), 1.0)

    @staticmethod
    def _embedding_score(
        query_emb: list[float] | None,
        record: MemoryRecord,
    ) -> float:
        if query_emb is None or record.embedding is None:
            return 0.0
        sim = cosine_similarity(query_emb, record.embedding)
        return max(sim, 0.0)

    @staticmethod
    def _recency_score(record: MemoryRecord) -> float:
        """Exponential decay: score = decay * exp(-0.01 * days_since_update)."""
        days = (time.time() - record.updated_at) / 86400.0
        return record.decay * math.exp(-0.01 * max(days, 0))

    @staticmethod
    def _rule_bonus(record: MemoryRecord, ctx: SessionContext) -> float:
        bonus = 0.0
        bonus += _TYPE_BOOST.get(record.memory_type, 0.0)
        if ctx.active_tags and record.tags:
            overlap = set(ctx.active_tags) & set(record.tags)
            bonus += 0.1 * len(overlap)
        return min(bonus, 1.0)

    @staticmethod
    def _determine_match_method(detail: RelevanceDetail) -> RecallMatchMethod:
        if detail.embedding_score > 0.5 and detail.lexical_score > 0.3:
            return RecallMatchMethod.HYBRID
        if detail.embedding_score > 0.5:
            return RecallMatchMethod.EMBEDDING
        if detail.lexical_score > 0.3:
            return RecallMatchMethod.LEXICAL
        return RecallMatchMethod.RULE

    @staticmethod
    def _explain(record: MemoryRecord, detail: RelevanceDetail) -> str:
        parts = []
        if detail.lexical_score > 0.3:
            parts.append(f"lexical={detail.lexical_score:.2f}")
        if detail.embedding_score > 0.3:
            parts.append(f"semantic={detail.embedding_score:.2f}")
        if detail.rule_bonus > 0:
            parts.append(f"bonus={detail.rule_bonus:.2f}")
        reason = ", ".join(parts) if parts else "recency"
        return f"[{record.memory_type.value}] {record.key}: {reason}"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())
