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
        """Retrieve top-K relevant memories. Target p95 < 100ms."""
        ctx = session_context or SessionContext()
        top_k = min(top_k, self._policy.max_recalls_per_turn)

        candidates = self._gather_candidates(ctx)
        if not candidates:
            return []

        query_emb = None
        if self._embedding:
            embs = await self._embedding.embed([query])
            query_emb = embs[0] if embs else None

        fts_scores = self._fts_lookup(query)

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

    def _gather_candidates(self, ctx: SessionContext) -> list[MemoryRecord]:
        """Collect non-expired records from prioritized scopes."""
        all_records: list[MemoryRecord] = []
        for scope in self._policy.scope_priority:
            all_records.extend(self._store.list_by_scope(scope))
        now = time.time()
        return [r for r in all_records if not (r.valid_to is not None and now > r.valid_to)]

    def _fts_lookup(self, query: str) -> dict[str, float]:
        """Pre-compute FTS5 BM25 scores keyed by record_id."""
        if self._backend is None:
            return {}
        try:
            hits = self._backend.search_fts(query, limit=100)
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
