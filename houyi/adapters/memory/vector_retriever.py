"""Two-stage vector retriever: FTS5 prefilter → vector rerank.

Pipeline:

1. Prefilter the candidate set with the FTS5 BM25 index. The
 sanitizer keeps the MATCH expression bounded, and the
 query is honored regardless of language because FTS5 runs the
 unicode61 tokenizer.
2. Rerank the prefiltered set with a vector similarity call against
 MemoryBackend.search_vector, which uses the sqlite-vec
 vec0 virtual table when present and silently falls back to a
 Python cosine pass otherwise.

This class is intentionally lower-level than the recall-layer
Retriever ABC: it returns (MemoryRecord, score) rather than
RecallCandidate so it can serve both the legacy
MemoryRetriever path and the recall orchestrator's future
VectorRetriever[recall] wrapper without forcing the
AtomicFact 6-tuple mapping that recall-layer retrievers require.

Stateless except for backend / provider handles, mirroring the recall
retriever convention.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from houyi.adapters.embedding import EmbeddingProvider
from houyi.adapters.memory.backends.base import MemoryBackend
from houyi.adapters.memory.types import MemoryRecord, MemoryScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorRetrieverConfig:
    """Runtime knobs for the two-stage retriever.

    Tunables are exposed here rather than as constructor kwargs so callers
    can build a single config and reuse it across many retrievers (e.g. a
    process-wide policy snapshot fed by EnvConfig).
    """

    prefilter_top_k: int = 200
    """Number of FTS5 candidates handed to the vector rerank stage.

 The budget (≤ 200) keeps the rerank latency bounded for ANN
 over even the largest in-memory stores we model.
 """

    min_prefilter_hits: int = 1
    """Skip prefilter and go straight to vector search when FTS5 yields
 fewer than this many hits. 1 reproduces the design intent: any
 BM25 signal at all is preferred over the global vector sweep.
 """

    vector_top_k: int = 20
    """How many final (record, similarity) rows the retriever returns
 before any downstream fusion / dedupe.
 """


class VectorRetriever:
    """Two-stage FTS5 prefilter + vector rerank retriever.

    Construction takes the storage backend and an embedding provider.
    The retriever holds no per-call state; retrieve is safe to call
    concurrently provided the backend itself is thread-safe (which
    SQLiteMemoryBackend is, via check_same_thread=False +
    thread-local connections).
    """

    def __init__(
        self,
        backend: MemoryBackend,
        embedding_provider: EmbeddingProvider,
        *,
        config: VectorRetrieverConfig | None = None,
    ) -> None:
        if backend is None:
            raise ValueError("backend is required")
        if embedding_provider is None:
            raise ValueError("embedding_provider is required")
        self._backend = backend
        self._provider = embedding_provider
        self._config = config or VectorRetrieverConfig()

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        scope: MemoryScope | None = None,
        top_k: int | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Return ranked (record, similarity) pairs for query.

        Algorithm:

        1. Sanitize the query and run MemoryBackend.search_fts
        with the configured prefilter_top_k. The result is a
        candidate rowid set bounded by .
        2. Embed the query through the injected
        EmbeddingProvider; the call is awaited so providers
        that hit the network (SiliconFlow) interleave with other
        coroutines naturally.
        3. Hand the rowid set to MemoryBackend.search_vector as
        rowid_filter; the backend chooses vec0 or Python cosine.

        When the prefilter set is below
        VectorRetrieverConfig.min_prefilter_hits (default: empty),
        step 3 runs unfiltered. This degraded path preserves the legacy
        "vector-only" semantics for queries with no usable tokens (e.g.
        symbol soup, very short questions).
        """
        wanted = top_k or self._config.vector_top_k

        prefilter_rowids = await self._collect_prefilter_rowids(query, scope=scope)

        embeddings = await self._provider.embed([query])
        if not embeddings or not embeddings[0]:
            return []
        query_vec = embeddings[0]

        rowid_filter = (
            prefilter_rowids
            if prefilter_rowids is not None
            and len(prefilter_rowids) >= self._config.min_prefilter_hits
            else None
        )
        hits = await asyncio.to_thread(
            self._backend.search_vector,
            query_vec,
            scope=scope,
            rowid_filter=rowid_filter,
            limit=wanted,
        )

        # Soft fallback: if we applied FTS prefiltering but got weak/insufficient results (top similarity < 0.8),
        # run a global unfiltered search to rescue semantic recall from FTS token mismatches.
        if rowid_filter is not None and (not hits or len(hits) < wanted or hits[0][1] < 0.8):
            global_hits = await asyncio.to_thread(
                self._backend.search_vector,
                query_vec,
                scope=scope,
                rowid_filter=None,
                limit=wanted,
            )
            seen_ids = {rec.record_id for rec, _ in hits}
            for rec, score in global_hits:
                if rec.record_id not in seen_ids:
                    hits.append((rec, score))
            hits = sorted(hits, key=lambda x: x[1], reverse=True)[:wanted]

        return hits

        # ------------------------------------------------------------------
        # Stage 1: FTS5 prefilter
        # ------------------------------------------------------------------

    async def _collect_prefilter_rowids(
        self,
        query: str,
        *,
        scope: MemoryScope | None,
    ) -> list[int] | None:
        """Run the FTS5 prefilter and convert hits to a rowid list.

        Returns None (rather than an empty list) when the query has
        no usable tokens, so the caller can distinguish "explicit empty
        prefilter" (── empty list ── means "veto every candidate") from
        "no prefilter at all" (None means "skip stage 1, search globally").
        """
        rowid_lookup = getattr(self._backend, "_rowid_for", None)
        if rowid_lookup is None:
            # Backends without rowid introspection still work; we just
            # cannot project FTS5 hits to a rowid filter so we degrade to
            # the unfiltered vector path. This branch should never trip
            # for SQLiteMemoryBackend but keeps the type contract
            # open for future backends.
            return None

        fts_hits = await asyncio.to_thread(
            self._backend.search_fts,
            query,
            scope,
            self._config.prefilter_top_k,
        )
        if not fts_hits:
            return None

        rowids: list[int] = []
        for record, _score in fts_hits:
            rowid = rowid_lookup(record.scope, record.key)
            if rowid is not None:
                rowids.append(rowid)
        return rowids


__all__ = ["VectorRetriever", "VectorRetrieverConfig"]
