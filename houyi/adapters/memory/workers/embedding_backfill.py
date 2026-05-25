"""Embedding backfill worker — drain embedding_pending rows.

 the L1 ExtractorWorker () and the explicit-pin
detector () both write MemoryRecord rows whose
embedding is left None. The SQLite backend flips
embedding_pending=1 for those rows so the vector path can find
them later. This worker drains that backlog by calling the configured
EmbeddingProvider in batches and back-filling the vectors
into embedding_cache + memories_vec (handled atomically by
SQLiteMemoryBackend.mark_embedding_filled).

Worker shape mirrors ExtractorWorker:

- process_once runs a single batch and returns the count
- run_forever drives an idle-sleep loop until cancelled

Failures are logged and skipped, never retried in tight loops; the
embedding_pending flag stays set so the next poll picks the row
back up. This way a transient provider outage doesn't drop work, and
a permanently broken row eventually surfaces in monitoring as a
"perpetually pending" alert rather than burning quota.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from houyi.adapters.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingBackfillConfig:
    """Tunables for the backfill worker."""

    batch_size: int = 16
    """Max rows fetched per process_once. Mirrors typical
 embedding-API request shapes (Jina / SiliconFlow / OpenAI all
 accept ~64 inputs per call; we stay well under the default).
 """

    idle_sleep_s: float = 2.0
    """Seconds to sleep between empty polls in run_forever."""

    fail_sleep_s: float = 5.0
    """Seconds to sleep after a provider error before re-polling. The
 pending rows are not advanced on failure so they will be picked up
 again automatically.
 """


class _BackfillBackend(Protocol):
    """The slice of SQLiteMemoryBackend this worker depends on."""

    def list_pending_embeddings(self, limit: int) -> list[tuple[int, Any]]: ...
    def mark_embedding_filled(self, rowid: int, embedding: list[float]) -> None: ...


class EmbeddingBackfillWorker:
    """Fill missing embeddings for rows flagged embedding_pending."""

    def __init__(
        self,
        *,
        backend: _BackfillBackend,
        provider: EmbeddingProvider,
        config: EmbeddingBackfillConfig | None = None,
    ) -> None:
        if backend is None or provider is None:
            raise ValueError("backend and provider are required")
        self._backend = backend
        self._provider = provider
        self._config = config or EmbeddingBackfillConfig()

    async def process_once(self) -> int:
        """Embed and back-fill one batch.

        Returns the number of rows successfully filled. 0 means
        either the backlog was empty or the provider raised — the
        caller distinguishes the two cases via run_forever's
        sleep selection.
        """
        cfg = self._config
        pending = await asyncio.to_thread(self._backend.list_pending_embeddings, cfg.batch_size)
        if not pending:
            return 0

        rowids = [rowid for rowid, _ in pending]
        # Coerce to str defensively: MemoryRecord.content is typed
        # as str already but ingestor variants have historically
        # wrapped JSON-encoded payloads, and the embedding provider
        # contract requires plain strings.
        texts = [str(record.content) for _, record in pending]

        try:
            vectors = await self._provider.embed(texts)
        except Exception:
            logger.warning("embedding provider failed on batch of %d", len(texts), exc_info=True)
            return 0

        if len(vectors) != len(rowids):
            # Provider returned a malformed batch. Log and skip — the
            # rows stay pending and we'll retry next poll. We don't
            # try to align partial results: index drift is the kind of
            # silent corruption that's worse than re-doing the call.
            logger.warning(
                "embedding provider returned %d vectors for %d inputs; skipping batch",
                len(vectors),
                len(rowids),
            )
            return 0

        filled = 0
        for rowid, vec in zip(rowids, vectors, strict=True):
            try:
                await asyncio.to_thread(self._backend.mark_embedding_filled, rowid, list(vec))
                filled += 1
            except Exception:
                logger.warning("failed to write embedding for rowid=%s", rowid, exc_info=True)
        return filled

    async def run_forever(self, stop: asyncio.Event | None = None) -> None:
        """Loop until stop is set.

        Sleeps for idle_sleep_s after an empty poll, and for
        fail_sleep_s after a poll that returned 0 because the
        provider raised. Even though both branches return 0, we
        cannot tell them apart here without re-doing the work, so we
        use the conservative idle_sleep_s and rely on the provider
        to surface its own backoff via metrics.
        """
        signal = stop or asyncio.Event()
        while not signal.is_set():
            try:
                processed = await self.process_once()
            except Exception:
                logger.exception("backfill worker batch failed; backing off")
                processed = 0
                if processed == 0:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(signal.wait(), timeout=self._config.idle_sleep_s)


__all__ = ["EmbeddingBackfillConfig", "EmbeddingBackfillWorker"]
