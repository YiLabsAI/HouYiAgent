"""Tests for the embedding backfill worker that drains pending-embedding rows."""

from __future__ import annotations

import asyncio

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.fact_promoter import MemoryRecordPromoter
from houyi.adapters.memory.types import (
    AtomicFact,
    Certainty,
    MemoryRecord,
    RawTurn,
)
from houyi.adapters.memory.workers import (
    EmbeddingBackfillConfig,
    EmbeddingBackfillWorker,
)


class _FakeProvider:
    """Returns deterministic 4-d vectors and counts calls."""

    def __init__(self, *, dim: int = 4, raise_for: str | None = None):
        self._dim = dim
        self.calls: list[list[str]] = []
        self.raise_for = raise_for

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.raise_for and any(self.raise_for in t for t in texts):
            raise RuntimeError("provider down")
        return [[float(i)] * self._dim for i, _ in enumerate(texts, start=1)]


class _MalformedProvider:
    """Returns a truncated vector list to exercise the length guard."""

    def dimension(self) -> int:
        return 4

    async def embed(self, texts):
        return [[1.0, 1.0, 1.0, 1.0]] * (len(texts) - 1)


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteMemoryBackend(db_path=tmp_path / "bf.db")
    yield b
    b.close()


def _seed_pending(backend, n: int = 3):
    """Drop n MemoryRecord rows with no embedding into the backend."""
    for i in range(n):
        backend.put(MemoryRecord(key=f"k{i}", content=f"text {i}", embedding=None))


class TestProcessOnce:
    async def test_empty_returns_zero(self, backend):
        worker = EmbeddingBackfillWorker(backend=backend, provider=_FakeProvider())
        assert await worker.process_once() == 0

    async def test_fills_pending_rows(self, backend):
        _seed_pending(backend, n=3)
        provider = _FakeProvider()
        worker = EmbeddingBackfillWorker(backend=backend, provider=provider)
        filled = await worker.process_once()
        assert filled == 3
        # No more pending rows.
        assert backend.list_pending_embeddings(limit=10) == []
        # Provider was called exactly once with all three texts.
        assert provider.calls == [["text 0", "text 1", "text 2"]]

    async def test_respects_batch_size(self, backend):
        _seed_pending(backend, n=5)
        worker = EmbeddingBackfillWorker(
            backend=backend,
            provider=_FakeProvider(),
            config=EmbeddingBackfillConfig(batch_size=2),
        )
        first = await worker.process_once()
        second = await worker.process_once()
        third = await worker.process_once()
        assert first == 2
        assert second == 2
        assert third == 1
        assert backend.list_pending_embeddings(limit=10) == []

    async def test_provider_failure_pending(self, backend):
        _seed_pending(backend, n=2)
        worker = EmbeddingBackfillWorker(backend=backend, provider=_FakeProvider(raise_for="text"))
        filled = await worker.process_once()
        assert filled == 0
        # Rows are still pending — will be retried next poll.
        assert len(backend.list_pending_embeddings(limit=10)) == 2

    async def test_malformed_provider_response_skipped(self, backend):
        _seed_pending(backend, n=3)
        worker = EmbeddingBackfillWorker(backend=backend, provider=_MalformedProvider())
        filled = await worker.process_once()
        assert filled == 0
        # Rows remain pending; the next provider with correct shape
        # will succeed.
        assert len(backend.list_pending_embeddings(limit=10)) == 3


class TestRunForever:
    async def test_drains_then_idles(self, backend):
        _seed_pending(backend, n=2)
        worker = EmbeddingBackfillWorker(
            backend=backend,
            provider=_FakeProvider(),
            config=EmbeddingBackfillConfig(batch_size=10, idle_sleep_s=0.05),
        )
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run_forever(stop))
        # Poll until the backlog is drained instead of a fixed sleep.
        for _ in range(20):
            if backend.list_pending_embeddings(limit=10) == []:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert backend.list_pending_embeddings(limit=10) == []


class TestEndToEndPipeline:
    async def test_promoter_backfill_roundtrip(self, backend):
        # Pin-style flow: promoter writes a deferred-embedding row…
        promoter = MemoryRecordPromoter(backend)
        turn = RawTurn(session_id="s", role="user", content="alice likes tea")
        backend.append_raw_turn(turn)
        promoter.promote(
            turn,
            AtomicFact(
                subject="alice",
                predicate="likes",
                object="tea",
                certainty=Certainty.CERTAIN,
                source_anchor=turn.turn_id,
            ),
        )
        assert len(backend.list_pending_embeddings(limit=10)) == 1

        # …backfill worker fills the embedding.
        worker = EmbeddingBackfillWorker(backend=backend, provider=_FakeProvider())
        filled = await worker.process_once()
        assert filled == 1
        assert backend.list_pending_embeddings(limit=10) == []


class TestConstruction:
    def test_requires_backend_and_provider(self, backend):
        with pytest.raises(ValueError):
            EmbeddingBackfillWorker(backend=None, provider=_FakeProvider())
            with pytest.raises(ValueError):
                EmbeddingBackfillWorker(backend=backend, provider=None)
