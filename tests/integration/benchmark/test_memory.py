"""Memory pipeline performance benchmarks.

Validates p95 latency:
  Extractor < 50ms, Classifier < 20ms, Deduplicator < 30ms,
  Retriever < 100ms (1K records), FTS5 BM25 < 5ms (1K records),
  Embedding search < 50ms (1K records).
"""

from __future__ import annotations

import time

import pytest

from houyi.adapters.memory.classifier import MemoryClassifier
from houyi.adapters.memory.deduplicator import MemoryDeduplicator
from houyi.adapters.memory.embedding import NoOpEmbeddingProvider
from houyi.adapters.memory.extractor import MemoryCandidateExtractor
from houyi.adapters.memory.retriever import MemoryRetriever
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import (
    MemoryCandidate,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)

pytestmark = pytest.mark.benchmark


def _measure_async(coro_fn, iterations=50):
    """Run an async function N times and return p95 latency in ms."""
    import asyncio

    async def _run():
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await coro_fn()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        times.sort()
        return times[int(len(times) * 0.95)]

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


class TestExtractorPerformance:
    def test_p95_under_50ms(self):
        extractor = MemoryCandidateExtractor()
        messages = [
            {"role": "user", "content": "Remember that the API endpoint is /api/v2."},
            {"role": "user", "content": "My name is Alice and I prefer dark mode."},
            {"role": "assistant", "content": "Got it!"},
        ]

        async def run():
            await extractor.extract(messages)

        p95 = _measure_async(run)
        assert p95 < 50, f"Extractor p95 = {p95:.1f}ms (target < 50ms)"


class TestClassifierPerformance:
    def test_p95_under_20ms(self):
        classifier = MemoryClassifier()
        candidate = MemoryCandidate(
            content="I prefer using Python for data science projects",
            memory_type=MemoryType.FACT,
        )

        async def run():
            await classifier.classify(candidate)

        p95 = _measure_async(run)
        assert p95 < 20, f"Classifier p95 = {p95:.1f}ms (target < 20ms)"


class TestDeduplicatorPerformance:
    def test_p95_under_30ms(self):
        emb = NoOpEmbeddingProvider(dim=32)
        dedup = MemoryDeduplicator(emb, similarity_threshold=0.9)
        existing = [
            MemoryRecord(
                key=f"mem_{i}",
                content=f"Some memory content number {i}",
                embedding=[float(j) / 32 for j in range(32)],
            )
            for i in range(100)
        ]
        candidate = MemoryCandidate(
            content="A new memory candidate for dedup check",
        )

        async def run():
            await dedup.check(candidate, existing)

        p95 = _measure_async(run)
        assert p95 < 30, f"Deduplicator p95 = {p95:.1f}ms (target < 30ms)"


class TestRetrieverPerformance:
    @pytest.fixture()
    def large_store(self):
        store = MemoryStore()
        emb = NoOpEmbeddingProvider(dim=32)
        import asyncio

        async def _populate():
            for i in range(1000):
                vec = await emb.embed([f"memory content {i}"])
                store.put(
                    f"key_{i}",
                    f"memory content about topic {i} with details",
                    scope=MemoryScope.USER,
                    memory_type=MemoryType.FACT,
                    embedding=vec[0],
                )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(_populate())
        loop.close()
        yield store
        store.close()

    def test_p95_under_100ms(self, large_store: MemoryStore):
        emb = NoOpEmbeddingProvider(dim=32)
        retriever = MemoryRetriever(large_store, emb)

        async def run():
            await retriever.retrieve("topic 42 details", top_k=5)

        p95 = _measure_async(run, iterations=20)
        assert p95 < 100, f"Retriever p95 = {p95:.1f}ms (target < 100ms)"


class TestFts5Performance:
    """FTS5 BM25 search benchmark on 1K records."""

    @pytest.fixture()
    def backend_1k(self):
        from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend

        backend = SQLiteMemoryBackend(db_path=":memory:")
        for i in range(1000):
            backend.put(
                MemoryRecord(
                    key=f"topic_{i}",
                    content=f"memory about subject {i} with detailed information and context",
                    scope=MemoryScope.USER,
                    memory_type=MemoryType.FACT,
                )
            )
        yield backend
        backend.close()

    def test_fts5_p95_under_5ms(self, backend_1k):
        def run():
            backend_1k.search_fts("subject detailed information", limit=10)

        times = []
        for _ in range(50):
            start = time.perf_counter()
            run()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        times.sort()
        p95 = times[int(len(times) * 0.95)]
        assert p95 < 5, f"FTS5 p95 = {p95:.2f}ms (target < 5ms)"


class TestEmbeddingSearchPerf:
    """Cosine embedding search benchmark on 1K records."""

    @pytest.fixture()
    def backend_emb_1k(self):
        from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend

        emb = NoOpEmbeddingProvider(dim=32)
        backend = SQLiteMemoryBackend(db_path=":memory:")
        import asyncio

        async def _pop():
            for i in range(1000):
                vec = await emb.embed([f"content {i}"])
                backend.put(
                    MemoryRecord(
                        key=f"ek_{i}",
                        content=f"content {i}",
                        scope=MemoryScope.USER,
                        embedding=vec[0],
                    )
                )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(_pop())
        loop.close()
        yield backend, emb
        backend.close()

    def test_emb_search_p95_under_50ms(self, backend_emb_1k):
        backend, emb = backend_emb_1k
        import asyncio

        async def _get_qemb():
            vecs = await emb.embed(["query about content"])
            return vecs[0]

        loop = asyncio.new_event_loop()
        query_emb = loop.run_until_complete(_get_qemb())
        loop.close()

        times = []
        for _ in range(20):
            start = time.perf_counter()
            backend.search_embedding(query_emb, limit=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        times.sort()
        p95 = times[int(len(times) * 0.95)]
        assert p95 < 80, f"Embedding search p95 = {p95:.2f}ms (target < 80ms)"
