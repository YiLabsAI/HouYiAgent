"""VectorRetriever — two-stage prefilter → vector rerank tests.

 validates the FTS5 prefilter + vector rerank wiring on top of
the SQLite backend, including the global-fallback path when FTS5 returns
no hits.
"""

from __future__ import annotations

import pytest

from houyi.adapters.embedding import NoOpEmbeddingProvider
from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.types import MemoryRecord, MemoryScope
from houyi.adapters.memory.vector_retriever import (
    VectorRetriever,
    VectorRetrieverConfig,
)


@pytest.fixture()
def backend(tmp_path):
    # File-backed DB rather than ":memory:" so the asyncio worker thread
    # spawned via asyncio.to_thread sees the same schema as the test
    # thread. SQLite's ":memory:" is per-connection and would otherwise
    # surface as "no such table: memories_fts" inside the worker.
    b = SQLiteMemoryBackend(db_path=tmp_path / "vec.db")
    yield b
    b.close()


@pytest.fixture()
def provider() -> NoOpEmbeddingProvider:
    return NoOpEmbeddingProvider(dim=32)


async def _populate(backend: SQLiteMemoryBackend, provider: NoOpEmbeddingProvider):
    """Seed three records whose embeddings come from the NoOp provider.

    NoOp produces deterministic hash vectors, so the cosine similarities
    are reproducible and self-similarity is exactly 1.0.
    """
    items = [
        ("python_lang", "Python is great for machine learning"),
        ("rust_lang", "Rust is fast and memory safe"),
        ("recipe", "Carbonara needs eggs pancetta and pasta"),
    ]
    embs = await provider.embed([content for _, content in items])
    for (key, content), emb in zip(items, embs, strict=True):
        backend.put(
            MemoryRecord(
                key=key,
                content=content,
                scope=MemoryScope.USER,
                embedding=emb,
            )
        )


class TestVectorRetriever:
    async def test_two_stage_relevant(self, backend, provider):
        await _populate(backend, provider)
        retriever = VectorRetriever(backend, provider)
        results = await retriever.retrieve("Python machine learning")
        assert results
        # FTS5 prefilter narrows to records containing 'python' /
        # 'machine' / 'learning'; vector rerank ranks the python record
        # at the top via self-similarity.
        assert results[0][0].key == "python_lang"

    async def test_prefilter_misses_global(self, backend, provider):
        await _populate(backend, provider)
        retriever = VectorRetriever(backend, provider)
        # No token in store matches the query — FTS5 returns nothing and
        # we still get vector hits because the retriever falls back to
        # an unfiltered vector pass.
        results = await retriever.retrieve("xyzzy_no_match")
        assert results  # at least some records ranked by vector similarity

    async def test_respects_top_k(self, backend, provider):
        await _populate(backend, provider)
        retriever = VectorRetriever(
            backend,
            provider,
            config=VectorRetrieverConfig(vector_top_k=1),
        )
        results = await retriever.retrieve("Python")
        assert len(results) <= 1

    async def test_scope_filter(self, backend, provider):
        await _populate(backend, provider)
        emb = (await provider.embed(["session python"]))[0]
        backend.put(
            MemoryRecord(
                key="session_py",
                content="session python note",
                scope=MemoryScope.SESSION,
                embedding=emb,
            )
        )
        retriever = VectorRetriever(backend, provider)
        results = await retriever.retrieve("python", scope=MemoryScope.USER)
        keys = [r.key for r, _ in results]
        assert "session_py" not in keys
        assert "python_lang" in keys

    def test_requires_backend_and_provider(self, provider, backend):
        with pytest.raises(ValueError):
            VectorRetriever(None, provider)  # type: ignore[arg-type]
            with pytest.raises(ValueError):
                VectorRetriever(backend, None)  # type: ignore[arg-type]
