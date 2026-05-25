"""MemoryRetriever unit tests.

Covers hybrid scoring, FTS fallback, match method classification,
context filtering, top-K truncation, embedding provider failure
degradation, and async ANN two-stage recall path.
"""

from __future__ import annotations

import time

import pytest

from houyi.adapters.embedding import EmbeddingProvider, NoOpEmbeddingProvider
from houyi.adapters.memory.retriever import MemoryRetriever
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import (
    MemoryPolicy,
    MemoryScope,
    MemoryType,
    RecallMatchMethod,
    SessionContext,
)


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(data_dir=tmp_path)
    s.put("python", "Python is great for ML", scope=MemoryScope.USER, memory_type=MemoryType.FACT)
    s.put("rust", "Rust is fast and safe", scope=MemoryScope.USER, memory_type=MemoryType.FACT)
    s.put("name", "User name: Alice", scope=MemoryScope.USER, memory_type=MemoryType.PROFILE)
    yield s
    s.close()


@pytest.fixture()
def retriever(store) -> MemoryRetriever:
    emb = NoOpEmbeddingProvider(dim=32)
    return MemoryRetriever(store, embedding_provider=emb)


@pytest.fixture()
def retriever_no_emb(store) -> MemoryRetriever:
    return MemoryRetriever(store)


@pytest.fixture()
def store_factory(tmp_path):
    stores: list[MemoryStore] = []

    def _make(*, backend=None) -> MemoryStore:
        if backend is not None:
            s = MemoryStore(backend=backend)
        else:
            s = MemoryStore(data_dir=tmp_path)
        stores.append(s)
        return s

    yield _make

    for s in stores:
        s.close()


class TestBasicRetrieval:
    async def test_returns_results(self, retriever):
        recalls = await retriever.retrieve("Python machine learning")
        assert len(recalls) >= 1
        assert recalls[0].score > 0

    async def test_empty_store(self, store_factory):
        store = store_factory()
        r = MemoryRetriever(store)
        recalls = await r.retrieve("anything")
        assert recalls == []

    async def test_results_sorted_by_score(self, retriever):
        recalls = await retriever.retrieve("programming language")
        scores = [r.score for r in recalls]
        assert scores == sorted(scores, reverse=True)


class TestTopK:
    async def test_respects_top_k(self, retriever):
        recalls = await retriever.retrieve("programming", top_k=1)
        assert len(recalls) <= 1

    async def test_policy_max_recalls(self, store):
        policy = MemoryPolicy(max_recalls_per_turn=1)
        r = MemoryRetriever(store, policy=policy)
        recalls = await r.retrieve("Python Rust Alice", top_k=10)
        assert len(recalls) <= 1


class TestLexicalFallback:
    async def test_lexical_without_embedding(self, retriever_no_emb):
        recalls = await retriever_no_emb.retrieve("Python ML")
        assert len(recalls) >= 1

    async def test_lexical_match_method(self, retriever_no_emb):
        recalls = await retriever_no_emb.retrieve("Python great ML")
        if recalls:
            assert recalls[0].matched_by in (RecallMatchMethod.LEXICAL, RecallMatchMethod.RULE)


class TestScopeFiltering:
    async def test_scope_priority(self, store_factory):
        store = store_factory()
        store.put("a", "session data", scope=MemoryScope.SESSION)
        policy = MemoryPolicy(scope_priority=[MemoryScope.USER])
        r = MemoryRetriever(store, policy=policy)
        recalls = await r.retrieve("session data")
        assert len(recalls) == 0

    async def test_expired_excluded(self, store_factory):
        store = store_factory()
        store.put("live", "Python is great", scope=MemoryScope.SESSION)
        rec = store.put("dead", "stale data", scope=MemoryScope.SESSION, ttl=0.001)
        time.sleep(0.01)
        r = MemoryRetriever(store)
        recalls = await r.retrieve("stale data")
        live_ids = {rc.memory_id for rc in recalls}
        assert rec.record_id not in live_ids


class TestFtsIntegration:
    async def test_fts_scores_used(self, store_factory):
        store = store_factory()
        store.put("ml", "machine learning with Python", scope=MemoryScope.SESSION)
        store.put("food", "pizza is delicious", scope=MemoryScope.SESSION)
        r = MemoryRetriever(store)
        recalls = await r.retrieve("machine learning Python")
        assert len(recalls) >= 1
        assert recalls[0].memory_id is not None


class TestExplanation:
    async def test_explanation_present(self, retriever):
        recalls = await retriever.retrieve("Python")
        if recalls:
            assert len(recalls[0].explanation) > 0

    async def test_relevance_detail_present(self, retriever):
        recalls = await retriever.retrieve("Python")
        if recalls:
            detail = recalls[0].relevance_detail
            assert detail.final_score > 0


class TestMatchMethod:
    async def test_hybrid_classification(self, retriever):
        recalls = await retriever.retrieve("Python great ML")
        for r in recalls:
            assert r.matched_by in (
                RecallMatchMethod.LEXICAL,
                RecallMatchMethod.EMBEDDING,
                RecallMatchMethod.RULE,
                RecallMatchMethod.HYBRID,
            )


class TestSessionContext:
    async def test_tag_boost(self, store_factory):
        store = store_factory()
        store.put("k", "tagged content", scope=MemoryScope.SESSION, tags=["python"])
        r = MemoryRetriever(store, embedding_provider=NoOpEmbeddingProvider(dim=32))
        ctx = SessionContext(active_tags=["python"])
        recalls = await r.retrieve("tagged", session_context=ctx)
        if recalls:
            assert recalls[0].relevance_detail.rule_bonus > 0

    async def test_type_boost_constraint(self, store_factory):
        store = store_factory()
        store.put(
            "c", "constraint data", scope=MemoryScope.SESSION, memory_type=MemoryType.CONSTRAINT
        )
        r = MemoryRetriever(store)
        recalls = await r.retrieve("constraint data")
        if recalls:
            assert recalls[0].relevance_detail.rule_bonus > 0


class TestProviderFailureDegradation:
    async def test_provider_degrades(self, store_factory):
        class _BrokenProvider(EmbeddingProvider):
            def dimension(self) -> int:
                return 4

            async def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("simulated provider outage")

        store = store_factory()
        store.put("py", "Python is wonderful", scope=MemoryScope.USER)
        r = MemoryRetriever(store, embedding_provider=_BrokenProvider())
        recalls = await r.retrieve("Python wonderful")
        assert isinstance(recalls, list)

    async def test_lexical_without_provider(self, store_factory):
        store = store_factory()
        store.put("a", "bicycle repair manual", scope=MemoryScope.USER)
        r = MemoryRetriever(store)
        recalls = await r.retrieve("bicycle repair")
        assert len(recalls) >= 1


class TestAnnTwoStage:
    async def test_ann_returns(self, tmp_path, store_factory):
        import time as _t

        from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
        from houyi.adapters.memory.types import MemoryRecord, MemoryType
        from houyi.adapters.memory.types import MemoryScope as MS

        backend = SQLiteMemoryBackend(db_path=tmp_path / "ann.db")
        dim = 4
        for i, (key, content) in enumerate(
            [
                ("doc_a", "machine learning model training"),
                ("doc_b", "pizza recipe ingredients"),
                ("doc_c", "deep neural network architecture"),
            ]
        ):
            rec = MemoryRecord(
                record_id=f"r{i}",
                scope=MS.USER,
                key=key,
                content=content,
                memory_type=MemoryType.FACT,
                embedding=[float(i), 0.0, 0.0, 1.0],
                created_at=_t.time(),
                updated_at=_t.time(),
            )
            backend.put(rec)
        store = store_factory(backend=backend)

        class _FixedProvider(EmbeddingProvider):
            def dimension(self) -> int:
                return dim

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0, 0.0, 0.0, 1.0]] * len(texts)

        r = MemoryRetriever(store, embedding_provider=_FixedProvider())
        recalls = await r.retrieve("neural network")
        assert isinstance(recalls, list)

    async def test_ann_uses_prefilter(self, tmp_path, store_factory):
        import time as _t

        from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
        from houyi.adapters.memory.types import MemoryRecord, MemoryType
        from houyi.adapters.memory.types import MemoryScope as MS

        backend = SQLiteMemoryBackend(db_path=tmp_path / "ann_prefilter.db")
        dim = 4
        for i in range(4):
            rec = MemoryRecord(
                record_id=f"p{i}",
                scope=MS.USER,
                key=f"k{i}",
                content=f"content {i}",
                memory_type=MemoryType.FACT,
                embedding=[float(i), 0.0, 0.0, 1.0],
                created_at=_t.time(),
                updated_at=_t.time(),
            )
            backend.put(rec)

        captured: dict[str, object] = {}
        original_search_vector = backend.search_vector

        def _wrapped_search_vector(
            query_embedding,
            *,
            scope=None,
            rowid_filter=None,
            limit=20,
        ):
            captured["rowid_filter"] = rowid_filter
            return original_search_vector(
                query_embedding,
                scope=scope,
                rowid_filter=rowid_filter,
                limit=limit,
            )

        def _prefilter_rowids(*, scopes=None, updated_after=None, updated_before=None, limit=0):
            captured["prefilter_limit"] = limit
            return [1, 2, 3]

        backend.search_vector = _wrapped_search_vector  # type: ignore[assignment]
        backend.prefilter_rowids = _prefilter_rowids  # type: ignore[assignment]

        store = store_factory(backend=backend)

        class _FixedProvider(EmbeddingProvider):
            def dimension(self) -> int:
                return dim

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0, 0.0, 0.0, 1.0]] * len(texts)

        r = MemoryRetriever(store, embedding_provider=_FixedProvider())
        await r.retrieve("content")
        assert captured.get("prefilter_limit") == 3000
        assert captured.get("rowid_filter") == [1, 2, 3]
