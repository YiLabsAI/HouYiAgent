"""MemoryRetriever unit tests.

Covers hybrid scoring, FTS fallback, match method classification,
context filtering, and top-K truncation.
"""

from __future__ import annotations

import time

import pytest

from houyi.adapters.memory.embedding import NoOpEmbeddingProvider
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
def store() -> MemoryStore:
    s = MemoryStore()
    s.put("python", "Python is great for ML", scope=MemoryScope.USER, memory_type=MemoryType.FACT)
    s.put("rust", "Rust is fast and safe", scope=MemoryScope.USER, memory_type=MemoryType.FACT)
    s.put("name", "User name: Alice", scope=MemoryScope.USER, memory_type=MemoryType.PROFILE)
    return s


@pytest.fixture()
def retriever(store) -> MemoryRetriever:
    emb = NoOpEmbeddingProvider(dim=32)
    return MemoryRetriever(store, embedding_provider=emb)


@pytest.fixture()
def retriever_no_emb(store) -> MemoryRetriever:
    return MemoryRetriever(store)


class TestBasicRetrieval:
    async def test_returns_results(self, retriever):
        recalls = await retriever.retrieve("Python machine learning")
        assert len(recalls) >= 1
        assert recalls[0].score > 0

    async def test_empty_store(self):
        store = MemoryStore()
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
    async def test_scope_priority(self):
        store = MemoryStore()
        store.put("a", "session data", scope=MemoryScope.SESSION)
        policy = MemoryPolicy(scope_priority=[MemoryScope.USER])
        r = MemoryRetriever(store, policy=policy)
        recalls = await r.retrieve("session data")
        assert len(recalls) == 0

    async def test_expired_excluded(self):
        store = MemoryStore()
        store.put("live", "Python is great", scope=MemoryScope.SESSION)
        rec = store.put("dead", "stale data", scope=MemoryScope.SESSION, ttl=0.001)
        time.sleep(0.01)
        r = MemoryRetriever(store)
        recalls = await r.retrieve("stale data")
        live_ids = {rc.memory_id for rc in recalls}
        assert rec.record_id not in live_ids


class TestFtsIntegration:
    async def test_fts_scores_used(self):
        store = MemoryStore()
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
    async def test_tag_boost(self):
        store = MemoryStore()
        store.put("k", "tagged content", scope=MemoryScope.SESSION, tags=["python"])
        r = MemoryRetriever(store, embedding_provider=NoOpEmbeddingProvider(dim=32))
        ctx = SessionContext(active_tags=["python"])
        recalls = await r.retrieve("tagged", session_context=ctx)
        if recalls:
            assert recalls[0].relevance_detail.rule_bonus > 0

    async def test_type_boost_constraint(self):
        store = MemoryStore()
        store.put(
            "c", "constraint data", scope=MemoryScope.SESSION, memory_type=MemoryType.CONSTRAINT
        )
        r = MemoryRetriever(store)
        recalls = await r.retrieve("constraint data")
        if recalls:
            assert recalls[0].relevance_detail.rule_bonus > 0
