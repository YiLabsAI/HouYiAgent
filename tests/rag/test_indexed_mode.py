"""Tests for IndexedMode."""

from __future__ import annotations

import tempfile

import pytest

from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
from houyi.rag.types import (
    RAGMode,
    RetrievalStrategy,
    SearchResult,
)


class FakeLLMAdapter:
    """Fake LLM adapter for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ['{"keywords": ["test"]}']
        self._index = 0

    async def chat(self, messages, **kwargs):
        from houyi.llm.base import LLMResponse

        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return LLMResponse(
            content=response,
            tool_calls=[],
            finish_reason="stop",
            usage={"total_tokens": 50},
            model="fake-model",
        )


class TestIndexedModeInit:
    """Test IndexedMode initialization."""

    def test_basic_init(self) -> None:
        """Test basic IndexedMode creation."""
        from houyi.rag.indexed.mode import IndexedMode

        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )
        assert mode._config.top_k == 10
        assert mode._reranker is None
        assert mode._answer_generator is None

    def test_init_with_llm(self) -> None:
        """Test IndexedMode with LLM adapter."""
        from houyi.rag.indexed.mode import IndexedMode

        adapter = FakeLLMAdapter()
        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
            llm_adapter=adapter,
        )
        assert mode._reranker is not None
        assert mode._answer_generator is not None
        assert mode._entity_extractor is not None


class TestRRFFusion:
    """Test RRF fusion logic."""

    def test_single_strategy(self) -> None:
        """Test RRF with single strategy returns as-is."""
        from houyi.rag.indexed.mode import IndexedMode

        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        results = [
            SearchResult(chunk_id="c1", content="Result 1", score=0.9),
            SearchResult(chunk_id="c2", content="Result 2", score=0.8),
        ]

        fused = mode._rrf_fusion([("vector", results)], k=10)
        assert len(fused) == 2
        assert fused[0].chunk_id == "c1"

    def test_multi_strategy_fusion(self) -> None:
        """Test RRF fusion across multiple strategies."""
        from houyi.rag.indexed.mode import IndexedMode

        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        vector_results = [
            SearchResult(chunk_id="c1", content="Python programming", score=0.9),
            SearchResult(chunk_id="c2", content="Java programming", score=0.7),
        ]
        bm25_results = [
            SearchResult(chunk_id="c1", content="Python programming", score=0.8),
            SearchResult(chunk_id="c3", content="Go programming", score=0.6),
        ]

        fused = mode._rrf_fusion([("vector", vector_results), ("bm25", bm25_results)], k=10)

        # c1 appears in both, should rank highest
        assert fused[0].chunk_id == "c1"
        assert len(fused) == 3

    def test_empty_fusion(self) -> None:
        """Test RRF with no results."""
        from houyi.rag.indexed.mode import IndexedMode

        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        fused = mode._rrf_fusion([], k=10)
        assert fused == []


class TestIndexedModeSearch:
    """Test IndexedMode search flow."""

    @pytest.mark.asyncio
    async def test_search_bm25_only(self) -> None:
        """Test search using only BM25 strategy."""
        pytest.importorskip("bm25s")
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            # Pre-populate sparse index
            from houyi.rag.indexed.index.sparse import SparseIndex
            from houyi.rag.types import Chunk

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python is great for data science"),
                    Chunk(chunk_id="c2", doc_id="d1", content="Java is used in enterprise"),
                ]
            )
            mode._sparse_index = sparse_index

            result = await mode.search("Python data science")

            assert result.mode_used == RAGMode.INDEXED
            assert RetrievalStrategy.BM25 in result.strategies_used
            assert result.answer

    @pytest.mark.asyncio
    async def test_search_with_llm_answer(self) -> None:
        """Test search with LLM answer generation."""
        pytest.importorskip("bm25s")
        from houyi.rag.indexed.mode import IndexedMode

        adapter = FakeLLMAdapter(
            [
                "Python is excellent for data science because of libraries like NumPy and Pandas.",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25], use_rerank=False),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
                llm_adapter=adapter,
            )

            from houyi.rag.indexed.index.sparse import SparseIndex
            from houyi.rag.types import Chunk

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python has NumPy and Pandas"),
                ]
            )
            mode._sparse_index = sparse_index

            result = await mode.search("Why Python for data science?")

            assert result.answer
            assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """Test search that returns no results."""
        pytest.importorskip("bm25s")
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            from houyi.rag.indexed.index.sparse import SparseIndex

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            mode._sparse_index = sparse_index

            result = await mode.search("query on empty index")

            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_search_with_graph_strategy(self) -> None:
        """Test search using graph strategy."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.GRAPH]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(enabled=True),
            )

            from houyi.rag.indexed.graph.store import GraphStore
            from houyi.rag.types import Entity, Relation

            async with GraphStore(knowledge_dir=tmpdir) as graph_store:
                await graph_store.add_entities(
                    [
                        Entity(entity_id="e1", name="Python", entity_type="language"),
                        Entity(entity_id="e2", name="Django", entity_type="framework"),
                    ]
                )
                await graph_store.add_relations(
                    [
                        Relation(
                            rel_id="r1",
                            source_id="e1",
                            target_id="e2",
                            rel_type="has_framework",
                            weight=1.0,
                        ),
                    ]
                )
                mode._graph_store = graph_store

                result = await mode.search("Python")

                assert result.mode_used == RAGMode.INDEXED
                assert len(result.search_results) > 0


class TestBuildAnswerSimple:
    """Test simple answer building."""

    def test_build_answer(self) -> None:
        """Test simple answer concatenation."""
        from houyi.rag.indexed.mode import IndexedMode

        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        results = [
            SearchResult(chunk_id="c1", content="Answer part 1", score=0.9),
            SearchResult(chunk_id="c2", content="Answer part 2", score=0.8),
        ]

        answer = mode._build_answer_simple(results)
        assert "Answer part 1" in answer
        assert "Answer part 2" in answer

    def test_build_answer_empty(self) -> None:
        """Test answer with no results."""
        from houyi.rag.indexed.mode import IndexedMode

        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        answer = mode._build_answer_simple([])
        assert answer  # Should return default message
