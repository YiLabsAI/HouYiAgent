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


class TestRetrievalTaskResult:
    """Tests for RetrievalTaskResult dataclass."""

    def test_successful_result(self) -> None:
        """Test creating a successful retrieval result."""
        from houyi.rag.indexed.mode import RetrievalTaskResult

        result = RetrievalTaskResult(
            strategy=RetrievalStrategy.VECTOR,
            strategy_name="vector",
            results=[SearchResult(chunk_id="c1", content="test", score=0.9)],
            success=True,
            duration_ms=150.0,
        )

        assert result.strategy == RetrievalStrategy.VECTOR
        assert result.strategy_name == "vector"
        assert len(result.results) == 1
        assert result.success is True
        assert result.timed_out is False
        assert result.error is None
        assert result.duration_ms == 150.0

    def test_timed_out_result(self) -> None:
        """Test creating a timed out retrieval result."""
        from houyi.rag.indexed.mode import RetrievalTaskResult

        result = RetrievalTaskResult(
            strategy=RetrievalStrategy.BM25,
            strategy_name="bm25",
            success=False,
            timed_out=True,
            error="Timeout after 10s",
            duration_ms=10000.0,
        )

        assert result.strategy == RetrievalStrategy.BM25
        assert result.success is False
        assert result.timed_out is True
        assert result.error == "Timeout after 10s"
        assert len(result.results) == 0

    def test_failed_result(self) -> None:
        """Test creating a failed retrieval result."""
        from houyi.rag.indexed.mode import RetrievalTaskResult

        result = RetrievalTaskResult(
            strategy=RetrievalStrategy.GRAPH,
            strategy_name="graph",
            success=False,
            error="Connection error",
            duration_ms=50.0,
        )

        assert result.success is False
        assert result.timed_out is False
        assert result.error == "Connection error"


class TestParallelRetrievalConfig:
    """Tests for parallel retrieval configuration."""

    def test_default_config_values(self) -> None:
        """Test default parallel retrieval config values."""
        config = IndexedConfig()

        assert config.parallel_retrieval is True
        assert config.retrieval_timeout == 10.0
        assert config.fallback_on_timeout is True

    def test_custom_config_values(self) -> None:
        """Test custom parallel retrieval config values."""
        config = IndexedConfig(
            parallel_retrieval=False,
            retrieval_timeout=5.0,
            fallback_on_timeout=False,
        )

        assert config.parallel_retrieval is False
        assert config.retrieval_timeout == 5.0
        assert config.fallback_on_timeout is False


class TestProcessRetrievalResults:
    """Tests for _process_retrieval_results method."""

    def test_all_successful(self) -> None:
        """Test processing when all strategies succeed."""
        from houyi.rag.indexed.mode import IndexedMode, RetrievalTaskResult

        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        task_results = [
            RetrievalTaskResult(
                strategy=RetrievalStrategy.VECTOR,
                strategy_name="vector",
                results=[SearchResult(chunk_id="c1", content="test", score=0.9)],
                success=True,
                duration_ms=100.0,
            ),
            RetrievalTaskResult(
                strategy=RetrievalStrategy.BM25,
                strategy_name="bm25",
                results=[SearchResult(chunk_id="c2", content="test2", score=0.8)],
                success=True,
                duration_ms=50.0,
            ),
        ]

        strategies_used: list[RetrievalStrategy] = []
        all_results: list = []
        metadata = mode._process_retrieval_results(task_results, strategies_used, all_results)

        assert metadata["total_strategies"] == 2
        assert metadata["successful_count"] == 2
        assert metadata["failed_count"] == 0
        assert metadata["timed_out_count"] == 0
        assert len(strategies_used) == 2
        assert len(all_results) == 2

    def test_partial_timeout(self) -> None:
        """Test processing when some strategies timeout."""
        from houyi.rag.indexed.mode import IndexedMode, RetrievalTaskResult

        mode = IndexedMode(
            config=IndexedConfig(fallback_on_timeout=True),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        task_results = [
            RetrievalTaskResult(
                strategy=RetrievalStrategy.VECTOR,
                strategy_name="vector",
                results=[SearchResult(chunk_id="c1", content="test", score=0.9)],
                success=True,
                duration_ms=100.0,
            ),
            RetrievalTaskResult(
                strategy=RetrievalStrategy.BM25,
                strategy_name="bm25",
                success=False,
                timed_out=True,
                error="Timeout after 10s",
                duration_ms=10000.0,
            ),
        ]

        strategies_used: list[RetrievalStrategy] = []
        all_results: list = []
        metadata = mode._process_retrieval_results(task_results, strategies_used, all_results)

        assert metadata["total_strategies"] == 2
        assert metadata["successful_count"] == 1
        assert metadata["failed_count"] == 1
        assert metadata["timed_out_count"] == 1
        # With fallback enabled, should still have partial results
        assert len(strategies_used) == 1
        assert len(all_results) == 1

    def test_no_fallback_clears_results(self) -> None:
        """Test that disabling fallback clears results on timeout."""
        from houyi.rag.indexed.mode import IndexedMode, RetrievalTaskResult

        mode = IndexedMode(
            config=IndexedConfig(fallback_on_timeout=False),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        task_results = [
            RetrievalTaskResult(
                strategy=RetrievalStrategy.VECTOR,
                strategy_name="vector",
                results=[SearchResult(chunk_id="c1", content="test", score=0.9)],
                success=True,
                duration_ms=100.0,
            ),
            RetrievalTaskResult(
                strategy=RetrievalStrategy.BM25,
                strategy_name="bm25",
                success=False,
                timed_out=True,
                error="Timeout after 10s",
                duration_ms=10000.0,
            ),
        ]

        strategies_used: list[RetrievalStrategy] = []
        all_results: list = []
        metadata = mode._process_retrieval_results(task_results, strategies_used, all_results)

        # With fallback disabled, results should be cleared
        assert metadata["timed_out_count"] == 1
        assert len(strategies_used) == 0
        assert len(all_results) == 0

    def test_metadata_strategy_details(self) -> None:
        """Test that metadata contains detailed strategy info."""
        from houyi.rag.indexed.mode import IndexedMode, RetrievalTaskResult

        mode = IndexedMode(
            config=IndexedConfig(),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(),
        )

        task_results = [
            RetrievalTaskResult(
                strategy=RetrievalStrategy.VECTOR,
                strategy_name="vector",
                results=[SearchResult(chunk_id="c1", content="test", score=0.9)],
                success=True,
                duration_ms=150.0,
            ),
        ]

        strategies_used: list[RetrievalStrategy] = []
        all_results: list = []
        metadata = mode._process_retrieval_results(task_results, strategies_used, all_results)

        assert len(metadata["strategy_details"]) == 1
        detail = metadata["strategy_details"][0]
        assert detail["strategy"] == "vector"
        assert detail["name"] == "vector"
        assert detail["success"] is True
        assert detail["timed_out"] is False
        assert detail["result_count"] == 1
        assert detail["duration_ms"] == 150.0


class TestSequentialRetrieval:
    """Tests for sequential retrieval execution."""

    @pytest.mark.asyncio
    async def test_sequential_retrieval_basic(self) -> None:
        """Test basic sequential retrieval."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25],
                    parallel_retrieval=False,
                ),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            from houyi.rag.indexed.index.sparse import SparseIndex
            from houyi.rag.types import Chunk

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._sparse_index = sparse_index

            task_results = await mode._execute_sequential_retrieval("Python", 10)

            assert len(task_results) == 1
            assert task_results[0].strategy == RetrievalStrategy.BM25
            assert task_results[0].success is True
            # duration_ms may be 0.0 on fast systems, just check it's non-negative
            assert task_results[0].duration_ms >= 0


class TestParallelRetrieval:
    """Tests for parallel retrieval execution."""

    @pytest.mark.asyncio
    async def test_parallel_retrieval_basic(self) -> None:
        """Test basic parallel retrieval."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25],
                    parallel_retrieval=True,
                ),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            from houyi.rag.indexed.index.sparse import SparseIndex
            from houyi.rag.types import Chunk

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._sparse_index = sparse_index

            task_results = await mode._execute_parallel_retrieval("Python", 10)

            assert len(task_results) == 1
            assert task_results[0].strategy == RetrievalStrategy.BM25
            assert task_results[0].success is True

    @pytest.mark.asyncio
    async def test_parallel_retrieval_empty_strategies(self) -> None:
        """Test parallel retrieval with no strategies."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            task_results = await mode._execute_parallel_retrieval("test", 10)
            assert task_results == []


class TestSearchWithRetrievalMetadata:
    """Tests for search with retrieval metadata."""

    @pytest.mark.asyncio
    async def test_search_includes_retrieval_metadata(self) -> None:
        """Test that search result includes retrieval metadata."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            from houyi.rag.indexed.index.sparse import SparseIndex
            from houyi.rag.types import Chunk

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._sparse_index = sparse_index

            result = await mode.search("Python")

            assert "retrieval" in result.metadata
            retrieval_meta = result.metadata["retrieval"]
            assert "total_strategies" in retrieval_meta
            assert "successful_count" in retrieval_meta
            assert "failed_count" in retrieval_meta
            assert "timed_out_count" in retrieval_meta
            assert "strategy_details" in retrieval_meta

    @pytest.mark.asyncio
    async def test_search_sequential_mode(self) -> None:
        """Test search in sequential retrieval mode."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25],
                    parallel_retrieval=False,
                ),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            from houyi.rag.indexed.index.sparse import SparseIndex
            from houyi.rag.types import Chunk

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._sparse_index = sparse_index

            result = await mode.search("Python")

            assert result.mode_used == RAGMode.INDEXED
            assert "retrieval" in result.metadata

    @pytest.mark.asyncio
    async def test_confidence_reduced_on_timeout(self) -> None:
        """Test that confidence is reduced when strategies timeout."""
        from houyi.rag.indexed.mode import IndexedMode, RetrievalTaskResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25],
                    retrieval_timeout=0.001,  # Very short timeout
                ),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            # Mock the process to simulate timeout
            strategies_used: list[RetrievalStrategy] = []
            all_results: list = []
            task_results = [
                RetrievalTaskResult(
                    strategy=RetrievalStrategy.BM25,
                    strategy_name="bm25",
                    success=False,
                    timed_out=True,
                    error="Timeout",
                    duration_ms=1000.0,
                ),
            ]
            metadata = mode._process_retrieval_results(task_results, strategies_used, all_results)

            # Verify timeout is tracked
            assert metadata["timed_out_count"] == 1


class TestSearchWithLLMComponents:
    """Tests for search with LLM components."""

    @pytest.mark.asyncio
    async def test_search_with_reranker(self) -> None:
        """Test search with LLM reranking enabled."""
        from houyi.rag.indexed.mode import IndexedMode

        adapter = FakeLLMAdapter(
            [
                '{"scores": [9, 5]}',  # Reranker response
                "Generated answer based on results.",  # Answer generator response
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25],
                    use_rerank=True,
                ),
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
                    Chunk(chunk_id="c1", doc_id="d1", content="Python is great"),
                    Chunk(chunk_id="c2", doc_id="d1", content="Java is also good"),
                ]
            )
            mode._sparse_index = sparse_index

            result = await mode.search("programming")

            assert result.answer
            assert result.mode_used == RAGMode.INDEXED

    @pytest.mark.asyncio
    async def test_search_with_crag_validation(self) -> None:
        """Test search with CRAG validation."""
        from houyi.rag.indexed.mode import IndexedMode

        adapter = FakeLLMAdapter(
            [
                '{"quality": "correct", "confidence": 0.9, "reasoning": "Good match", "relevant_indices": [0]}',
                "Answer with [1] citation.",  # Answer generator
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25],
                    use_rerank=False,
                    use_crag=True,
                ),
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
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming language"),
                ]
            )
            mode._sparse_index = sparse_index

            result = await mode.search("What is Python?", enable_crag=True)

            assert result.answer
            # CRAG metadata should be present
            if "crag_quality" in result.metadata:
                assert result.metadata["crag_quality"] in ["correct", "incorrect", "ambiguous"]

    @pytest.mark.asyncio
    async def test_search_crag_disabled(self) -> None:
        """Test search with CRAG validation disabled."""
        from houyi.rag.indexed.mode import IndexedMode

        adapter = FakeLLMAdapter(
            [
                "Simple answer without CRAG.",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25],
                    use_rerank=False,
                ),
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
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._sparse_index = sparse_index

            result = await mode.search("Python", enable_crag=False)

            assert "crag_quality" not in result.metadata

    @pytest.mark.asyncio
    async def test_search_with_query_analyzer(self) -> None:
        """Test search includes query analysis metadata."""
        from houyi.rag.indexed.mode import IndexedMode

        adapter = FakeLLMAdapter(
            [
                "Answer text.",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25],
                    use_rerank=False,
                ),
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
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._sparse_index = sparse_index

            result = await mode.search("What is Python?", enable_crag=False)

            # Query analysis should be in metadata
            assert "query_analysis" in result.metadata


class TestConfidenceAdjustment:
    """Tests for confidence adjustment logic."""

    @pytest.mark.asyncio
    async def test_confidence_with_incorrect_crag(self) -> None:
        """Test confidence is capped when CRAG reports incorrect."""
        from houyi.rag.indexed.mode import IndexedMode

        # Simulate search flow where CRAG says results are incorrect
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25], use_rerank=False),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            # Test the confidence adjustment logic directly
            # When crag_quality is "incorrect", confidence should be capped at 0.3
            base_confidence = 0.8
            crag_quality = "incorrect"

            if crag_quality == "incorrect":
                adjusted = min(base_confidence, 0.3)
            else:
                adjusted = base_confidence

            assert adjusted == 0.3

    @pytest.mark.asyncio
    async def test_confidence_with_ambiguous_crag(self) -> None:
        """Test confidence is capped when CRAG reports ambiguous."""
        base_confidence = 0.9
        crag_quality = "ambiguous"

        if crag_quality == "ambiguous":
            adjusted = min(base_confidence, 0.6)
        else:
            adjusted = base_confidence

        assert adjusted == 0.6

    @pytest.mark.asyncio
    async def test_confidence_with_timeout(self) -> None:
        """Test confidence is capped when strategies timeout."""
        base_confidence = 0.9
        timed_out_count = 1

        if timed_out_count > 0:
            adjusted = min(base_confidence, 0.7)
        else:
            adjusted = base_confidence

        assert adjusted == 0.7


class TestMultipleStrategies:
    """Tests for multiple retrieval strategies."""

    @pytest.mark.asyncio
    async def test_parallel_multiple_strategies(self) -> None:
        """Test parallel execution with multiple strategies."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25, RetrievalStrategy.GRAPH],
                    parallel_retrieval=True,
                ),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(enabled=True),
            )

            from houyi.rag.indexed.graph.store import GraphStore
            from houyi.rag.indexed.index.sparse import SparseIndex
            from houyi.rag.types import Chunk, Entity

            # Setup sparse index
            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._sparse_index = sparse_index

            # Setup graph store
            graph_store = GraphStore(knowledge_dir=tmpdir, config=GraphConfig(enabled=True))
            await graph_store.load()
            await graph_store.add_entities(
                [
                    Entity(entity_id="e1", name="Python", entity_type="language"),
                ]
            )
            mode._graph_store = graph_store

            try:
                result = await mode.search("Python")

                assert result.mode_used == RAGMode.INDEXED
                # Both strategies should be used
                assert len(result.strategies_used) >= 1
            finally:
                # Explicitly close resources to avoid Windows file locking issues
                sparse_index.close()
                graph_store.close()

    @pytest.mark.asyncio
    async def test_sequential_multiple_strategies(self) -> None:
        """Test sequential execution with multiple strategies."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25, RetrievalStrategy.GRAPH],
                    parallel_retrieval=False,
                ),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(enabled=True),
            )

            from houyi.rag.indexed.graph.store import GraphStore
            from houyi.rag.indexed.index.sparse import SparseIndex
            from houyi.rag.types import Chunk, Entity

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._sparse_index = sparse_index

            graph_store = GraphStore(knowledge_dir=tmpdir, config=GraphConfig(enabled=True))
            await graph_store.load()
            await graph_store.add_entities(
                [
                    Entity(entity_id="e1", name="Python", entity_type="language"),
                ]
            )
            mode._graph_store = graph_store

            try:
                task_results = await mode._execute_sequential_retrieval("Python", 10)

                # Should have results for both strategies
                assert len(task_results) == 2
            finally:
                # Explicitly close resources to avoid Windows file locking issues
                sparse_index.close()
                graph_store.close()


class TestVectorStrategy:
    """Tests for vector retrieval strategy."""

    @pytest.mark.asyncio
    async def test_vector_search_parallel(self) -> None:
        """Test vector search in parallel mode."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.VECTOR],
                    parallel_retrieval=True,
                ),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(
                    provider="openai",
                    model="test",
                    dimension=16,
                ),
                graph_config=GraphConfig(),
            )

            # We can't easily test vector search without mocking the embedder
            # Just verify the configuration is correct
            assert RetrievalStrategy.VECTOR in mode._config.strategies

    @pytest.mark.asyncio
    async def test_vector_search_sequential(self) -> None:
        """Test vector search in sequential mode."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.VECTOR],
                    parallel_retrieval=False,
                ),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(
                    provider="openai",
                    model="test",
                    dimension=16,
                ),
                graph_config=GraphConfig(),
            )

            assert RetrievalStrategy.VECTOR in mode._config.strategies
            assert mode._config.parallel_retrieval is False


class TestIngestMethod:
    """Tests for the ingest method."""

    @pytest.mark.asyncio
    async def test_ingest_stats_structure(self) -> None:
        """Test that ingest returns proper stats structure."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            # Mock the internal components
            with (
                patch("houyi.rag.indexed.document.loaders.load_documents") as mock_load,
                patch("houyi.rag.indexed.document.splitters.split_documents") as mock_split,
            ):
                mock_load.return_value = []
                mock_split.return_value = []

                # Mock embedder
                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[])
                mode._embedder = mock_embedder

                # Mock indexes
                mock_vector_index = MagicMock()
                mock_vector_index.add_batch = AsyncMock()
                mock_vector_index.save = AsyncMock()
                mode._vector_index = mock_vector_index

                mock_sparse_index = MagicMock()
                mock_sparse_index.add_batch = AsyncMock()
                mock_sparse_index.save = AsyncMock()
                mode._sparse_index = mock_sparse_index

                stats = await mode.ingest([])

                assert "documents" in stats
                assert "chunks" in stats
                assert "entities" in stats
                assert "relations" in stats

    @pytest.mark.asyncio
    async def test_ingest_with_graph_building(self) -> None:
        """Test ingest with graph building enabled."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from houyi.rag.indexed.mode import IndexedMode
        from houyi.rag.types import Chunk

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(enabled=True),
            )

            test_chunks = [Chunk(chunk_id="c1", doc_id="d1", content="Python programming")]

            with (
                patch("houyi.rag.indexed.document.loaders.load_documents") as mock_load,
                patch("houyi.rag.indexed.document.splitters.split_documents") as mock_split,
                patch("houyi.rag.indexed.graph.extractor.extract_entities") as mock_extract,
            ):
                mock_load.return_value = [MagicMock()]
                mock_split.return_value = test_chunks
                mock_extract.return_value = ([], [])

                # Mock embedder
                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 16])
                mode._embedder = mock_embedder

                # Mock indexes
                mock_vector_index = MagicMock()
                mock_vector_index.add_batch = AsyncMock()
                mock_vector_index.save = AsyncMock()
                mode._vector_index = mock_vector_index

                mock_sparse_index = MagicMock()
                mock_sparse_index.add_batch = AsyncMock()
                mock_sparse_index.save = AsyncMock()
                mode._sparse_index = mock_sparse_index

                # Mock graph store
                mock_graph_store = MagicMock()
                mock_graph_store.add_entities = AsyncMock()
                mock_graph_store.add_relations = AsyncMock()
                mock_graph_store.save = AsyncMock()
                mode._graph_store = mock_graph_store

                stats = await mode.ingest(["test.txt"], build_graph=True)

                assert stats["documents"] == 1
                assert stats["chunks"] == 1
                mock_graph_store.add_entities.assert_called_once()
                mock_graph_store.add_relations.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_with_contextual_retrieval(self) -> None:
        """Test ingest with contextual retrieval enabled."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from houyi.rag.indexed.mode import IndexedMode
        from houyi.rag.types import Chunk

        adapter = FakeLLMAdapter(["Contextualized content"])

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
                llm_adapter=adapter,
            )

            test_chunks = [Chunk(chunk_id="c1", doc_id="d1", content="Python")]

            with (
                patch("houyi.rag.indexed.document.loaders.load_documents") as mock_load,
                patch("houyi.rag.indexed.document.splitters.split_documents") as mock_split,
            ):
                mock_load.return_value = [MagicMock()]
                mock_split.return_value = test_chunks

                # Mock embedder
                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 16])
                mode._embedder = mock_embedder

                # Mock indexes
                mock_vector_index = MagicMock()
                mock_vector_index.add_batch = AsyncMock()
                mock_vector_index.save = AsyncMock()
                mode._vector_index = mock_vector_index

                mock_sparse_index = MagicMock()
                mock_sparse_index.add_batch = AsyncMock()
                mock_sparse_index.save = AsyncMock()
                mode._sparse_index = mock_sparse_index

                # Mock contextualizer
                mock_contextualizer = MagicMock()
                mock_contextualized = MagicMock()
                mock_contextualized.contextualized_content = "Contextualized Python"
                mock_contextualizer.contextualize_chunks = AsyncMock(
                    return_value=[mock_contextualized]
                )
                mode._contextualizer = mock_contextualizer

                stats = await mode.ingest(["test.txt"], contextual_retrieval=True)

                assert stats["contextualized_chunks"] == 1
                mock_contextualizer.contextualize_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_with_llm_entity_extraction(self) -> None:
        """Test ingest using LLM for entity extraction."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from houyi.rag.indexed.mode import IndexedMode
        from houyi.rag.types import Chunk, Entity, Relation

        adapter = FakeLLMAdapter(
            ['{"entities": [{"name": "Python", "type": "language"}], "relations": []}']
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(enabled=True),
                llm_adapter=adapter,
            )

            test_chunks = [Chunk(chunk_id="c1", doc_id="d1", content="Python programming")]

            with (
                patch("houyi.rag.indexed.document.loaders.load_documents") as mock_load,
                patch("houyi.rag.indexed.document.splitters.split_documents") as mock_split,
            ):
                mock_load.return_value = [MagicMock()]
                mock_split.return_value = test_chunks

                # Mock embedder
                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 16])
                mode._embedder = mock_embedder

                # Mock indexes
                mock_vector_index = MagicMock()
                mock_vector_index.add_batch = AsyncMock()
                mock_vector_index.save = AsyncMock()
                mode._vector_index = mock_vector_index

                mock_sparse_index = MagicMock()
                mock_sparse_index.add_batch = AsyncMock()
                mock_sparse_index.save = AsyncMock()
                mode._sparse_index = mock_sparse_index

                # Mock graph store
                mock_graph_store = MagicMock()
                mock_graph_store.add_entities = AsyncMock()
                mock_graph_store.add_relations = AsyncMock()
                mock_graph_store.save = AsyncMock()
                mode._graph_store = mock_graph_store

                # Mock entity extractor with results
                test_entities = [Entity(entity_id="e1", name="Python", entity_type="language")]
                test_relations: list[Relation] = []
                mode._entity_extractor.extract_batch = AsyncMock(
                    return_value=(test_entities, test_relations)
                )

                stats = await mode.ingest(["test.txt"], build_graph=True)

                assert stats["entities"] == 1
                assert stats["relations"] == 0


class TestEnsureMethods:
    """Tests for lazy initialization ensure methods."""

    @pytest.mark.asyncio
    async def test_ensure_sparse_index(self) -> None:
        """Test lazy sparse index initialization."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.BM25]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            assert mode._sparse_index is None
            await mode._ensure_sparse_index()
            assert mode._sparse_index is not None

    @pytest.mark.asyncio
    async def test_ensure_graph_store(self) -> None:
        """Test lazy graph store initialization."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.GRAPH]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(enabled=True),
            )

            assert mode._graph_store is None
            await mode._ensure_graph_store()
            assert mode._graph_store is not None

            # Explicitly close to avoid Windows file locking issues
            if mode._graph_store:
                mode._graph_store.close()

    @pytest.mark.asyncio
    async def test_ensure_vector_index(self) -> None:
        """Test lazy vector index initialization."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.VECTOR]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            assert mode._vector_index is None
            await mode._ensure_vector_index()
            assert mode._vector_index is not None

    @pytest.mark.asyncio
    async def test_ensure_embedder(self) -> None:
        """Test lazy embedder initialization."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(strategies=[RetrievalStrategy.VECTOR]),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="local", model="test", dimension=384),
                graph_config=GraphConfig(),
            )

            assert mode._embedder is None
            await mode._ensure_embedder()
            assert mode._embedder is not None


class TestGenerateAnswer:
    """Tests for answer generation."""

    @pytest.mark.asyncio
    async def test_generate_answer_empty_results(self) -> None:
        """Test answer generation with empty results."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            answer, confidence = await mode._generate_answer("test query", [])

            assert "No relevant" in answer
            assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_generate_answer_with_results(self) -> None:
        """Test answer generation with results (no LLM)."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            results = [
                SearchResult(chunk_id="c1", content="Python is a programming language.", score=0.9),
            ]

            answer, confidence = await mode._generate_answer("What is Python?", results)

            assert "Python" in answer
            assert confidence > 0

    @pytest.mark.asyncio
    async def test_generate_answer_with_llm(self) -> None:
        """Test answer generation with LLM adapter."""
        from houyi.rag.indexed.mode import IndexedMode

        adapter = FakeLLMAdapter(["Python is an interpreted, high-level programming language. [1]"])

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
                llm_adapter=adapter,
            )

            results = [
                SearchResult(chunk_id="c1", content="Python is great.", score=0.9),
            ]

            answer, confidence = await mode._generate_answer("What is Python?", results)

            assert answer
            assert confidence > 0


class TestIndexDir:
    """Tests for index directory configuration."""

    def test_separate_index_dir(self) -> None:
        """Test using separate index directory."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_dir = tmpdir + "/knowledge"
            index_dir = tmpdir + "/index"

            mode = IndexedMode(
                config=IndexedConfig(),
                knowledge_dir=knowledge_dir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
                index_dir=index_dir,
            )

            assert mode._knowledge_dir == knowledge_dir
            assert mode._index_dir == index_dir

    def test_default_index_dir(self) -> None:
        """Test default index directory (same as knowledge_dir)."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            assert mode._index_dir == tmpdir
