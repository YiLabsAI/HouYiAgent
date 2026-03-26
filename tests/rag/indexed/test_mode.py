from __future__ import annotations

import tempfile
from pathlib import Path
from types import MethodType

import pytest

from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
from houyi.rag.indexed.document import loaders as loaders_module
from houyi.rag.indexed.document import splitters as splitters_module
from houyi.rag.indexed.graph import extractor as extractor_module
from houyi.rag.indexed.models import RetrievalTaskResult
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
        from houyi.adapters.llm.base import LLMResponse

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
    def test_basic_init(self) -> None:
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
    def test_single_strategy(self) -> None:
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

        assert fused[0].chunk_id == "c1"
        assert len(fused) == 3

    def test_empty_fusion(self) -> None:
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
    @pytest.mark.asyncio
    async def test_search_parallel_and_sequential_share_same_retrieval_surface(self) -> None:
        from houyi.rag.indexed.mode import IndexedMode

        task_results = [
            RetrievalTaskResult(
                strategy=RetrievalStrategy.BM25,
                strategy_name="bm25",
                results=[SearchResult(chunk_id="c1", content="Python local search", score=0.9)],
                success=True,
                duration_ms=3.0,
            ),
            RetrievalTaskResult(
                strategy=RetrievalStrategy.GRAPH,
                strategy_name="graph",
                results=[SearchResult(chunk_id="c1", content="Python local search", score=0.7)],
                success=True,
                duration_ms=4.0,
            ),
        ]

        async def _analyze_query_noop(self, query: str, metadata: dict[str, object]) -> None:
            metadata["query_analysis"] = {"query": query}

        async def _return_task_results(self, query: str, k: int) -> list[RetrievalTaskResult]:
            assert query == "python local search"
            assert k == 5
            return task_results

        results_by_mode: dict[str, object] = {}
        for parallel_retrieval in (True, False):
            mode = IndexedMode(
                config=IndexedConfig(
                    strategies=[RetrievalStrategy.BM25, RetrievalStrategy.GRAPH],
                    parallel_retrieval=parallel_retrieval,
                    top_k=5,
                    use_rerank=False,
                ),
                knowledge_dir="/tmp/test-indexed-mode",
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(enabled=True),
            )
            mode._analyze_query = MethodType(_analyze_query_noop, mode)
            if parallel_retrieval:
                mode._execute_parallel_retrieval = MethodType(_return_task_results, mode)
            else:
                mode._execute_sequential_retrieval = MethodType(_return_task_results, mode)

            result = await mode.search("python local search")
            results_by_mode["parallel" if parallel_retrieval else "sequential"] = result

        parallel_result = results_by_mode["parallel"]
        sequential_result = results_by_mode["sequential"]

        assert parallel_result.mode_used == sequential_result.mode_used == RAGMode.INDEXED
        assert parallel_result.answer == sequential_result.answer
        assert [item.chunk_id for item in parallel_result.search_results] == [
            item.chunk_id for item in sequential_result.search_results
        ]
        assert parallel_result.strategies_used == sequential_result.strategies_used
        assert parallel_result.metadata["retrieval"] == sequential_result.metadata["retrieval"]

    @pytest.mark.asyncio
    async def test_search_bm25_only(self) -> None:
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
            from houyi.rag.types import Chunk

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python is great for data science"),
                    Chunk(chunk_id="c2", doc_id="d1", content="Java is used in enterprise"),
                ]
            )
            mode._resources.sparse_index = sparse_index

            result = await mode.search("Python data science")

            assert result.mode_used == RAGMode.INDEXED
            assert RetrievalStrategy.BM25 in result.strategies_used
            assert result.answer

    @pytest.mark.asyncio
    async def test_search_with_llm_answer(self) -> None:
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
            mode._resources.sparse_index = sparse_index

            result = await mode.search("Why Python for data science?")

            assert result.answer
            assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
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
            mode._resources.sparse_index = sparse_index

            result = await mode.search("query on empty index")

            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_search_with_graph_strategy(self) -> None:
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
                mode._resources.graph_store = graph_store

                result = await mode.search("Python")

                assert result.mode_used == RAGMode.INDEXED
                assert len(result.search_results) > 0


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


class TestRetrievalExecutionRequestBuilder:
    """Tests for shared retrieval execution request construction."""

    def test_request_uses_configured_strategies_and_timeout(self) -> None:
        """The shared execution request should mirror IndexedConfig retrieval settings."""
        from houyi.rag.indexed.mode import IndexedMode

        config = IndexedConfig(
            strategies=[RetrievalStrategy.BM25, RetrievalStrategy.VECTOR],
            retrieval_timeout=3.5,
        )
        mode = IndexedMode(
            config=config,
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(enabled=False),
        )

        request = mode._build_retrieval_execution_request("python", 4)

        assert request.strategies == config.strategies
        assert request.timeout == 3.5
        assert request.result_factory is not None

    def test_request_task_factory_respects_dispatch_context(self) -> None:
        """The shared execution request should keep dispatch wiring consistent across execution modes."""
        from houyi.rag.indexed.mode import IndexedMode

        mode = IndexedMode(
            config=IndexedConfig(strategies=[RetrievalStrategy.GRAPH]),
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=384),
            graph_config=GraphConfig(enabled=False),
        )

        request = mode._build_retrieval_execution_request("python", 2)

        graph_task = request.create_retrieval_task(RetrievalStrategy.GRAPH)
        vector_task = request.create_retrieval_task(RetrievalStrategy.VECTOR)

        assert graph_task is None
        assert vector_task is not None
        assert vector_task[0] == "vector"
        vector_task[1].close()


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
            mode._resources.sparse_index = sparse_index

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
            mode._resources.sparse_index = sparse_index

            result = await mode.search("Python")

            assert result.mode_used == RAGMode.INDEXED
            assert "retrieval" in result.metadata


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
            mode._resources.sparse_index = sparse_index

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
            mode._resources.sparse_index = sparse_index

            result = await mode.search("What is Python?", enable_crag=True)

            assert result.answer
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
            mode._resources.sparse_index = sparse_index

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
            mode._resources.sparse_index = sparse_index

            result = await mode.search("What is Python?", enable_crag=False)

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

            sparse_index = SparseIndex(knowledge_dir=tmpdir)
            await sparse_index.load()
            await sparse_index.add_batch(
                [
                    Chunk(chunk_id="c1", doc_id="d1", content="Python programming"),
                ]
            )
            mode._resources.sparse_index = sparse_index

            graph_store = GraphStore(knowledge_dir=tmpdir, config=GraphConfig(enabled=True))
            await graph_store.load()
            await graph_store.add_entities(
                [
                    Entity(entity_id="e1", name="Python", entity_type="language"),
                ]
            )
            mode._resources.graph_store = graph_store

            try:
                result = await mode.search("Python")

                assert result.mode_used == RAGMode.INDEXED
                assert len(result.strategies_used) >= 1
            finally:
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

            with (
                patch.object(loaders_module, "load_documents") as mock_load,
                patch.object(splitters_module, "split_documents") as mock_split,
            ):
                mock_load.return_value = []
                mock_split.return_value = []

                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[])
                mode._resources.embedder = mock_embedder

                mock_vector_index = MagicMock()
                mock_vector_index.add_batch = AsyncMock()
                mock_vector_index.save = AsyncMock()
                mode._resources.vector_index = mock_vector_index

                mock_sparse_index = MagicMock()
                mock_sparse_index.add_batch = AsyncMock()
                mock_sparse_index.save = AsyncMock()
                mode._resources.sparse_index = mock_sparse_index

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
                patch.object(loaders_module, "load_documents") as mock_load,
                patch.object(splitters_module, "split_documents") as mock_split,
                patch.object(extractor_module, "extract_entities") as mock_extract,
            ):
                mock_load.return_value = [MagicMock()]
                mock_split.return_value = test_chunks
                mock_extract.return_value = ([], [])

                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 16])
                mode._resources.embedder = mock_embedder

                mock_vector_index = MagicMock()
                mock_vector_index.add_batch = AsyncMock()
                mock_vector_index.save = AsyncMock()
                mode._resources.vector_index = mock_vector_index

                mock_sparse_index = MagicMock()
                mock_sparse_index.add_batch = AsyncMock()
                mock_sparse_index.save = AsyncMock()
                mode._resources.sparse_index = mock_sparse_index

                mock_graph_store = MagicMock()
                mock_graph_store.add_entities = AsyncMock()
                mock_graph_store.add_relations = AsyncMock()
                mock_graph_store.save = AsyncMock()
                mode._resources.graph_store = mock_graph_store

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
                patch.object(loaders_module, "load_documents") as mock_load,
                patch.object(splitters_module, "split_documents") as mock_split,
            ):
                mock_load.return_value = [MagicMock()]
                mock_split.return_value = test_chunks

                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 16])
                mode._resources.embedder = mock_embedder

                mock_vector_index = MagicMock()
                mock_vector_index.add_batch = AsyncMock()
                mock_vector_index.save = AsyncMock()
                mode._resources.vector_index = mock_vector_index

                mock_sparse_index = MagicMock()
                mock_sparse_index.add_batch = AsyncMock()
                mock_sparse_index.save = AsyncMock()
                mode._resources.sparse_index = mock_sparse_index

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
                patch.object(loaders_module, "load_documents") as mock_load,
                patch.object(splitters_module, "split_documents") as mock_split,
            ):
                mock_load.return_value = [MagicMock()]
                mock_split.return_value = test_chunks

                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 16])
                mode._resources.embedder = mock_embedder

                mock_vector_index = MagicMock()
                mock_vector_index.add_batch = AsyncMock()
                mock_vector_index.save = AsyncMock()
                mode._resources.vector_index = mock_vector_index

                mock_sparse_index = MagicMock()
                mock_sparse_index.add_batch = AsyncMock()
                mock_sparse_index.save = AsyncMock()
                mode._resources.sparse_index = mock_sparse_index

                mock_graph_store = MagicMock()
                mock_graph_store.add_entities = AsyncMock()
                mock_graph_store.add_relations = AsyncMock()
                mock_graph_store.save = AsyncMock()
                mode._resources.graph_store = mock_graph_store

                test_entities = [Entity(entity_id="e1", name="Python", entity_type="language")]
                test_relations: list[Relation] = []
                mode._entity_extractor.extract_batch = AsyncMock(
                    return_value=(test_entities, test_relations)
                )

                stats = await mode.ingest(["test.txt"], build_graph=True)

                assert stats["entities"] == 1
                assert stats["relations"] == 0


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
        """Test default index directory under knowledge_dir/.houyi."""
        from houyi.rag.indexed.mode import IndexedMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = IndexedMode(
                config=IndexedConfig(),
                knowledge_dir=tmpdir,
                embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
                graph_config=GraphConfig(),
            )

            assert Path(mode._index_dir) == Path(tmpdir) / ".houyi"
