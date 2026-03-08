"""Tests for RAG retrieval components."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.rag.indexed import embedding as embedding_module
from houyi.rag.retrieval import (
    HybridRetriever,
    HybridRetrieverConfig,
)
from houyi.rag.types import SearchResult


class TestHybridRetrieverConfig:
    """Tests for HybridRetrieverConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = HybridRetrieverConfig()

        assert config.fusion_method == "rrf"
        assert config.rrf_k == 60
        assert config.vector_weight == 0.4
        assert config.sparse_weight == 0.4
        assert config.graph_weight == 0.2

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = HybridRetrieverConfig(
            fusion_method="weighted",
            rrf_k=100,
            vector_weight=0.5,
            sparse_weight=0.3,
            graph_weight=0.2,
        )

        assert config.fusion_method == "weighted"
        assert config.rrf_k == 100
        assert config.vector_weight == 0.5


class TestHybridRetrieverInit:
    """Tests for HybridRetriever initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        retriever = HybridRetriever()

        assert retriever._vector_index is None
        assert retriever._sparse_index is None
        assert retriever._graph_store is None
        assert retriever._embedder is None
        assert retriever._config.fusion_method == "rrf"

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = HybridRetrieverConfig(fusion_method="weighted")
        retriever = HybridRetriever(config=config)

        assert retriever._config.fusion_method == "weighted"

    def test_init_with_components(self) -> None:
        """Test initialization with mock components."""
        mock_vector = MagicMock()
        mock_sparse = MagicMock()
        mock_graph = MagicMock()
        mock_embedder = MagicMock()

        retriever = HybridRetriever(
            vector_index=mock_vector,
            sparse_index=mock_sparse,
            graph_store=mock_graph,
            embedder=mock_embedder,
        )

        assert retriever._vector_index is mock_vector
        assert retriever._sparse_index is mock_sparse
        assert retriever._graph_store is mock_graph
        assert retriever._embedder is mock_embedder


class TestHybridRetrieverRRFFusion:
    """Tests for RRF fusion method."""

    def test_rrf_fusion_empty(self) -> None:
        """Test RRF fusion with empty results."""
        retriever = HybridRetriever()
        fused = retriever._rrf_fusion([], top_k=10)
        assert fused == []

    def test_rrf_fusion_single_strategy(self) -> None:
        """Test RRF fusion with single strategy returns as-is."""
        retriever = HybridRetriever()
        results = [
            SearchResult(chunk_id="c1", content="Result 1", score=0.9),
            SearchResult(chunk_id="c2", content="Result 2", score=0.8),
        ]

        fused = retriever._rrf_fusion([("vector", results)], top_k=10)

        assert len(fused) == 2
        assert fused[0].chunk_id == "c1"

    def test_rrf_fusion_multiple_strategies(self) -> None:
        """Test RRF fusion across multiple strategies."""
        retriever = HybridRetriever()

        vector_results = [
            SearchResult(chunk_id="c1", content="Result 1", score=0.9),
            SearchResult(chunk_id="c2", content="Result 2", score=0.7),
        ]
        sparse_results = [
            SearchResult(chunk_id="c1", content="Result 1", score=0.8),
            SearchResult(chunk_id="c3", content="Result 3", score=0.6),
        ]

        fused = retriever._rrf_fusion(
            [("vector", vector_results), ("sparse", sparse_results)],
            top_k=10,
        )

        # c1 appears in both, should rank highest
        assert fused[0].chunk_id == "c1"
        assert len(fused) == 3

    def test_rrf_fusion_respects_top_k(self) -> None:
        """Test RRF fusion respects top_k limit."""
        retriever = HybridRetriever()

        results = [
            SearchResult(chunk_id=f"c{i}", content=f"Result {i}", score=0.9 - i * 0.1)
            for i in range(10)
        ]

        fused = retriever._rrf_fusion([("vector", results)], top_k=3)

        assert len(fused) == 3

    def test_rrf_fusion_uses_content_hash_for_empty_chunk_id(self) -> None:
        """Test RRF fusion uses content hash when chunk_id is empty."""
        retriever = HybridRetriever()

        # Create results with empty chunk_id - deduplication uses content hash
        vector_results = [
            SearchResult(chunk_id="", content="Same content", score=0.9),
        ]
        sparse_results = [
            SearchResult(chunk_id="", content="Same content", score=0.8),
        ]

        fused = retriever._rrf_fusion(
            [("vector", vector_results), ("sparse", sparse_results)],
            top_k=10,
        )

        # Should deduplicate based on content hash (empty chunk_id falls back to hash)
        assert len(fused) == 1


class TestHybridRetrieverWeightedFusion:
    """Tests for weighted fusion method."""

    def test_weighted_fusion_empty(self) -> None:
        """Test weighted fusion with empty results."""
        retriever = HybridRetriever()
        fused = retriever._weighted_fusion([], top_k=10)
        assert fused == []

    def test_weighted_fusion_applies_weights(self) -> None:
        """Test weighted fusion applies correct weights."""
        config = HybridRetrieverConfig(
            fusion_method="weighted",
            vector_weight=0.6,
            sparse_weight=0.4,
        )
        retriever = HybridRetriever(config=config)

        vector_results = [
            SearchResult(chunk_id="c1", content="Vector result", score=1.0),
        ]
        sparse_results = [
            SearchResult(chunk_id="c2", content="Sparse result", score=1.0),
        ]

        fused = retriever._weighted_fusion(
            [("vector", vector_results), ("sparse", sparse_results)],
            top_k=10,
        )

        # c1 with vector weight 0.6 should rank higher than c2 with sparse weight 0.4
        assert len(fused) == 2
        assert fused[0].chunk_id == "c1"
        assert fused[0].score == 0.6

    def test_weighted_fusion_combines_same_result(self) -> None:
        """Test weighted fusion combines scores for same result."""
        config = HybridRetrieverConfig(
            fusion_method="weighted",
            vector_weight=0.5,
            sparse_weight=0.5,
        )
        retriever = HybridRetriever(config=config)

        vector_results = [
            SearchResult(chunk_id="c1", content="Same result", score=1.0),
        ]
        sparse_results = [
            SearchResult(chunk_id="c1", content="Same result", score=1.0),
        ]

        fused = retriever._weighted_fusion(
            [("vector", vector_results), ("sparse", sparse_results)],
            top_k=10,
        )

        # c1 should have combined score of 0.5 + 0.5 = 1.0
        assert len(fused) == 1
        assert fused[0].score == 1.0

    def test_weighted_fusion_unknown_strategy_uses_default_weight(self) -> None:
        """Test weighted fusion uses default weight for unknown strategy."""
        retriever = HybridRetriever()

        results = [
            SearchResult(chunk_id="c1", content="Result", score=1.0),
        ]

        fused = retriever._weighted_fusion(
            [("unknown_strategy", results)],
            top_k=10,
        )

        assert len(fused) == 1
        assert fused[0].score == pytest.approx(0.33, abs=0.01)


class TestHybridRetrieverRetrieve:
    """Tests for retrieve method."""

    @pytest.mark.asyncio
    async def test_retrieve_with_protocol_shaped_collaborators(self) -> None:
        """Test retrieve works with protocol-shaped collaborators instead of indexed concretes."""

        class FakeVectorBackend:
            async def search(self, query_embedding, top_k: int = 10):
                assert query_embedding == [0.1, 0.2]
                assert top_k == 2
                return [
                    SearchResult(chunk_id="vector-1", content="Vector result", score=0.9),
                ]

        class FakeSparseBackend:
            async def search(self, query: str, top_k: int = 10):
                assert query == "protocol query"
                assert top_k == 2
                return [
                    SearchResult(chunk_id="sparse-1", content="Sparse result", score=0.8),
                ]

        class FakeEmbedder:
            async def embed(self, text: str):
                assert text == "protocol query"
                return [0.1, 0.2]

        retriever = HybridRetriever(
            vector_index=FakeVectorBackend(),
            sparse_index=FakeSparseBackend(),
            embedder=FakeEmbedder(),
        )

        results = await retriever.retrieve("protocol query", top_k=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_no_indexes(self) -> None:
        """Test retrieve with no indexes returns empty."""
        retriever = HybridRetriever()
        results = await retriever.retrieve("test query", top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_sparse_only(self) -> None:
        """Test retrieve with sparse index only."""
        mock_sparse = MagicMock()
        mock_sparse.search = AsyncMock(
            return_value=[
                SearchResult(chunk_id="c1", content="BM25 result", score=0.9),
            ]
        )

        retriever = HybridRetriever(sparse_index=mock_sparse)
        results = await retriever.retrieve("test query", top_k=10)

        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        mock_sparse.search.assert_called_once_with("test query", 10)

    @pytest.mark.asyncio
    async def test_retrieve_vector_and_sparse(self) -> None:
        """Test retrieve with vector and sparse indexes."""
        mock_vector = MagicMock()
        mock_vector.search = AsyncMock(
            return_value=[
                SearchResult(chunk_id="c1", content="Vector result", score=0.9),
            ]
        )

        mock_sparse = MagicMock()
        mock_sparse.search = AsyncMock(
            return_value=[
                SearchResult(chunk_id="c2", content="Sparse result", score=0.8),
            ]
        )

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 16)

        retriever = HybridRetriever(
            vector_index=mock_vector,
            sparse_index=mock_sparse,
            embedder=mock_embedder,
        )

        results = await retriever.retrieve("test query", top_k=10)

        assert len(results) == 2
        mock_vector.search.assert_called_once()
        mock_sparse.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_graph(self) -> None:
        """Test retrieve with graph store."""
        mock_graph = MagicMock()
        mock_graph.search = AsyncMock(
            return_value=[
                SearchResult(chunk_id="c1", content="Graph result", score=0.9),
            ]
        )

        retriever = HybridRetriever(graph_store=mock_graph)
        results = await retriever.retrieve("test query", top_k=10)

        assert len(results) == 1
        mock_graph.search.assert_called_once_with("test query", 10)

    @pytest.mark.asyncio
    async def test_retrieve_handles_vector_failure(self) -> None:
        """Test retrieve continues when vector search fails."""
        mock_vector = MagicMock()
        mock_vector.search = AsyncMock(side_effect=Exception("Vector error"))

        mock_sparse = MagicMock()
        mock_sparse.search = AsyncMock(
            return_value=[
                SearchResult(chunk_id="c1", content="Sparse result", score=0.9),
            ]
        )

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 16)

        retriever = HybridRetriever(
            vector_index=mock_vector,
            sparse_index=mock_sparse,
            embedder=mock_embedder,
        )

        results = await retriever.retrieve("test query", top_k=10)

        # Should still return sparse results
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_retrieve_handles_sparse_failure(self) -> None:
        """Test retrieve continues when sparse search fails."""
        mock_sparse = MagicMock()
        mock_sparse.search = AsyncMock(side_effect=Exception("Sparse error"))

        mock_graph = MagicMock()
        mock_graph.search = AsyncMock(
            return_value=[
                SearchResult(chunk_id="c1", content="Graph result", score=0.9),
            ]
        )

        retriever = HybridRetriever(
            sparse_index=mock_sparse,
            graph_store=mock_graph,
        )

        results = await retriever.retrieve("test query", top_k=10)

        # Should still return graph results
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_handles_graph_failure(self) -> None:
        """Test retrieve continues when graph search fails."""
        mock_graph = MagicMock()
        mock_graph.search = AsyncMock(side_effect=Exception("Graph error"))

        mock_sparse = MagicMock()
        mock_sparse.search = AsyncMock(
            return_value=[
                SearchResult(chunk_id="c1", content="Sparse result", score=0.9),
            ]
        )

        retriever = HybridRetriever(
            sparse_index=mock_sparse,
            graph_store=mock_graph,
        )

        results = await retriever.retrieve("test query", top_k=10)

        # Should still return sparse results
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_uses_weighted_fusion(self) -> None:
        """Test retrieve uses weighted fusion when configured."""
        config = HybridRetrieverConfig(fusion_method="weighted")

        mock_sparse = MagicMock()
        mock_sparse.search = AsyncMock(
            return_value=[
                SearchResult(chunk_id="c1", content="Sparse result", score=0.9),
            ]
        )

        retriever = HybridRetriever(sparse_index=mock_sparse, config=config)
        results = await retriever.retrieve("test query", top_k=10)

        assert len(results) == 1
        # Weighted fusion applies weight
        assert results[0].score == pytest.approx(0.4 * 0.9, abs=0.01)


class TestCreateHybridRetriever:
    """Tests for create_hybrid_retriever factory function."""

    def test_create_hybrid_retriever_basic(self) -> None:
        """Test creating hybrid retriever with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from houyi.rag.retrieval import create_hybrid_retriever

            mock_embedder = MagicMock()

            with patch.object(embedding_module, "create_embedder", return_value=mock_embedder):
                retriever = create_hybrid_retriever(
                    knowledge_dir=tmpdir,
                    embedding_dimension=16,
                )

            assert retriever._vector_index is not None
            assert retriever._sparse_index is not None
            assert retriever._embedder is mock_embedder
            assert retriever._graph_store is None

    def test_create_hybrid_retriever_with_graph(self) -> None:
        """Test creating hybrid retriever with graph enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from houyi.rag.retrieval import create_hybrid_retriever

            mock_embedder = MagicMock()

            with patch.object(embedding_module, "create_embedder", return_value=mock_embedder):
                retriever = create_hybrid_retriever(
                    knowledge_dir=tmpdir,
                    embedding_dimension=16,
                    enable_graph=True,
                )

            assert retriever._vector_index is not None
            assert retriever._sparse_index is not None
            assert retriever._graph_store is not None
