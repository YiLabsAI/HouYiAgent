"""RAG retrieval components and protocols.

This module provides:
- Protocol definitions for retrieval components (Retriever, Reranker, etc.)
- HybridRetriever: Multi-strategy retrieval with RRF fusion

These are internal components used by IndexedMode. For public API, use the RAG class.

Example (internal usage):
    from houyi.rag.retrieval import HybridRetriever

    retriever = HybridRetriever(
        vector_index=VectorIndex(dimension=384),
        sparse_index=SparseIndex(),
        fusion_method="rrf",
    )
    results = await retriever.retrieve("What is RAG?", top_k=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from houyi.rag.types import SearchResult

if TYPE_CHECKING:
    from houyi.rag.indexed.graph.store import GraphStore
    from houyi.rag.indexed.index.sparse import SparseIndex
    from houyi.rag.indexed.index.vector import VectorIndex

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols for retrieval components
# ============================================================================


class Retriever(Protocol):
    """Protocol for retrieval components."""

    async def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve relevant results for a query."""
        ...


class Reranker(Protocol):
    """Protocol for reranking components."""

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 10
    ) -> list[SearchResult]:
        """Rerank search results."""
        ...


class Generator(Protocol):
    """Protocol for answer generation components."""

    async def generate(self, query: str, results: list[SearchResult]) -> tuple[str, float]:
        """Generate answer from results. Returns (answer, confidence)."""
        ...


class Validator(Protocol):
    """Protocol for result validation components (e.g., CRAG)."""

    async def validate(
        self, query: str, results: list[SearchResult]
    ) -> tuple[list[SearchResult], bool]:
        """Validate and filter results. Returns (filtered_results, needs_web_search)."""
        ...


# ============================================================================
# HybridRetriever - Multi-strategy retrieval with RRF fusion
# ============================================================================


@dataclass
class HybridRetrieverConfig:
    """Configuration for HybridRetriever."""

    fusion_method: str = "rrf"  # "rrf" or "weighted"
    rrf_k: int = 60
    vector_weight: float = 0.4
    sparse_weight: float = 0.4
    graph_weight: float = 0.2


class HybridRetriever:
    """Hybrid retriever combining vector, sparse, and graph search.

    Supports multiple fusion strategies:
    - RRF (Reciprocal Rank Fusion): Default, parameter-free
    - Weighted: Linear combination with configurable weights

    Example:
        retriever = HybridRetriever(
            vector_index=VectorIndex(dimension=384),
            sparse_index=SparseIndex(),
            fusion_method="rrf",
        )
        results = await retriever.retrieve("What is RAG?", top_k=10)
    """

    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        sparse_index: SparseIndex | None = None,
        graph_store: GraphStore | None = None,
        embedder: Any = None,
        config: HybridRetrieverConfig | None = None,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            vector_index: Vector index for semantic search
            sparse_index: Sparse index for BM25 search
            graph_store: Graph store for PPR-based search
            embedder: Embedding model for query encoding
            config: Retriever configuration
        """
        self._vector_index = vector_index
        self._sparse_index = sparse_index
        self._graph_store = graph_store
        self._embedder = embedder
        self._config = config or HybridRetrieverConfig()

    async def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Execute hybrid retrieval.

        Args:
            query: User query
            top_k: Number of results to return

        Returns:
            Fused and sorted search results
        """
        strategy_results: list[tuple[str, list[SearchResult]]] = []

        # Vector search
        if self._vector_index is not None and self._embedder is not None:
            try:
                query_embedding = await self._embedder.embed(query)
                results = await self._vector_index.search(query_embedding, top_k)
                strategy_results.append(("vector", results))
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        # Sparse search
        if self._sparse_index is not None:
            try:
                results = await self._sparse_index.search(query, top_k)
                strategy_results.append(("sparse", results))
            except Exception as e:
                logger.warning("Sparse search failed: %s", e)

        # Graph search
        if self._graph_store is not None:
            try:
                results = await self._graph_store.search(query, top_k)
                strategy_results.append(("graph", results))
            except Exception as e:
                logger.warning("Graph search failed: %s", e)

        # Fuse results
        if self._config.fusion_method == "weighted":
            return self._weighted_fusion(strategy_results, top_k)
        else:
            return self._rrf_fusion(strategy_results, top_k)

    def _rrf_fusion(
        self,
        strategy_results: list[tuple[str, list[SearchResult]]],
        top_k: int,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion.

        RRF score(d) = Σ 1/(k + rank_i(d))
        """
        if not strategy_results:
            return []

        if len(strategy_results) == 1:
            return strategy_results[0][1][:top_k]

        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}
        rrf_k = self._config.rrf_k

        for _, results in strategy_results:
            for rank, result in enumerate(results, 1):
                key = result.chunk_id or str(hash(result.content))
                if key not in scores:
                    scores[key] = 0.0
                    result_map[key] = result
                scores[key] += 1.0 / (rrf_k + rank)

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused = []
        for key in sorted_keys[:top_k]:
            result = result_map[key]
            result.score = scores[key]
            fused.append(result)

        return fused

    def _weighted_fusion(
        self,
        strategy_results: list[tuple[str, list[SearchResult]]],
        top_k: int,
    ) -> list[SearchResult]:
        """Weighted linear combination fusion."""
        if not strategy_results:
            return []

        weight_map = {
            "vector": self._config.vector_weight,
            "sparse": self._config.sparse_weight,
            "graph": self._config.graph_weight,
        }

        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for strategy_name, results in strategy_results:
            weight = weight_map.get(strategy_name, 0.33)
            for result in results:
                key = result.chunk_id or str(hash(result.content))
                if key not in scores:
                    scores[key] = 0.0
                    result_map[key] = result
                scores[key] += weight * result.score

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused = []
        for key in sorted_keys[:top_k]:
            result = result_map[key]
            result.score = scores[key]
            fused.append(result)

        return fused


# ============================================================================
# Factory functions
# ============================================================================


def create_hybrid_retriever(
    knowledge_dir: str,
    embedding_dimension: int = 384,
    enable_graph: bool = False,
) -> HybridRetriever:
    """Create a HybridRetriever with default configuration.

    Args:
        knowledge_dir: Knowledge base directory
        embedding_dimension: Embedding dimension
        enable_graph: Whether to enable graph retrieval

    Returns:
        Configured HybridRetriever
    """
    from houyi.rag.config import EmbeddingConfig
    from houyi.rag.indexed.embedding import create_embedder
    from houyi.rag.indexed.index.sparse import SparseIndex
    from houyi.rag.indexed.index.vector import VectorIndex

    vector_index = VectorIndex(dimension=embedding_dimension, knowledge_dir=knowledge_dir)
    sparse_index = SparseIndex(knowledge_dir=knowledge_dir)
    embedding_config = EmbeddingConfig(
        provider="local",
        dimension=embedding_dimension,
    )
    embedder = create_embedder(embedding_config)

    graph_store = None
    if enable_graph:
        from houyi.rag.indexed.graph.store import GraphStore

        graph_store = GraphStore(knowledge_dir=knowledge_dir)

    return HybridRetriever(
        vector_index=vector_index,
        sparse_index=sparse_index,
        graph_store=graph_store,
        embedder=embedder,
    )
