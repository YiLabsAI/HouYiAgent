"""Hybrid retrieval implementation for RAG."""

from __future__ import annotations

import logging

from houyi.rag.retrieval.config import HybridRetrieverConfig
from houyi.rag.retrieval.fusion import accumulate_fusion_score, rrf_fusion
from houyi.rag.retrieval.protocols import QueryEmbedder, TextSearchBackend, VectorSearchBackend
from houyi.rag.types import SearchResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Reusable hybrid retriever over protocol-shaped retrieval collaborators.

    The retrieval domain owns the fusion/orchestration logic here, while concrete
    backends may still come from `indexed/` or any other mode as long as they satisfy
    the local retrieval protocols.
    """

    def __init__(
        self,
        vector_index: VectorSearchBackend | None = None,
        sparse_index: TextSearchBackend | None = None,
        graph_store: TextSearchBackend | None = None,
        embedder: QueryEmbedder | None = None,
        config: HybridRetrieverConfig | None = None,
    ) -> None:
        self._vector_index = vector_index
        self._sparse_index = sparse_index
        self._graph_store = graph_store
        self._embedder = embedder
        self._config = config or HybridRetrieverConfig()

    async def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        strategy_results: list[tuple[str, list[SearchResult]]] = []

        if self._vector_index is not None and self._embedder is not None:
            try:
                query_embedding = await self._embedder.embed(query)
                results = await self._vector_index.search(query_embedding, top_k)
                strategy_results.append(("vector", results))
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        if self._sparse_index is not None:
            try:
                results = await self._sparse_index.search(query, top_k)
                strategy_results.append(("sparse", results))
            except Exception as e:
                logger.warning("Sparse search failed: %s", e)

        if self._graph_store is not None:
            try:
                results = await self._graph_store.search(query, top_k)
                strategy_results.append(("graph", results))
            except Exception as e:
                logger.warning("Graph search failed: %s", e)

        if self._config.fusion_method == "weighted":
            return self._weighted_fusion(strategy_results, top_k)
        return self._rrf_fusion(strategy_results, top_k)

    def _rrf_fusion(
        self,
        strategy_results: list[tuple[str, list[SearchResult]]],
        top_k: int,
    ) -> list[SearchResult]:
        return rrf_fusion(
            strategy_results=strategy_results,
            top_k=top_k,
            rrf_k=self._config.rrf_k,
        )

    def _weighted_fusion(
        self,
        strategy_results: list[tuple[str, list[SearchResult]]],
        top_k: int,
    ) -> list[SearchResult]:
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
                accumulate_fusion_score(
                    result=result,
                    delta=weight * result.score,
                    scores=scores,
                    result_map=result_map,
                )

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused: list[SearchResult] = []
        for key in sorted_keys[:top_k]:
            result = result_map[key]
            result.score = scores[key]
            fused.append(result)
        return fused
