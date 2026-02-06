"""Indexed mode implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
from houyi.rag.types import RAGMode, RetrievalResult, RetrievalStrategy, SearchResult

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class IndexedMode:
    """Indexed RAG mode with pre-built vector/graph/sparse indexes.

    This mode provides traditional RAG capabilities:
    1. Vector retrieval using hnswlib
    2. Sparse retrieval using BM25
    3. Graph retrieval using PPR (optional)
    4. Hybrid fusion with RRF
    5. LLM reranking and answer generation
    """

    def __init__(
        self,
        config: IndexedConfig,
        knowledge_dir: str,
        embedding_config: EmbeddingConfig,
        graph_config: GraphConfig,
        llm_adapter: LLMAdapter | None = None,
    ) -> None:
        """Initialize Indexed mode.

        Args:
            config: Indexed mode configuration
            knowledge_dir: Knowledge base directory
            embedding_config: Embedding configuration
            graph_config: Graph retrieval configuration
            llm_adapter: Optional LLM adapter for reranking and generation
        """
        self._config = config
        self._knowledge_dir = knowledge_dir
        self._embedding_config = embedding_config
        self._graph_config = graph_config
        self._llm_adapter = llm_adapter

        # Lazy initialization of components
        self._vector_index: Any = None
        self._sparse_index: Any = None
        self._graph_store: Any = None
        self._embedder: Any = None

        # Initialize LLM components if adapter is provided
        self._reranker = None
        self._answer_generator = None
        self._entity_extractor = None
        if llm_adapter:
            from houyi.rag.llm import AnswerGenerator, LLMEntityExtractor, LLMReranker

            self._reranker = LLMReranker(llm_adapter)
            self._answer_generator = AnswerGenerator(llm_adapter)
            self._entity_extractor = LLMEntityExtractor(llm_adapter)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute indexed search.

        The search process:
        1. Analyze query to select strategies
        2. Execute parallel retrieval across strategies
        3. Fuse results with RRF
        4. Rerank with LLM (if available)
        5. Generate answer with LLM (if available)

        Args:
            query: User query string
            top_k: Override number of results
            **kwargs: Additional options

        Returns:
            RetrievalResult with answer and sources
        """
        k = top_k or self._config.top_k
        strategies_used = []
        all_results = []

        # Execute enabled strategies
        for strategy in self._config.strategies:
            if strategy == RetrievalStrategy.VECTOR:
                results = await self._vector_search(query, k)
                strategies_used.append(strategy)
                all_results.append(("vector", results))

            elif strategy == RetrievalStrategy.BM25:
                results = await self._sparse_search(query, k)
                strategies_used.append(strategy)
                all_results.append(("bm25", results))

            elif strategy == RetrievalStrategy.GRAPH and self._graph_config.enabled:
                results = await self._graph_search(query, k)
                strategies_used.append(strategy)
                all_results.append(("graph", results))

        # Fuse results with RRF
        fused_results = self._rrf_fusion(all_results, k)

        # LLM rerank if enabled and available
        if self._config.use_rerank and fused_results and self._reranker:
            try:
                fused_results = await self._reranker.rerank(query, fused_results, top_k=k)
            except Exception as e:
                logger.warning("LLM reranking failed: %s", e)

        # Generate answer
        answer, confidence = await self._generate_answer(query, fused_results)

        # Build response
        sources = [r.source for r in fused_results if r.source]

        return RetrievalResult(
            answer=answer,
            sources=sources[:10],
            confidence=confidence,
            search_results=fused_results,
            mode_used=RAGMode.INDEXED,
            strategies_used=strategies_used,
        )

    async def ingest(
        self,
        paths: list[str],
        build_graph: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ingest documents into indexes.

        Args:
            paths: Document paths to ingest
            build_graph: Whether to build knowledge graph
            **kwargs: Additional options

        Returns:
            Ingestion statistics
        """
        stats = {
            "documents": 0,
            "chunks": 0,
            "entities": 0,
            "relations": 0,
        }

        # Load and process documents
        from houyi.rag.indexed.document.loaders import load_documents
        from houyi.rag.indexed.document.splitters import split_documents

        documents = await load_documents(paths)
        stats["documents"] = len(documents)

        chunks = await split_documents(documents)
        stats["chunks"] = len(chunks)

        # Generate embeddings
        await self._ensure_embedder()
        embeddings = await self._embedder.embed_batch([c.content for c in chunks])

        # Update vector index
        await self._ensure_vector_index()
        await self._vector_index.add_batch(chunks, embeddings)

        # Update sparse index
        await self._ensure_sparse_index()
        await self._sparse_index.add_batch(chunks)

        # Optionally build graph with LLM entity extraction
        if build_graph and self._graph_config.enabled:
            await self._ensure_graph_store()

            if self._entity_extractor:
                # Use LLM for entity extraction
                try:
                    entities, relations = await self._entity_extractor.extract_batch(chunks)
                    stats["entities"] = len(entities)
                    stats["relations"] = len(relations)

                    await self._graph_store.add_entities(entities)
                    await self._graph_store.add_relations(relations)
                except Exception as e:
                    logger.warning("LLM entity extraction failed: %s, using simple", e)
                    # Fall back to simple extraction
                    from houyi.rag.indexed.graph.extractor import extract_entities

                    entities, relations = await extract_entities(chunks)
                    stats["entities"] = len(entities)
                    stats["relations"] = len(relations)

                    await self._graph_store.add_entities(entities)
                    await self._graph_store.add_relations(relations)
            else:
                # Use simple rule-based extraction
                from houyi.rag.indexed.graph.extractor import extract_entities

                entities, relations = await extract_entities(chunks)
                stats["entities"] = len(entities)
                stats["relations"] = len(relations)

                await self._graph_store.add_entities(entities)
                await self._graph_store.add_relations(relations)

        return stats

    async def _vector_search(self, query: str, k: int) -> list[Any]:
        """Search vector index."""
        await self._ensure_embedder()
        await self._ensure_vector_index()

        query_embedding = await self._embedder.embed(query)
        return await self._vector_index.search(query_embedding, k)

    async def _sparse_search(self, query: str, k: int) -> list[Any]:
        """Search sparse (BM25) index."""
        await self._ensure_sparse_index()
        return await self._sparse_index.search(query, k)

    async def _graph_search(self, query: str, k: int) -> list[Any]:
        """Search knowledge graph using PPR."""
        await self._ensure_graph_store()
        return await self._graph_store.search(query, k)

    async def _generate_answer(
        self,
        query: str,
        results: list[SearchResult],
    ) -> tuple[str, float]:
        """Generate answer using LLM if available."""
        if not results:
            return "未找到相关信息。", 0.0

        if self._answer_generator:
            try:
                answer, confidence = await self._answer_generator.generate(
                    query=query,
                    results=results,
                    include_sources=True,
                )
                return answer, confidence
            except Exception as e:
                logger.warning("LLM answer generation failed: %s", e)

        # Fallback to simple concatenation
        return self._build_answer_simple(results), min(len(results) * 0.1, 0.7)

    def _build_answer_simple(self, results: list[SearchResult]) -> str:
        """Build answer by concatenating results (fallback when LLM unavailable)."""
        contents = []
        for r in results[:5]:
            if r.content:
                contents.append(r.content.strip())

        if not contents:
            return "未找到相关信息。"

        return "\n\n---\n\n".join(contents)

    def _rrf_fusion(
        self,
        strategy_results: list[tuple[str, list[Any]]],
        k: int,
        rrf_k: int = 60,
    ) -> list[Any]:
        """Fuse results from multiple strategies using RRF.

        RRF score(d) = Σ 1/(k + rank_i(d))

        Args:
            strategy_results: List of (strategy_name, results) tuples
            k: Number of final results
            rrf_k: RRF constant (default 60)

        Returns:
            Fused and sorted results
        """
        if not strategy_results:
            return []

        # Single strategy - no fusion needed
        if len(strategy_results) == 1:
            return strategy_results[0][1][:k]

        # Calculate RRF scores
        scores: dict[str, float] = {}
        result_map: dict[str, Any] = {}

        for _, results in strategy_results:
            for rank, result in enumerate(results, 1):
                # Use chunk_id or content hash as key
                key = result.chunk_id or hash(result.content)
                if key not in scores:
                    scores[key] = 0.0
                    result_map[key] = result

                scores[key] += 1.0 / (rrf_k + rank)

        # Sort by RRF score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Update scores in results
        fused = []
        for key in sorted_keys[:k]:
            result = result_map[key]
            result.score = scores[key]
            fused.append(result)

        return fused

    async def _ensure_embedder(self) -> None:
        """Ensure embedder is initialized."""
        if self._embedder is None:
            from houyi.rag.indexed.embedding import create_embedder

            self._embedder = create_embedder(self._embedding_config)

    async def _ensure_vector_index(self) -> None:
        """Ensure vector index is initialized."""
        if self._vector_index is None:
            from houyi.rag.indexed.index.vector import VectorIndex

            self._vector_index = VectorIndex(
                dimension=self._embedding_config.dimension,
                knowledge_dir=self._knowledge_dir,
            )
            await self._vector_index.load()

    async def _ensure_sparse_index(self) -> None:
        """Ensure sparse index is initialized."""
        if self._sparse_index is None:
            from houyi.rag.indexed.index.sparse import SparseIndex

            self._sparse_index = SparseIndex(
                knowledge_dir=self._knowledge_dir,
            )
            await self._sparse_index.load()

    async def _ensure_graph_store(self) -> None:
        """Ensure graph store is initialized."""
        if self._graph_store is None:
            from houyi.rag.indexed.graph.store import GraphStore

            self._graph_store = GraphStore(
                knowledge_dir=self._knowledge_dir,
                config=self._graph_config,
            )
            await self._graph_store.load()
