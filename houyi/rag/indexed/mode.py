"""Indexed mode implementation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
from houyi.rag.types import RAGMode, RetrievalResult, RetrievalStrategy, SearchResult

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTaskResult:
    """Result of a single retrieval task."""

    strategy: RetrievalStrategy
    strategy_name: str
    results: list[Any] = field(default_factory=list)
    success: bool = True
    timed_out: bool = False
    error: str | None = None
    duration_ms: float = 0.0


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
        index_dir: str | None = None,
    ) -> None:
        """Initialize Indexed mode.

        Args:
            config: Indexed mode configuration
            knowledge_dir: Knowledge base source directory
            embedding_config: Embedding configuration
            graph_config: Graph retrieval configuration
            llm_adapter: Optional LLM adapter for reranking and generation
            index_dir: Index storage directory (default: {knowledge_dir}/.houyi)
        """
        self._config = config
        self._knowledge_dir = knowledge_dir
        # Use separate index_dir if provided, otherwise default to knowledge_dir
        self._index_dir = index_dir or knowledge_dir
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
        self._crag_validator = None
        self._contextualizer = None
        self._query_analyzer = None
        if llm_adapter:
            from houyi.rag.llm import AnswerGenerator, LLMEntityExtractor, LLMReranker

            self._reranker = LLMReranker(llm_adapter)
            self._answer_generator = AnswerGenerator(llm_adapter)
            self._entity_extractor = LLMEntityExtractor(llm_adapter)

            # Initialize CRAG validator
            from houyi.rag.generation.crag import CRAGValidator

            self._crag_validator = CRAGValidator(adapter=llm_adapter)

            # Initialize Contextualizer for Contextual Retrieval
            from houyi.rag.indexed.document.contextualizer import Contextualizer

            self._contextualizer = Contextualizer(adapter=llm_adapter)

            # Initialize Query Analyzer
            from houyi.rag.processors.query_analyzer import QueryAnalyzer

            self._query_analyzer = QueryAnalyzer(adapter=llm_adapter)
        else:
            # Initialize Query Analyzer without LLM (heuristic mode)
            from houyi.rag.processors.query_analyzer import QueryAnalyzer

            self._query_analyzer = QueryAnalyzer()

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        enable_crag: bool = True,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute indexed search.

        The search process:
        1. Analyze query to select strategies
        2. Execute parallel retrieval across strategies (with timeout)
        3. Fuse results with RRF
        4. Rerank with LLM (if available)
        5. Validate with CRAG (if enabled and LLM available)
        6. Generate answer with LLM (if available)

        Args:
            query: User query string
            top_k: Override number of results
            enable_crag: Whether to enable CRAG validation (default: True)
            **kwargs: Additional options

        Returns:
            RetrievalResult with answer and sources
        """
        k = top_k or self._config.top_k
        strategies_used = []
        all_results = []
        metadata: dict[str, Any] = {}

        # Analyze query for strategy selection and UI display
        if self._query_analyzer:
            try:
                query_analysis = await self._query_analyzer.analyze(query)
                metadata["query_analysis"] = query_analysis.to_dict()
            except Exception as e:
                logger.warning("Query analysis failed: %s", e)

        # Execute retrieval strategies (parallel or sequential)
        if self._config.parallel_retrieval:
            task_results = await self._execute_parallel_retrieval(query, k)
        else:
            task_results = await self._execute_sequential_retrieval(query, k)

        # Process retrieval results
        retrieval_metadata = self._process_retrieval_results(
            task_results, strategies_used, all_results
        )
        metadata["retrieval"] = retrieval_metadata

        # Fuse results with RRF
        fused_results = self._rrf_fusion(all_results, k)

        # LLM rerank if enabled and available
        if self._config.use_rerank and fused_results and self._reranker:
            try:
                fused_results = await self._reranker.rerank(query, fused_results, top_k=k)
            except Exception as e:
                logger.warning("LLM reranking failed: %s", e)

        # CRAG validation if enabled and available
        crag_quality = None
        if enable_crag and self._crag_validator and fused_results:
            try:
                crag_result = await self._crag_validator.validate(query, fused_results)
                crag_quality = crag_result.quality.value
                metadata["crag_quality"] = crag_quality
                metadata["crag_confidence"] = crag_result.confidence
                metadata["crag_reasoning"] = crag_result.reasoning

                # Use only relevant results if CRAG filtered some out
                if crag_result.relevant_results:
                    fused_results = crag_result.relevant_results
                    logger.debug(
                        "CRAG filtered results: %d -> %d (quality: %s)",
                        len(fused_results),
                        len(crag_result.relevant_results),
                        crag_quality,
                    )
            except Exception as e:
                logger.warning("CRAG validation failed: %s", e)

        # Generate answer
        answer, confidence = await self._generate_answer(query, fused_results)

        # Adjust confidence based on CRAG quality
        if crag_quality == "incorrect":
            confidence = min(confidence, 0.3)
        elif crag_quality == "ambiguous":
            confidence = min(confidence, 0.6)

        # Adjust confidence if some strategies timed out
        if retrieval_metadata.get("timed_out_count", 0) > 0:
            confidence = min(confidence, 0.7)

        # Build response
        sources = [r.source for r in fused_results if r.source]

        return RetrievalResult(
            answer=answer,
            sources=sources[:10],
            confidence=confidence,
            search_results=fused_results,
            mode_used=RAGMode.INDEXED,
            strategies_used=strategies_used,
            metadata=metadata,
        )

    async def _execute_parallel_retrieval(self, query: str, k: int) -> list[RetrievalTaskResult]:
        """Execute retrieval strategies in parallel with timeout.

        Args:
            query: User query string
            k: Number of results per strategy

        Returns:
            List of retrieval task results
        """
        import time

        tasks = []
        task_info = []

        for strategy in self._config.strategies:
            if strategy == RetrievalStrategy.VECTOR:
                tasks.append(self._vector_search(query, k))
                task_info.append((strategy, "vector"))
            elif strategy == RetrievalStrategy.BM25:
                tasks.append(self._sparse_search(query, k))
                task_info.append((strategy, "bm25"))
            elif strategy == RetrievalStrategy.GRAPH and self._graph_config.enabled:
                tasks.append(self._graph_search(query, k))
                task_info.append((strategy, "graph"))

        if not tasks:
            return []

        timeout = self._config.retrieval_timeout
        results = []

        # Execute all tasks with timeout
        start_time = time.time()

        try:
            done, pending = await asyncio.wait(
                [asyncio.create_task(t) for t in tasks],
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            elapsed = (time.time() - start_time) * 1000  # ms

            # Process completed tasks
            completed_indices = set()
            for task in done:
                # Find the index of this task
                for i, (strategy, name) in enumerate(task_info):
                    if i not in completed_indices:
                        try:
                            task_results = task.result()
                            results.append(
                                RetrievalTaskResult(
                                    strategy=strategy,
                                    strategy_name=name,
                                    results=task_results,
                                    success=True,
                                    duration_ms=elapsed,
                                )
                            )
                            completed_indices.add(i)
                            break
                        except Exception as e:
                            results.append(
                                RetrievalTaskResult(
                                    strategy=strategy,
                                    strategy_name=name,
                                    success=False,
                                    error=str(e),
                                    duration_ms=elapsed,
                                )
                            )
                            completed_indices.add(i)
                            break

            # Handle timed out tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Mark timed out strategies
            for i, (strategy, name) in enumerate(task_info):
                if i not in completed_indices:
                    results.append(
                        RetrievalTaskResult(
                            strategy=strategy,
                            strategy_name=name,
                            success=False,
                            timed_out=True,
                            error=f"Timeout after {timeout}s",
                            duration_ms=timeout * 1000,
                        )
                    )
                    logger.warning("Strategy %s timed out after %s seconds", name, timeout)

        except Exception as e:
            logger.error("Parallel retrieval failed: %s", e)
            # Return empty results for all strategies
            for strategy, name in task_info:
                results.append(
                    RetrievalTaskResult(
                        strategy=strategy,
                        strategy_name=name,
                        success=False,
                        error=str(e),
                    )
                )

        return results

    async def _execute_sequential_retrieval(self, query: str, k: int) -> list[RetrievalTaskResult]:
        """Execute retrieval strategies sequentially.

        Args:
            query: User query string
            k: Number of results per strategy

        Returns:
            List of retrieval task results
        """
        import time

        results = []
        timeout = self._config.retrieval_timeout

        for strategy in self._config.strategies:
            start_time = time.time()

            if strategy == RetrievalStrategy.VECTOR:
                name = "vector"
                coro = self._vector_search(query, k)
            elif strategy == RetrievalStrategy.BM25:
                name = "bm25"
                coro = self._sparse_search(query, k)
            elif strategy == RetrievalStrategy.GRAPH and self._graph_config.enabled:
                name = "graph"
                coro = self._graph_search(query, k)
            else:
                continue

            try:
                task_results = await asyncio.wait_for(coro, timeout=timeout)
                elapsed = (time.time() - start_time) * 1000
                results.append(
                    RetrievalTaskResult(
                        strategy=strategy,
                        strategy_name=name,
                        results=task_results,
                        success=True,
                        duration_ms=elapsed,
                    )
                )
            except asyncio.TimeoutError:
                elapsed = (time.time() - start_time) * 1000
                results.append(
                    RetrievalTaskResult(
                        strategy=strategy,
                        strategy_name=name,
                        success=False,
                        timed_out=True,
                        error=f"Timeout after {timeout}s",
                        duration_ms=elapsed,
                    )
                )
                logger.warning("Strategy %s timed out after %s seconds", name, timeout)
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                results.append(
                    RetrievalTaskResult(
                        strategy=strategy,
                        strategy_name=name,
                        success=False,
                        error=str(e),
                        duration_ms=elapsed,
                    )
                )
                logger.warning("Strategy %s failed: %s", name, e)

        return results

    def _process_retrieval_results(
        self,
        task_results: list[RetrievalTaskResult],
        strategies_used: list[RetrievalStrategy],
        all_results: list[tuple[str, list[Any]]],
    ) -> dict[str, Any]:
        """Process retrieval task results and build metadata.

        Args:
            task_results: Results from retrieval tasks
            strategies_used: List to append successful strategies to
            all_results: List to append results tuples to

        Returns:
            Retrieval metadata dictionary
        """
        metadata: dict[str, Any] = {
            "total_strategies": len(task_results),
            "successful_count": 0,
            "failed_count": 0,
            "timed_out_count": 0,
            "strategy_details": [],
        }

        for result in task_results:
            detail = {
                "strategy": result.strategy.value,
                "name": result.strategy_name,
                "success": result.success,
                "timed_out": result.timed_out,
                "result_count": len(result.results),
                "duration_ms": result.duration_ms,
            }
            if result.error:
                detail["error"] = result.error

            metadata["strategy_details"].append(detail)

            if result.success:
                metadata["successful_count"] += 1
                strategies_used.append(result.strategy)
                all_results.append((result.strategy_name, result.results))
            else:
                metadata["failed_count"] += 1
                if result.timed_out:
                    metadata["timed_out_count"] += 1

        # Check fallback behavior
        if not self._config.fallback_on_timeout and metadata["timed_out_count"] > 0:
            # If fallback is disabled and any strategy timed out,
            # don't use partial results
            logger.warning(
                "Fallback disabled and %d strategies timed out, clearing results",
                metadata["timed_out_count"],
            )
            strategies_used.clear()
            all_results.clear()

        return metadata

    async def ingest(
        self,
        paths: list[str],
        build_graph: bool = False,
        contextual_retrieval: bool = False,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ingest documents into indexes.

        Args:
            paths: Document paths to ingest
            build_graph: Whether to build knowledge graph
            contextual_retrieval: Whether to use Contextual Retrieval (requires LLM)
            progress_callback: Optional callback(processed, total, batch_size) for progress
            **kwargs: Additional options

        Returns:
            Ingestion statistics
        """
        stats = {
            "documents": 0,
            "chunks": 0,
            "contextualized_chunks": 0,
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

        # Apply Contextual Retrieval if enabled and LLM available
        embed_contents = [c.content for c in chunks]
        if contextual_retrieval and self._contextualizer:
            try:
                logger.info("Applying Contextual Retrieval to %d chunks", len(chunks))
                contextualized = await self._contextualizer.contextualize_chunks(chunks)
                # Use contextualized content for embedding
                embed_contents = [c.contextualized_content for c in contextualized]
                stats["contextualized_chunks"] = len(contextualized)
                logger.debug("Contextualized %d chunks", len(contextualized))
            except Exception as e:
                logger.warning("Contextual Retrieval failed: %s, using raw content", e)

        # Generate embeddings with progress reporting
        await self._ensure_embedder()
        embeddings = await self._embedder.embed_batch(
            embed_contents,
            progress_callback=progress_callback,
        )

        # Update vector index
        await self._ensure_vector_index()
        await self._vector_index.add_batch(chunks, embeddings)
        await self._vector_index.save()

        # Update sparse index
        await self._ensure_sparse_index()
        await self._sparse_index.add_batch(chunks)
        await self._sparse_index.save()

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

            # Save graph after building
            await self._graph_store.save()

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
            return "No relevant information found.", 0.0

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
            return "No relevant information found."

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
                knowledge_dir=self._index_dir,  # Use index_dir for storage
            )
            await self._vector_index.load()

    async def _ensure_sparse_index(self) -> None:
        """Ensure sparse index is initialized."""
        if self._sparse_index is None:
            from houyi.rag.indexed.index.sparse import SparseIndex

            self._sparse_index = SparseIndex(
                knowledge_dir=self._index_dir,  # Use index_dir for storage
            )
            await self._sparse_index.load()

    async def _ensure_graph_store(self) -> None:
        """Ensure graph store is initialized."""
        if self._graph_store is None:
            from houyi.rag.indexed.graph.store import GraphStore

            self._graph_store = GraphStore(
                knowledge_dir=self._index_dir,  # Use index_dir for storage
                config=self._graph_config,
            )
            await self._graph_store.load()
