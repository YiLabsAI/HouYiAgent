from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
from houyi.rag.indexed.ingest_pipeline import (
    extract_graph_entities,
    load_chunks_for_ingest,
    maybe_build_graph,
    resolve_embed_contents,
    update_retrieval_indexes,
)
from houyi.rag.indexed.models import RetrievalTaskResult
from houyi.rag.indexed.result_processing import (
    adjust_confidence,
    apply_crag,
    collect_sources,
    generate_answer,
    process_retrieval_results,
)
from houyi.rag.indexed.retrieval.dispatch import (
    RetrievalDispatchContext,
    create_retrieval_task,
)
from houyi.rag.indexed.retrieval.execution import (
    RetrievalExecutionRequest,
    execute_parallel_retrieval,
    execute_sequential_retrieval,
)
from houyi.rag.indexed.runtime_resources import IndexedRuntimeResources
from houyi.rag.indexed.search_backend import (
    graph_search,
    sparse_search,
    vector_search,
)
from houyi.rag.retrieval.fusion import rrf_fusion
from houyi.rag.types import RAGMode, RetrievalResult

if TYPE_CHECKING:
    from houyi.adapters.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class IndexedMode:
    """Runtime facade for indexed retrieval and ingest orchestration.

    `IndexedMode` coordinates indexed retrieval and ingest workflows while delegating
    resource lifecycle, retrieval dispatch, backend search, result post-processing,
    and ingest pipeline steps to dedicated collaborators.

    This facade remains responsible for wiring runtime collaborators and returning
    a stable public API. It should not absorb backend-specific initialization,
    retrieval dispatch policy, or ingest pipeline implementation details.
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
            index_dir: Index storage directory (default: `{knowledge_dir}/.houyi`).
                The public `RAG` entrypoint is expected to pass the effective value
                from `RAGConfig.get_index_dir()` so the facade and config share one
                storage contract.
        """
        self._config = config
        self._knowledge_dir = knowledge_dir
        self._index_dir = index_dir or str(Path(knowledge_dir) / ".houyi")
        self._embedding_config = embedding_config
        self._graph_config = graph_config
        self._llm_adapter = llm_adapter

        # Runtime resources are owned outside the facade so search / ingest orchestration
        # does not need to manage embedder/index lifecycle directly.
        self._resources = IndexedRuntimeResources(
            embedding_config=embedding_config,
            graph_config=graph_config,
            index_dir=self._index_dir,
        )

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

            from houyi.rag.generation.crag import CRAGValidator

            self._crag_validator = CRAGValidator(adapter=llm_adapter)

            from houyi.rag.indexed.document.contextualizer import Contextualizer

            self._contextualizer = Contextualizer(adapter=llm_adapter)

            from houyi.rag.processors.query_analyzer import QueryAnalyzer

            self._query_analyzer = QueryAnalyzer(adapter=llm_adapter)
        else:
            from houyi.rag.processors.query_analyzer import QueryAnalyzer

            # Query analysis remains available without an LLM so strategy selection and
            # metadata capture can still run in heuristic mode.
            self._query_analyzer = QueryAnalyzer()

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        enable_crag: bool = True,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute indexed retrieval orchestration.

        This facade method coordinates query analysis, retrieval execution,
        result fusion, optional reranking / CRAG validation, and final answer
        generation through indexed collaborators.

        Retrieval-time failures are handled as soft degradation where possible:
        strategy failures and timeouts are reflected in retrieval metadata, rerank /
        query-analysis / CRAG failures fall back to available results, and the final
        `RetrievalResult` preserves the indexed facade contract.

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

        await self._analyze_query(query, metadata)

        if self._config.parallel_retrieval:
            task_results = await self._execute_parallel_retrieval(query, k)
        else:
            task_results = await self._execute_sequential_retrieval(query, k)

        retrieval_metadata = process_retrieval_results(
            task_results=task_results,
            strategies_used=strategies_used,
            all_results=all_results,
            fallback_on_timeout=self._config.fallback_on_timeout,
        )
        metadata["retrieval"] = retrieval_metadata

        fused_results = self._rrf_fusion(all_results, k)

        if self._config.use_rerank and fused_results and self._reranker:
            try:
                fused_results = await self._reranker.rerank(query, fused_results, top_k=k)
            except Exception as e:
                logger.warning("LLM reranking failed: %s", e)

        fused_results, crag_quality = await apply_crag(
            validator=self._crag_validator,
            query=query,
            fused_results=fused_results,
            enable_crag=enable_crag,
            metadata=metadata,
        )

        answer, confidence = await generate_answer(
            answer_generator=self._answer_generator,
            query=query,
            results=fused_results,
        )
        confidence = adjust_confidence(
            confidence=confidence,
            crag_quality=crag_quality,
            retrieval_metadata=retrieval_metadata,
        )

        return RetrievalResult(
            answer=answer,
            sources=collect_sources(fused_results),
            confidence=confidence,
            search_results=fused_results,
            mode_used=RAGMode.INDEXED,
            strategies_used=strategies_used,
            metadata=metadata,
        )

    async def _execute_parallel_retrieval(self, query: str, k: int) -> list[RetrievalTaskResult]:
        """Execute indexed retrieval concurrently using a shared execution request boundary."""
        return await execute_parallel_retrieval(self._build_retrieval_execution_request(query, k))

    async def _execute_sequential_retrieval(self, query: str, k: int) -> list[RetrievalTaskResult]:
        """Execute indexed retrieval sequentially using the same request contract."""
        return await execute_sequential_retrieval(self._build_retrieval_execution_request(query, k))

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
            contextual_retrieval: Whether to use Contextual Retrieval.
                If no contextualizer is available or contextualization fails, ingest
                falls back to raw chunk content.
            progress_callback: Optional callback(processed, total, batch_size) for progress
            **kwargs: Additional options

        Returns:
            Ingestion statistics for documents, chunks, optional contextualization,
            and optional graph extraction.
        """
        stats = {
            "documents": 0,
            "chunks": 0,
            "contextualized_chunks": 0,
            "entities": 0,
            "relations": 0,
        }

        chunks = await load_chunks_for_ingest(paths=paths, stats=stats)

        if not chunks:
            logger.info("No chunks produced during ingest; skipping index updates")
            return stats

        embed_contents = await resolve_embed_contents(
            chunks=chunks,
            contextual_retrieval=contextual_retrieval,
            contextualizer=self._contextualizer,
            stats=stats,
        )
        await update_retrieval_indexes(
            chunks=chunks,
            embed_contents=embed_contents,
            progress_callback=progress_callback,
            resources=self._resources,
        )
        await maybe_build_graph(
            chunks=chunks,
            build_graph=build_graph,
            graph_enabled=self._graph_config.enabled,
            stats=stats,
            extract_graph_entities=self._extract_graph_entities,
            resources=self._resources,
        )

        return stats

    async def _analyze_query(self, query: str, metadata: dict[str, Any]) -> None:
        if not self._query_analyzer:
            return
        try:
            query_analysis = await self._query_analyzer.analyze(query)
            metadata["query_analysis"] = query_analysis.to_dict()
        except Exception as exc:
            logger.warning("Query analysis failed: %s", exc)

    def _build_retrieval_execution_request(
        self,
        query: str,
        k: int,
    ) -> RetrievalExecutionRequest:
        """Build the stable execution request shared by parallel and sequential retrieval.

        `IndexedMode` still owns the high-level collaborator wiring, but execution modes
        should consume one consistent request contract so timeout policy, enabled
        strategies, and dispatch wiring cannot drift between paths.
        """
        dispatch_context = self._build_retrieval_dispatch_context(query, k)
        return RetrievalExecutionRequest(
            strategies=self._config.strategies,
            timeout=self._config.retrieval_timeout,
            create_retrieval_task=lambda strategy: create_retrieval_task(
                strategy=strategy,
                context=dispatch_context,
            ),
            result_factory=RetrievalTaskResult,
        )

    def _build_retrieval_dispatch_context(self, query: str, k: int) -> RetrievalDispatchContext:
        return RetrievalDispatchContext(
            query=query,
            k=k,
            graph_enabled=self._graph_config.enabled,
            vector_search=lambda dispatch_query, dispatch_k: vector_search(
                query=dispatch_query,
                k=dispatch_k,
                resources=self._resources,
            ),
            sparse_search=lambda dispatch_query, dispatch_k: sparse_search(
                query=dispatch_query,
                k=dispatch_k,
                resources=self._resources,
            ),
            graph_search=lambda dispatch_query, dispatch_k: graph_search(
                query=dispatch_query,
                k=dispatch_k,
                resources=self._resources,
            ),
        )

    async def _extract_graph_entities(self, chunks: list[Any]) -> tuple[list[Any], list[Any]]:
        return await extract_graph_entities(chunks=chunks, entity_extractor=self._entity_extractor)

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
        return rrf_fusion(
            strategy_results=strategy_results,
            top_k=k,
            rrf_k=rrf_k,
        )
