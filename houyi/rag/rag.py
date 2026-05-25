"""RAG - Unified entry point for Houyi RAG.

This is the primary public API for Houyi RAG, providing a simple interface
for both Agentic and Indexed modes.

Example:
    # Zero config (Agentic mode)
    rag = RAG("./docs")
    result = await rag.query("What is RAG?")

    # With indexing (Indexed mode)
    rag = RAG("./docs", mode="indexed")
    await rag.index()
    result = await rag.query("What is RAG?")

    # Advanced config
    rag = RAG(
        "./docs",
        mode="indexed",
        strategies=["bm25", "vector", "graph"],
        llm="openai:gpt-4o-mini",
    )
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from houyi.rag.config import RAGConfig
from houyi.rag.types import RAGMode, RetrievalResult, RetrievalStrategy

if TYPE_CHECKING:
    from houyi.adapters.llm.base import LLMAdapter
    from houyi.application.runtime.agent import Agent

logger = logging.getLogger(__name__)


class RAG:
    """Houyi RAG - Unified API for Agentic and Indexed retrieval.

    This facade provides the stable public entry point for all RAG operations:
    - index(): Build indexes for Indexed mode
    - query(): Execute retrieval queries
    - add(): Add documents to existing index
    - refresh(): Rebuild indexes

    The facade owns config normalization, mode selection, and lazy runtime
    creation for Agentic and Indexed modes. Concrete retrieval / ingest logic
    stays in the respective subdomains once a mode instance is constructed.

    Example:
        # Simple usage (Agentic mode, no indexing needed)
        rag = RAG("./docs")
        result = await rag.query("What is RAG?")

        # Indexed mode with explicit indexing
        rag = RAG("./docs", mode="indexed")
        await rag.index()
        result = await rag.query("What is RAG?")

        # With LLM for enhanced capabilities
        rag = RAG("./docs", mode="indexed", llm="openai:gpt-4o-mini")
    """

    def __init__(
        self,
        knowledge_dir: str | RAGConfig | None = None,
        *,
        config: RAGConfig | None = None,
        mode: str | RAGMode = RAGMode.AUTO,
        strategies: list[str] | None = None,
        agent: Agent | None = None,
        llm: LLMAdapter | str | None = None,
        llm_adapter: LLMAdapter | None = None,  # Backward compatibility
        llm_provider: str | None = None,  # Backward compatibility
        llm_model: str | None = None,  # Backward compatibility
        hooks: Any = None,
        contextual_retrieval: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize RAG.

        Args:
            knowledge_dir: Knowledge base directory path (or RAGConfig for backward compat)
            config: Full RAG configuration (overrides other params)
            mode: RAG mode ("agentic", "indexed", "auto")
            strategies: List of retrieval strategies for indexed mode
            agent: Optional Agent instance for Agentic mode (reuses tools)
            llm: LLM adapter or provider string (e.g., "openai:gpt-4o-mini")
            llm_adapter: Backward compatibility - use llm instead
            llm_provider: Backward compatibility - use llm instead
            llm_model: Backward compatibility - use llm instead
            hooks: Optional RAGHooks instance for customization
            contextual_retrieval: Enable Contextual Retrieval for indexing
            **kwargs: Additional config options
        """
        config, knowledge_dir = self._normalize_config_input(
            config=config, knowledge_dir=knowledge_dir
        )
        resolved_llm_adapter, resolved_llm_provider, resolved_llm_model = self._resolve_llm_inputs(
            llm=llm,
            llm_adapter=llm_adapter,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        self._config = self._build_config(
            config=config,
            knowledge_dir=knowledge_dir,
            mode=mode,
            strategies=strategies,
            contextual_retrieval=contextual_retrieval,
            llm_model=resolved_llm_model,
            extra_kwargs=kwargs,
        )

        self._agent = agent
        self._llm_adapter = resolved_llm_adapter
        self._hooks = hooks
        self._agentic_mode: Any = None
        self._indexed_mode: Any = None

        # Auto-create LLM adapter if provider specified
        if resolved_llm_adapter is None and resolved_llm_provider:
            self._llm_adapter = self._create_llm_adapter(resolved_llm_provider, resolved_llm_model)

    @staticmethod
    def _normalize_config_input(
        *,
        config: RAGConfig | None,
        knowledge_dir: str | RAGConfig | None,
    ) -> tuple[RAGConfig | None, str | None]:
        if isinstance(knowledge_dir, RAGConfig):
            config = knowledge_dir
            return config, config.knowledge_dir
        return config, knowledge_dir

    @staticmethod
    def _resolve_llm_inputs(
        *,
        llm: LLMAdapter | str | None,
        llm_adapter: LLMAdapter | None,
        llm_provider: str | None,
        llm_model: str | None,
    ) -> tuple[LLMAdapter | None, str | None, str | None]:
        resolved_adapter = llm_adapter
        resolved_provider = llm_provider
        resolved_model = llm_model

        if llm is None:
            return resolved_adapter, resolved_provider, resolved_model
        if isinstance(llm, str):
            if ":" in llm:
                resolved_provider, resolved_model = llm.split(":", 1)
            else:
                resolved_provider = llm
            return resolved_adapter, resolved_provider, resolved_model
        return llm, resolved_provider, resolved_model

    @staticmethod
    def _build_config_kwargs(
        *,
        llm_model: str | None,
        contextual_retrieval: bool,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        config_kwargs = {**extra_kwargs}
        if llm_model:
            config_kwargs["llm_model"] = llm_model
        if contextual_retrieval:
            config_kwargs["contextual_retrieval"] = contextual_retrieval
        return config_kwargs

    @staticmethod
    def _normalize_mode(mode: str | RAGMode) -> RAGMode:
        return RAGMode(mode) if isinstance(mode, str) else mode

    @staticmethod
    def _normalize_strategies(strategies: list[str] | None) -> list[RetrievalStrategy] | None:
        if not strategies:
            return None
        return [RetrievalStrategy(strategy) for strategy in strategies]

    @classmethod
    def _build_config(
        cls,
        *,
        config: RAGConfig | None,
        knowledge_dir: str | None,
        mode: str | RAGMode,
        strategies: list[str] | None,
        contextual_retrieval: bool,
        llm_model: str | None,
        extra_kwargs: dict[str, Any],
    ) -> RAGConfig:
        if config is not None:
            return config

        resolved_mode = cls._normalize_mode(mode)
        strategy_list = cls._normalize_strategies(strategies)
        config_kwargs = cls._build_config_kwargs(
            llm_model=llm_model,
            contextual_retrieval=contextual_retrieval,
            extra_kwargs=extra_kwargs,
        )

        if resolved_mode == RAGMode.AGENTIC:
            return RAGConfig.for_agentic(knowledge_dir=knowledge_dir, **config_kwargs)
        if resolved_mode == RAGMode.INDEXED:
            return RAGConfig.for_indexed(
                knowledge_dir=knowledge_dir,
                strategies=strategy_list,
                **config_kwargs,
            )

        config_values: dict[str, Any] = {"mode": resolved_mode, **config_kwargs}
        if knowledge_dir is not None:
            config_values["knowledge_dir"] = knowledge_dir
        return RAGConfig(**config_values)

    def _create_llm_adapter(self, provider: str, model: str | None) -> LLMAdapter | None:
        """Create LLM adapter from provider name.

        Args:
            provider: Provider name ("openai", "anthropic", "vertex")
            model: Optional model name

        Returns:
            LLMAdapter instance or None if creation fails
        """
        from houyi.adapters.llm.models import (
            CLAUDE_35_HAIKU,
            GPT_4O_MINI,
            PROVIDER_ANTHROPIC,
            PROVIDER_OPENAI,
            PROVIDER_VERTEX,
        )

        try:
            if provider == PROVIDER_OPENAI:
                from houyi.adapters.llm.openai_adapter import OpenAIAdapter

                return OpenAIAdapter(model=model or GPT_4O_MINI)
            elif provider == PROVIDER_ANTHROPIC:
                from houyi.adapters.llm.anthropic_adapter import AnthropicAdapter

                return AnthropicAdapter(model=model or CLAUDE_35_HAIKU)
            elif provider == PROVIDER_VERTEX:
                from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

                return GoogleVertexGeminiAdapter.from_env()
            else:
                logger.warning("Unknown LLM provider: %s", provider)
                return None
        except ImportError as e:
            logger.warning("Failed to import LLM adapter for %s: %s", provider, e)
            return None
        except Exception as e:
            logger.warning("Failed to create LLM adapter: %s", e)
            return None

    @property
    def config(self) -> RAGConfig:
        """Get current configuration."""
        return self._config

    @property
    def knowledge_dir(self) -> str:
        """Get knowledge directory path."""
        return self._config.knowledge_dir

    def _select_mode(self, query: str) -> RAGMode:
        """Select operating mode based on query and knowledge base.

        Auto mode selection logic:
        - Small knowledge base (< 100 files) → Agentic
        - Large knowledge base → Indexed
        """
        logger.debug("_select_mode called: config.mode=%s", self._config.mode)
        if self._config.mode != RAGMode.AUTO:
            return self._config.mode

        # Check knowledge base size
        kb_path = self._config.knowledge_dir
        if not os.path.exists(kb_path):
            return RAGMode.AGENTIC

        file_count = sum(1 for _ in _iter_files(kb_path))
        logger.debug("_select_mode: file_count=%d in %s", file_count, kb_path)

        # Simple heuristic: small KB uses Agentic, large uses Indexed
        if file_count < 100:
            return RAGMode.AGENTIC
        return RAGMode.INDEXED

    async def query(
        self,
        query: str,
        *,
        mode: RAGMode | None = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute a RAG query.

        Args:
            query: User query string
            mode: Override mode selection (optional)
            top_k: Number of results to retrieve
            **kwargs: Additional query options

        Returns:
            RetrievalResult with answer, sources, confidence, and quality
        """
        from houyi.rag.types import QualitySummary

        selected_mode = mode or self._select_mode(query)

        if selected_mode == RAGMode.AGENTIC:
            result = await self._query_agentic(query, top_k=top_k, **kwargs)
        else:
            result = await self._query_indexed(query, top_k=top_k, **kwargs)

        # Calculate quality summary from search results (v1.1)
        if result.search_results and result.quality is None:
            scores = [r.score for r in result.search_results]
            result.quality = QualitySummary.from_scores(scores)

        return result

    async def stream_query(
        self,
        query: str,
        *,
        mode: RAGMode | None = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Any:
        """Execute a streaming RAG query.

        Args:
            query: User query string
            mode: Override mode selection (optional)
            top_k: Number of results to retrieve
            **kwargs: Additional query options

        Yields:
            StreamEvent objects for SSE streaming
        """
        from houyi.rag.generation.streaming import build_non_streaming_events

        selected_mode = mode or self._select_mode(query)

        # Currently only Indexed mode supports streaming
        if selected_mode == RAGMode.AGENTIC:
            # Fallback to non-streaming for Agentic mode
            result = await self._query_agentic(query, top_k=top_k, **kwargs)
            for event in build_non_streaming_events(result):
                yield event
            return

        # Stream from Indexed mode
        async for event in self._stream_query_indexed(query, top_k=top_k, **kwargs):
            yield event

    async def _stream_query_indexed(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Any:
        """Stream query in Indexed mode."""
        await self._ensure_indexed_mode()

        # Get search results first
        result = await self._indexed_mode.search(query, top_k=top_k, **kwargs)

        # If no LLM adapter, yield non-streaming result
        if not self._llm_adapter:
            from houyi.rag.generation.streaming import build_non_streaming_events

            for event in build_non_streaming_events(result):
                yield event
            return

        # Use streaming generator
        from houyi.rag.generation.streaming import StreamingAnswerGenerator

        generator = StreamingAnswerGenerator(adapter=self._llm_adapter)
        async for event in generator.stream_generate(query, result.search_results):
            yield event

    async def index(
        self,
        paths: list[str] | str | None = None,
        *,
        build_graph: bool = False,
        contextual_retrieval: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build indexes for the knowledge base.

        This method indexes documents for Indexed mode retrieval.
        For Agentic mode, this is a no-op (returns empty stats).

        Args:
            paths: Document paths to index (default: all files in knowledge_dir)
            build_graph: Whether to extract entities and build graph
            contextual_retrieval: Enable Contextual Retrieval (uses config default if None)
            **kwargs: Additional index options

        Returns:
            Indexing statistics
        """
        if self._config.mode == RAGMode.AGENTIC:
            logger.info("index() is a no-op for Agentic mode")
            return {"documents": 0, "chunks": 0, "mode": "agentic"}

        if self._indexed_mode is None:
            await self._ensure_indexed_mode()

        # Default to all files in knowledge_dir
        if paths is None:
            paths = list(_iter_files(self._config.knowledge_dir))
        elif isinstance(paths, str):
            paths = [paths]

        # Handle empty paths case
        if not paths:
            return {"documents": 0, "chunks": 0}

        # Use contextual_retrieval from config if not explicitly specified
        if contextual_retrieval is None:
            contextual_retrieval = getattr(self._config, "contextual_retrieval", False)

        return await self._indexed_mode.ingest(
            paths,
            build_graph=build_graph,
            contextual_retrieval=contextual_retrieval,
            **kwargs,
        )

    async def add(
        self,
        paths: list[str] | str,
        *,
        build_graph: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add documents to existing index.

        Args:
            paths: Document paths to add
            build_graph: Whether to extract entities and build graph
            **kwargs: Additional options

        Returns:
            Indexing statistics for added documents
        """
        return await self.index(paths, build_graph=build_graph, **kwargs)

    async def refresh(self, **kwargs: Any) -> dict[str, Any]:
        """Rebuild indexes from scratch.

        Args:
            **kwargs: Additional options

        Returns:
            Indexing statistics
        """
        # Clear existing indexes
        self._indexed_mode = None

        # Rebuild
        return await self.index(**kwargs)

    async def _query_agentic(self, query: str, **kwargs: Any) -> RetrievalResult:
        """Execute query in Agentic mode."""
        await self._ensure_agentic_mode()
        return await self._agentic_mode.search(query, **kwargs)

    async def _query_indexed(self, query: str, **kwargs: Any) -> RetrievalResult:
        """Execute query in Indexed mode."""
        await self._ensure_indexed_mode()
        return await self._indexed_mode.search(query, **kwargs)

    def _create_agentic_mode(self) -> Any:
        """Construct the Agentic runtime from facade-owned config and collaborators."""
        from houyi.rag.agentic import AgenticMode

        return AgenticMode(
            config=self._config.agentic,
            knowledge_dir=self._config.knowledge_dir,
            agent=self._agent,
            llm_adapter=self._llm_adapter,
        )

    def _create_indexed_mode(self) -> Any:
        """Construct the Indexed runtime using the facade's storage and LLM wiring."""
        from houyi.rag.indexed import IndexedMode

        return IndexedMode(
            config=self._config.indexed,
            knowledge_dir=self._config.knowledge_dir,
            index_dir=self._config.get_index_dir(),
            embedding_config=self._config.embedding,
            graph_config=self._config.graph,
            llm_adapter=self._llm_adapter,
        )

    async def _ensure_agentic_mode(self) -> None:
        """Ensure the cached Agentic runtime exists before query execution."""
        if self._agentic_mode is None:
            self._agentic_mode = self._create_agentic_mode()

    async def _ensure_indexed_mode(self) -> None:
        """Ensure the cached Indexed runtime exists before retrieval or ingest."""
        if self._indexed_mode is None:
            self._indexed_mode = self._create_indexed_mode()


def _iter_files(directory: str) -> Iterator[str]:
    """Iterate over files in directory."""
    for root, _, files in os.walk(directory):
        for f in files:
            yield os.path.join(root, f)


async def search(
    query: str,
    knowledge_dir: str | None = None,
    mode: str = "auto",
    llm: str | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """One-liner RAG search function.

    This is the simplest API for quick, zero-config usage.

    Example:
        from houyi.rag import search

        # Simple usage
        result = await search("What is RAG?", knowledge_dir="./docs")
        print(result.answer)

        # With LLM for better answers
        result = await search(
            "What is RAG?",
            knowledge_dir="./docs",
            llm="openai:gpt-4o-mini",
        )

    Args:
        query: User query string
        knowledge_dir: Knowledge base directory
        mode: RAG mode ("agentic", "indexed", "auto")
        llm: LLM provider string (e.g., "openai", "openai:gpt-4o-mini")
        **kwargs: Additional options

    Returns:
        RetrievalResult with answer and sources
    """
    rag = RAG(
        knowledge_dir=knowledge_dir,
        mode=mode,
        llm=llm,
        **kwargs,
    )
    return await rag.query(query)
