"""RAGService - Main entry point for Houyi RAG."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from houyi.rag.config import RAGConfig
from houyi.rag.types import RAGMode, RetrievalResult, RetrievalStrategy

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter
    from houyi.runtime.agent import Agent

logger = logging.getLogger(__name__)


class RAGService:
    """Houyi RAG service supporting Agentic and Indexed modes.

    Example:
        # Level 2: Configuration-based usage
        rag = RAGService(mode="agentic", knowledge_dir="./docs")
        result = await rag.query("What is RAG?")

        # Level 3: With LLM adapter for enhanced capabilities
        from houyi.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(model="gpt-4o-mini")
        rag = RAGService(mode="indexed", llm_adapter=adapter)

        # Level 4: Expert mode with custom config
        config = RAGConfig(mode=RAGMode.INDEXED, ...)
        rag = RAGService(config=config)
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        mode: str | RAGMode = RAGMode.AUTO,
        knowledge_dir: str = "knowledge/",
        retrieval: list[str] | None = None,
        agent: Agent | None = None,
        llm_adapter: LLMAdapter | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RAG service.

        Args:
            config: Full RAG configuration (overrides other params)
            mode: RAG mode ("agentic", "indexed", "auto")
            knowledge_dir: Knowledge base directory path
            retrieval: List of retrieval strategies for indexed mode
            agent: Optional Agent instance for Agentic mode (reuses tools)
            llm_adapter: Optional LLM adapter for reranking and generation
            llm_provider: LLM provider name ("openai", "anthropic") for auto-creation
            llm_model: LLM model name for auto-creation
            **kwargs: Additional config options
        """
        # Extract llm_model for config
        config_kwargs = {**kwargs}
        if llm_model:
            config_kwargs["llm_model"] = llm_model

        if config is not None:
            self._config = config
        else:
            # Build config from params
            if isinstance(mode, str):
                mode = RAGMode(mode)

            strategies = None
            if retrieval:
                strategies = [RetrievalStrategy(s) for s in retrieval]

            if mode == RAGMode.AGENTIC:
                self._config = RAGConfig.for_agentic(knowledge_dir, **config_kwargs)
            elif mode == RAGMode.INDEXED:
                self._config = RAGConfig.for_indexed(
                    knowledge_dir, strategies=strategies, **config_kwargs
                )
            else:
                self._config = RAGConfig(
                    mode=mode, knowledge_dir=knowledge_dir, **config_kwargs
                )

        self._agent = agent
        self._llm_adapter = llm_adapter
        self._agentic_mode: Any = None
        self._indexed_mode: Any = None

        # Auto-create LLM adapter if provider specified
        if llm_adapter is None and llm_provider:
            self._llm_adapter = self._create_llm_adapter(llm_provider, llm_model)

    def _create_llm_adapter(
        self, provider: str, model: str | None
    ) -> LLMAdapter | None:
        """Create LLM adapter from provider name.

        Args:
            provider: Provider name ("openai", "anthropic", "vertex")
            model: Optional model name

        Returns:
            LLMAdapter instance or None if creation fails
        """
        try:
            if provider == "openai":
                from houyi.llm.openai_adapter import OpenAIAdapter

                return OpenAIAdapter(model=model or "gpt-4o-mini")
            elif provider == "anthropic":
                from houyi.llm.anthropic_adapter import AnthropicAdapter

                return AnthropicAdapter(model=model or "claude-3-haiku-20240307")
            elif provider == "vertex":
                from houyi.llm.vertex_gemini_adapter import VertexGeminiAdapter

                return VertexGeminiAdapter(model=model or "gemini-1.5-flash")
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

    def _select_mode(self, query: str) -> RAGMode:
        """Select operating mode based on query and knowledge base.

        Auto mode selection logic:
        - Small knowledge base (< 100 files) → Agentic
        - Large knowledge base → Indexed
        - Query complexity can also influence selection
        """
        if self._config.mode != RAGMode.AUTO:
            return self._config.mode

        # Check knowledge base size
        kb_path = self._config.knowledge_dir
        if not os.path.exists(kb_path):
            return RAGMode.AGENTIC

        file_count = sum(
            1 for _ in _iter_files(kb_path)
        )

        # Simple heuristic: small KB uses Agentic, large uses Indexed
        if file_count < 100:
            return RAGMode.AGENTIC
        return RAGMode.INDEXED

    async def query(
        self,
        query: str,
        *,
        mode: RAGMode | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute a RAG query.

        Args:
            query: User query string
            mode: Override mode selection (optional)
            **kwargs: Additional query options

        Returns:
            RetrievalResult with answer, sources, and confidence
        """
        selected_mode = mode or self._select_mode(query)

        if selected_mode == RAGMode.AGENTIC:
            return await self._query_agentic(query, **kwargs)
        return await self._query_indexed(query, **kwargs)

    async def _query_agentic(self, query: str, **kwargs: Any) -> RetrievalResult:
        """Execute query in Agentic mode."""
        if self._agentic_mode is None:
            from houyi.rag.agentic import AgenticMode

            self._agentic_mode = AgenticMode(
                config=self._config.agentic,
                knowledge_dir=self._config.knowledge_dir,
                agent=self._agent,
                llm_adapter=self._llm_adapter,
            )

        return await self._agentic_mode.search(query, **kwargs)

    async def _query_indexed(self, query: str, **kwargs: Any) -> RetrievalResult:
        """Execute query in Indexed mode."""
        if self._indexed_mode is None:
            from houyi.rag.indexed import IndexedMode

            self._indexed_mode = IndexedMode(
                config=self._config.indexed,
                knowledge_dir=self._config.knowledge_dir,
                embedding_config=self._config.embedding,
                graph_config=self._config.graph,
                llm_adapter=self._llm_adapter,
            )

        return await self._indexed_mode.search(query, **kwargs)

    async def ingest(
        self,
        paths: list[str] | str,
        *,
        build_graph: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ingest documents into the indexed knowledge base.

        Only applicable for Indexed mode.

        Args:
            paths: Document paths to ingest
            build_graph: Whether to extract entities and build graph
            **kwargs: Additional ingest options

        Returns:
            Ingestion statistics
        """
        if self._indexed_mode is None:
            from houyi.rag.indexed import IndexedMode

            self._indexed_mode = IndexedMode(
                config=self._config.indexed,
                knowledge_dir=self._config.knowledge_dir,
                embedding_config=self._config.embedding,
                graph_config=self._config.graph,
                llm_adapter=self._llm_adapter,
            )

        if isinstance(paths, str):
            paths = [paths]

        return await self._indexed_mode.ingest(paths, build_graph=build_graph, **kwargs)


def _iter_files(directory: str) -> Iterator[str]:
    """Iterate over files in directory."""
    for root, _, files in os.walk(directory):
        for f in files:
            yield os.path.join(root, f)


async def search(
    query: str,
    knowledge_dir: str = "knowledge/",
    mode: str = "auto",
    llm_provider: str | None = None,
    llm_model: str | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """One-liner RAG search function.

    This is the Level 1 API for quick, zero-config usage.

    Example:
        from houyi.rag import search

        # Simple usage
        result = await search("What is RAG?", knowledge_dir="./docs")
        print(result.answer)

        # With LLM for better answers
        result = await search(
            "What is RAG?",
            knowledge_dir="./docs",
            llm_provider="openai",
        )

    Args:
        query: User query string
        knowledge_dir: Knowledge base directory
        mode: RAG mode ("agentic", "indexed", "auto")
        llm_provider: LLM provider for enhanced capabilities ("openai", "anthropic")
        llm_model: LLM model name
        **kwargs: Additional options

    Returns:
        RetrievalResult with answer and sources
    """
    service = RAGService(
        mode=mode,
        knowledge_dir=knowledge_dir,
        llm_provider=llm_provider,
        llm_model=llm_model,
        **kwargs,
    )
    return await service.query(query)
