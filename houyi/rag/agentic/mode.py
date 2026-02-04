"""Agentic mode implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from houyi.rag.agentic.navigator import DirectoryNavigator
from houyi.rag.agentic.searcher import AgenticSearcher
from houyi.rag.config import AgenticConfig
from houyi.rag.types import RAGMode, RetrievalResult, SearchResult

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter
    from houyi.runtime.agent import Agent

logger = logging.getLogger(__name__)


class AgenticMode:
    """Agentic RAG mode using LLM + Agent tools.

    This mode operates without pre-built indexes by:
    1. Reading hierarchical directory indexes (data_structure.md)
    2. Using grep/read_file tools to search and retrieve content
    3. Iteratively refining search over multiple rounds
    4. Using LLM for keyword extraction and answer generation (when available)

    Reference: https://github.com/ConardLi/rag-skill
    """

    def __init__(
        self,
        config: AgenticConfig,
        knowledge_dir: str,
        agent: Agent | None = None,
        llm_adapter: LLMAdapter | None = None,
    ) -> None:
        """Initialize Agentic mode.

        Args:
            config: Agentic mode configuration
            knowledge_dir: Knowledge base root directory
            agent: Optional Agent instance for tool execution
            llm_adapter: Optional LLM adapter for enhanced extraction and generation
        """
        self._config = config
        self._knowledge_dir = knowledge_dir
        self._llm_adapter = llm_adapter
        self._navigator = DirectoryNavigator(
            knowledge_dir=knowledge_dir,
            index_file=config.index_file,
        )
        self._searcher = AgenticSearcher(
            knowledge_dir=knowledge_dir,
            chunk_limit=config.chunk_limit,
            agent=agent,
        )

        # Initialize LLM components if adapter is provided
        self._keyword_extractor = None
        self._answer_generator = None
        if llm_adapter:
            from houyi.rag.llm import AnswerGenerator, KeywordExtractor

            self._keyword_extractor = KeywordExtractor(llm_adapter)
            self._answer_generator = AnswerGenerator(llm_adapter)

    async def search(
        self,
        query: str,
        max_rounds: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute agentic search.

        The search process:
        1. Navigate directory structure using index files
        2. Identify candidate files based on query relevance
        3. Search files using grep for keywords (LLM-extracted if available)
        4. Read relevant chunks around matches
        5. Iterate up to max_rounds times
        6. Generate answer using LLM (if available) or concatenate results

        Args:
            query: User query string
            max_rounds: Override max search iterations
            **kwargs: Additional options

        Returns:
            RetrievalResult with answer and sources
        """
        rounds = max_rounds or self._config.max_rounds

        # Step 1: Navigate to relevant directories
        candidate_paths = await self._navigator.find_candidates(query)

        if not candidate_paths:
            return RetrievalResult(
                answer="未在知识库中找到相关信息。",
                sources=[],
                confidence=0.0,
                mode_used=RAGMode.AGENTIC,
            )

        # Step 2: Extract initial keywords (use LLM if available)
        initial_keywords = await self._extract_keywords_async(query)

        # Step 3: Iterative search
        all_results: list[SearchResult] = []
        searched_files: set[str] = set()
        current_keywords = initial_keywords

        for round_num in range(rounds):
            # Search candidate files
            round_results = await self._searcher.search_files(
                paths=candidate_paths,
                keywords=current_keywords,
                exclude_files=searched_files,
            )

            # Track searched files
            for result in round_results:
                if result.source:
                    searched_files.add(result.source.file_path)

            all_results.extend(round_results)

            # Check if we have enough results
            if self._has_sufficient_results(all_results, query):
                break

            # Expand keywords for next round if LLM available
            if round_num < rounds - 1 and self._keyword_extractor:
                try:
                    current_keywords = await self._keyword_extractor.expand(
                        keywords=current_keywords,
                        context="",
                        previous_results=[r.content for r in all_results[:3]],
                    )
                except Exception as e:
                    logger.debug("Keyword expansion failed: %s", e)

        # Step 4: Build response
        sources = [
            r.source for r in all_results
            if r.source is not None
        ]

        # Generate answer using LLM if available, otherwise concatenate
        answer, confidence = await self._generate_answer(query, all_results)

        return RetrievalResult(
            answer=answer,
            sources=sources[:10],  # Limit to top 10 sources
            confidence=confidence,
            search_results=all_results,
            mode_used=RAGMode.AGENTIC,
        )

    async def _extract_keywords_async(self, query: str) -> list[str]:
        """Extract keywords using LLM if available, otherwise use simple extraction."""
        if self._keyword_extractor:
            try:
                result = await self._keyword_extractor.extract(query)
                keywords = result.get("keywords", [])
                if keywords:
                    return keywords
            except Exception as e:
                logger.debug("LLM keyword extraction failed: %s", e)

        # Fallback to simple extraction
        return self._extract_keywords_simple(query)

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

    def _extract_keywords_simple(self, query: str) -> list[str]:
        """Simple keyword extraction (fallback when LLM unavailable)."""
        stop_words = {"的", "是", "在", "了", "和", "与", "或", "a", "an", "the", "is", "are"}
        words = query.replace("?", "").replace("？", "").split()
        return [w for w in words if w.lower() not in stop_words and len(w) > 1]

    def _has_sufficient_results(
        self,
        results: list[Any],
        query: str,
    ) -> bool:
        """Check if we have enough results to answer the query."""
        # Simple heuristic: at least 3 relevant results
        high_score_results = [r for r in results if r.score > 0.5]
        return len(high_score_results) >= 3

    def _build_answer_simple(self, results: list[SearchResult]) -> str:
        """Build answer by concatenating results (fallback when LLM unavailable)."""
        contents = []
        for r in results[:5]:
            if r.content:
                contents.append(r.content.strip())

        if not contents:
            return "未找到相关信息。"

        return "\n\n---\n\n".join(contents)
