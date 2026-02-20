"""Agentic mode implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from houyi.rag.agentic.navigator import DirectoryNavigator
from houyi.rag.agentic.searcher import AgenticSearcher
from houyi.rag.config import AgenticConfig
from houyi.rag.types import RAGMode, RetrievalResult, SearchResult

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter
    from houyi.runtime.agent import Agent

logger = logging.getLogger(__name__)


class SearchRoundType(str, Enum):
    """Types of search rounds in the 5-round strategy."""

    BROAD = "broad"  # Round 1: Initial broad search
    FOCUSED = "focused"  # Round 2: Narrow based on initial results
    SEMANTIC = "semantic"  # Round 3: Semantic expansion
    CROSS_REF = "cross_ref"  # Round 4: Cross-reference related docs
    VERIFY = "verify"  # Round 5: Verify top results


@dataclass
class RoundResult:
    """Result from a single search round."""

    round_type: SearchRoundType
    round_num: int
    keywords_used: list[str]
    results: list[SearchResult]
    files_searched: int
    metadata: dict[str, Any] = field(default_factory=dict)


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
        """Execute agentic search using 5-round strategy.

        The 5-round strategy:
        1. BROAD: Initial keyword search across all candidates
        2. FOCUSED: Narrow search in high-scoring files
        3. SEMANTIC: LLM-expanded keywords for semantic search
        4. CROSS_REF: Find related documents via entities
        5. VERIFY: Re-check top results for quality

        Args:
            query: User query string
            max_rounds: Override max search iterations (default: 5)
            **kwargs: Additional options

        Returns:
            RetrievalResult with answer and sources
        """
        rounds = max_rounds or self._config.max_rounds
        rounds = min(rounds, 5)  # Cap at 5 rounds

        # Step 1: Navigate to relevant directories
        candidate_paths = await self._navigator.find_candidates(query)

        if not candidate_paths:
            return RetrievalResult(
                answer="No relevant information found in knowledge base.",
                sources=[],
                confidence=0.0,
                mode_used=RAGMode.AGENTIC,
                metadata={"rounds_executed": 0},
            )

        # Initialize tracking
        all_results: list[SearchResult] = []
        searched_files: set[str] = set()
        round_history: list[RoundResult] = []
        initial_keywords = await self._extract_keywords_async(query)

        # Execute rounds based on strategy
        round_types = [
            SearchRoundType.BROAD,
            SearchRoundType.FOCUSED,
            SearchRoundType.SEMANTIC,
            SearchRoundType.CROSS_REF,
            SearchRoundType.VERIFY,
        ]

        for round_num, round_type in enumerate(round_types[:rounds]):
            round_result = await self._execute_round(
                round_type=round_type,
                round_num=round_num,
                query=query,
                candidate_paths=candidate_paths,
                initial_keywords=initial_keywords,
                all_results=all_results,
                searched_files=searched_files,
            )

            round_history.append(round_result)
            all_results.extend(round_result.results)

            # Track searched files
            for result in round_result.results:
                if result.source:
                    searched_files.add(result.source.file_path)

            # Check early termination
            if self._should_terminate(all_results, round_type):
                logger.debug(
                    "Early termination at round %d (%s): sufficient results",
                    round_num + 1,
                    round_type.value,
                )
                break

        # Deduplicate and rank results
        all_results = self._deduplicate_results(all_results)

        # Build response
        sources = [r.source for r in all_results if r.source is not None]

        # Generate answer
        answer, confidence = await self._generate_answer(query, all_results)

        # Build metadata
        metadata: dict[str, Any] = {
            "rounds_executed": len(round_history),
            "round_history": [
                {
                    "round": r.round_num + 1,
                    "type": r.round_type.value,
                    "keywords": r.keywords_used,
                    "results_found": len(r.results),
                    "files_searched": r.files_searched,
                }
                for r in round_history
            ],
            "total_files_searched": len(searched_files),
        }

        return RetrievalResult(
            answer=answer,
            sources=sources[:10],
            confidence=confidence,
            search_results=all_results,
            mode_used=RAGMode.AGENTIC,
            metadata=metadata,
        )

    async def _execute_round(
        self,
        round_type: SearchRoundType,
        round_num: int,
        query: str,
        candidate_paths: list[str],
        initial_keywords: list[str],
        all_results: list[SearchResult],
        searched_files: set[str],
    ) -> RoundResult:
        """Execute a single search round based on type."""
        keywords: list[str] = []
        results: list[SearchResult] = []
        paths_to_search = candidate_paths

        if round_type == SearchRoundType.BROAD:
            # Round 1: Broad search with initial keywords
            keywords = initial_keywords
            results = await self._searcher.search_files(
                paths=paths_to_search,
                keywords=keywords,
                exclude_files=searched_files,
            )

        elif round_type == SearchRoundType.FOCUSED:
            # Round 2: Focus on files that had hits in round 1
            top_files = self._get_top_files(all_results, limit=5)
            if top_files:
                paths_to_search = top_files
            keywords = initial_keywords
            results = await self._searcher.search_files(
                paths=paths_to_search,
                keywords=keywords,
                exclude_files=searched_files,
            )

        elif round_type == SearchRoundType.SEMANTIC:
            # Round 3: Semantic expansion using LLM
            keywords = await self._expand_keywords_semantic(
                initial_keywords,
                all_results[:5],
            )
            results = await self._searcher.search_files(
                paths=candidate_paths,
                keywords=keywords,
                exclude_files=searched_files,
            )

        elif round_type == SearchRoundType.CROSS_REF:
            # Round 4: Cross-reference via entities
            entities = self._extract_entities(all_results[:10])
            if entities:
                keywords = entities[:5]
                results = await self._searcher.search_files(
                    paths=candidate_paths,
                    keywords=keywords,
                    exclude_files=searched_files,
                )

        elif round_type == SearchRoundType.VERIFY:
            # Round 5: Re-verify top results with refined query
            refined_keywords = self._refine_keywords(query, all_results[:5])
            keywords = refined_keywords or initial_keywords
            top_files = self._get_top_files(all_results, limit=3)
            if top_files:
                results = await self._searcher.search_files(
                    paths=top_files,
                    keywords=keywords,
                    exclude_files=set(),  # Allow re-searching
                )

        return RoundResult(
            round_type=round_type,
            round_num=round_num,
            keywords_used=keywords,
            results=results,
            files_searched=len(paths_to_search),
        )

    def _should_terminate(
        self,
        results: list[SearchResult],
        current_round: SearchRoundType,
    ) -> bool:
        """Determine if search should terminate early."""
        # Don't terminate before round 2
        if current_round == SearchRoundType.BROAD:
            return False

        # Check for sufficient high-quality results
        high_score = [r for r in results if r.score > 0.7]
        if len(high_score) >= 5:
            return True

        return len(results) >= 10

    def _get_top_files(
        self,
        results: list[SearchResult],
        limit: int = 5,
    ) -> list[str]:
        """Get paths of top-scoring files."""
        file_scores: dict[str, float] = {}
        for r in results:
            if r.source and r.source.file_path:
                path = r.source.file_path
                file_scores[path] = max(file_scores.get(path, 0), r.score)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_files[:limit]]

    async def _expand_keywords_semantic(
        self,
        keywords: list[str],
        results: list[SearchResult],
    ) -> list[str]:
        """Expand keywords using semantic understanding."""
        if not self._keyword_extractor:
            return keywords

        try:
            context = "\n".join(r.content[:200] for r in results if r.content)
            expanded = await self._keyword_extractor.expand(
                keywords=keywords,
                context=context,
                previous_results=[r.content for r in results[:3]],
            )
            return expanded or keywords
        except Exception as e:
            logger.debug("Semantic expansion failed: %s", e)
            return keywords

    def _extract_entities(self, results: list[SearchResult]) -> list[str]:
        """Extract entities from results for cross-referencing."""
        import re

        entities = set()
        for r in results:
            if not r.content:
                continue

            # Extract capitalized words (potential entities)
            words = r.content.split()
            for word in words:
                clean = re.sub(r"[^\w]", "", word)
                if clean and clean[0].isupper() and len(clean) > 2:
                    entities.add(clean)

            # Extract quoted phrases
            quoted = re.findall(r'"([^"]+)"', r.content)
            entities.update(quoted)

        return list(entities)[:10]

    def _refine_keywords(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[str]:
        """Refine keywords based on query and results."""
        # Extract words that appear in both query and results
        query_words = set(query.lower().split())
        result_words: set[str] = set()

        for r in results:
            if r.content:
                result_words.update(r.content.lower().split())

        # Find intersection and filter
        common = query_words & result_words
        refined = [w for w in common if len(w) > 2]

        return refined[:5] if refined else []

    def _deduplicate_results(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Remove duplicate results and sort by score."""
        seen_content: set[str] = set()
        unique: list[SearchResult] = []

        for r in sorted(results, key=lambda x: x.score, reverse=True):
            # Use content hash for deduplication
            content_key = r.content[:100] if r.content else r.chunk_id or ""
            if content_key and content_key not in seen_content:
                seen_content.add(content_key)
                unique.append(r)

        return unique

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

    def _extract_keywords_simple(self, query: str) -> list[str]:
        """Simple keyword extraction (fallback when LLM unavailable)."""
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
        }
        words = query.replace("?", "").split()
        return [w for w in words if w.lower() not in stop_words and len(w) > 1]

    def _build_answer_simple(self, results: list[SearchResult]) -> str:
        """Build answer by concatenating results (fallback when LLM unavailable)."""
        contents = []
        for r in results[:5]:
            if r.content:
                contents.append(r.content.strip())

        if not contents:
            return "No relevant information found."

        return "\n\n---\n\n".join(contents)
