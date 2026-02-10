"""Corrective RAG (CRAG) implementation.

CRAG validates retrieval results and decides whether additional retrieval
is needed based on relevance assessment.

Reference:
- Corrective Retrieval Augmented Generation (CRAG) paper
- https://arxiv.org/abs/2401.15884
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.llm.base import BaseLLMAdapter

from houyi.rag.types import SearchResult

logger = logging.getLogger(__name__)


class RetrievalQuality(str, Enum):
    """Retrieval quality assessment."""

    CORRECT = "correct"  # Results are relevant
    INCORRECT = "incorrect"  # Results are not relevant
    AMBIGUOUS = "ambiguous"  # Results are partially relevant


@dataclass
class CRAGResult:
    """Result of CRAG validation."""

    quality: RetrievalQuality
    confidence: float
    relevant_results: list[SearchResult]
    needs_refinement: bool
    suggested_queries: list[str]
    reasoning: str = ""


class CRAGValidator:
    """Corrective RAG validator.

    Assesses retrieval quality and suggests corrective actions.
    """

    SYSTEM_PROMPT = """You are a retrieval quality assessor. Given a query and retrieved passages,
determine if the passages are relevant to answering the query.

Respond in JSON format:
{
    "quality": "correct" | "incorrect" | "ambiguous",
    "confidence": 0.0-1.0,
    "relevant_indices": [list of 0-indexed relevant passage indices],
    "reasoning": "brief explanation",
    "suggested_queries": ["alternative query 1", "alternative query 2"] (if quality != "correct")
}"""

    def __init__(
        self,
        adapter: BaseLLMAdapter | None = None,
        relevance_threshold: float = 0.5,
    ) -> None:
        """Initialize CRAG validator.

        Args:
            adapter: LLM adapter for quality assessment
            relevance_threshold: Minimum relevance score to consider correct
        """
        self._adapter = adapter
        self.relevance_threshold = relevance_threshold

    async def validate(
        self,
        query: str,
        results: list[SearchResult],
        **kwargs: Any,
    ) -> CRAGResult:
        """Validate retrieval quality.

        Args:
            query: Original query
            results: Retrieved results
            **kwargs: Additional arguments

        Returns:
            CRAGResult with quality assessment
        """
        if not results:
            return CRAGResult(
                quality=RetrievalQuality.INCORRECT,
                confidence=1.0,
                relevant_results=[],
                needs_refinement=True,
                suggested_queries=[query],
                reasoning="No results retrieved",
            )

        # Use LLM if available
        if self._adapter:
            return await self._validate_with_llm(query, results)

        # Fallback to heuristic validation
        return self._validate_heuristic(query, results)

    async def _validate_with_llm(
        self,
        query: str,
        results: list[SearchResult],
    ) -> CRAGResult:
        """Validate using LLM assessment.

        Args:
            query: Original query
            results: Retrieved results

        Returns:
            CRAGResult from LLM assessment
        """
        # Build prompt
        passages = []
        for i, result in enumerate(results[:10]):  # Limit to top 10
            content = result.content[:500] if result.content else ""
            passages.append(f"[{i}] {result.file_path}: {content}")

        user_prompt = f"""Query: {query}

Retrieved passages:
{chr(10).join(passages)}

Assess the relevance of these passages to the query."""

        try:
            response = await self._adapter.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            # Parse response
            import json

            data = json.loads(response.content)

            quality = RetrievalQuality(data.get("quality", "ambiguous"))
            confidence = float(data.get("confidence", 0.5))
            relevant_indices = data.get("relevant_indices", [])
            reasoning = data.get("reasoning", "")
            suggested_queries = data.get("suggested_queries", [])

            # Filter relevant results
            relevant_results = [results[i] for i in relevant_indices if i < len(results)]

            return CRAGResult(
                quality=quality,
                confidence=confidence,
                relevant_results=relevant_results,
                needs_refinement=quality != RetrievalQuality.CORRECT,
                suggested_queries=suggested_queries,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.warning("LLM validation failed: %s, using heuristic", e)
            return self._validate_heuristic(query, results)

    def _validate_heuristic(
        self,
        query: str,
        results: list[SearchResult],
    ) -> CRAGResult:
        """Validate using heuristic rules.

        Args:
            query: Original query
            results: Retrieved results

        Returns:
            CRAGResult from heuristic assessment
        """
        query_words = set(query.lower().split())
        relevant_results = []
        total_score = 0.0

        for result in results:
            # Check content overlap with query
            content_words = set((result.content or "").lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1)

            # Consider retrieval score
            score = result.score if result.score > 0 else overlap

            if score >= self.relevance_threshold or overlap >= 0.3:
                relevant_results.append(result)
                total_score += score

        # Determine quality
        if not relevant_results:
            quality = RetrievalQuality.INCORRECT
            confidence = 0.8
        elif len(relevant_results) >= len(results) * 0.5:
            quality = RetrievalQuality.CORRECT
            confidence = total_score / max(len(relevant_results), 1)
        else:
            quality = RetrievalQuality.AMBIGUOUS
            confidence = 0.5

        # Generate suggested queries if needed
        suggested_queries = []
        if quality != RetrievalQuality.CORRECT:
            # Simple query expansion
            suggested_queries = [
                f"{query} definition",
                f"{query} example",
            ]

        return CRAGResult(
            quality=quality,
            confidence=confidence,
            relevant_results=relevant_results,
            needs_refinement=quality != RetrievalQuality.CORRECT,
            suggested_queries=suggested_queries,
            reasoning=f"Heuristic: {len(relevant_results)}/{len(results)} results relevant",
        )

    async def refine_query(
        self,
        original_query: str,
        failed_results: list[SearchResult],
    ) -> list[str]:
        """Generate refined queries based on failed retrieval.

        Args:
            original_query: Original failed query
            failed_results: Results that were deemed irrelevant

        Returns:
            List of refined queries to try
        """
        if self._adapter:
            try:
                response = await self._adapter.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": 'Generate 2-3 alternative search queries for the given query. Return JSON: {"queries": [...]}',
                        },
                        {
                            "role": "user",
                            "content": f"Original query: {original_query}\nGenerate alternative queries that might find better results.",
                        },
                    ],
                    response_format={"type": "json_object"},
                )

                import json

                data = json.loads(response.content)
                return data.get("queries", [original_query])

            except Exception as e:
                logger.warning("Query refinement failed: %s", e)

        # Fallback: simple variations
        return [
            original_query,
            f"what is {original_query}",
            f"{original_query} explained",
        ]
