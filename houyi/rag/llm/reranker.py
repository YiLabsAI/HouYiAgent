"""Rerank search results using LLM."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from houyi.rag.types import SearchResult

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class LLMReranker:
    """Rerank search results using LLM for improved relevance.

    Implements listwise reranking:
    - Present all candidates to LLM
    - LLM returns ordered relevance scores
    - More accurate than vector similarity for complex queries

    Reference: RankGPT paper (https://arxiv.org/abs/2304.09542)
    """

    SYSTEM_PROMPT = """You are a relevance judge for information retrieval.
Given a query and candidate passages, rate each passage's relevance to the query.

Return ONLY valid JSON in this format:
{"scores": [score1, score2, ...]}

Scoring guidelines (0-10 scale):
- 10: Perfect match, directly answers the query
- 7-9: Highly relevant, contains key information
- 4-6: Somewhat relevant, tangentially related
- 1-3: Low relevance, minimal connection
- 0: Completely irrelevant

Consider:
- Semantic match to the query intent
- Completeness of information
- Specificity and directness"""

    def __init__(
        self,
        adapter: LLMAdapter,
        batch_size: int = 10,
        temperature: float = 0.1,
    ) -> None:
        """Initialize LLM reranker.

        Args:
            adapter: LLM adapter for making API calls
            batch_size: Number of passages to score per LLM call
            temperature: LLM temperature (lower = more consistent)
        """
        self._adapter = adapter
        self._batch_size = batch_size
        self._temperature = temperature

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Rerank results using LLM scoring.

        Args:
            query: User's search query
            results: Search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of search results
        """
        if len(results) <= 1:
            return results

        # Process in batches to avoid token limits
        all_scored: list[tuple[SearchResult, float]] = []

        for i in range(0, len(results), self._batch_size):
            batch = results[i : i + self._batch_size]
            try:
                scores = await self._score_batch(query, batch)
                for result, score in zip(batch, scores, strict=True):
                    all_scored.append((result, score))
            except Exception as e:
                logger.warning("Batch scoring failed: %s, using original scores", e)
                for result in batch:
                    all_scored.append((result, result.score))

        # Sort by LLM scores (descending)
        all_scored.sort(key=lambda x: x[1], reverse=True)

        # Update scores and return top_k
        reranked = []
        for result, new_score in all_scored[:top_k]:
            # Create new result with updated score
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=new_score / 10.0,  # Normalize to 0-1
                    source=result.source,
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "llm_score": new_score,
                    },
                )
            )

        return reranked

    async def _score_batch(
        self,
        query: str,
        batch: list[SearchResult],
    ) -> list[float]:
        """Score a batch of passages using LLM."""
        from houyi.llm.base import LLMMessage, MessageRole

        # Format passages for LLM
        passages_text = self._format_passages(batch)

        prompt = f"""Query: {query}

Passages to evaluate:
{passages_text}

Rate the relevance of each passage (0-10) to the query.
Return JSON: {{"scores": [score1, score2, ...]}}"""

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
            LLMMessage(role=MessageRole.USER, content=prompt),
        ]

        response = await self._adapter.chat(
            messages=messages,
            temperature=self._temperature,
            max_tokens=200,
        )

        scores = self._parse_scores(response.content, len(batch))
        return scores

    def _format_passages(self, results: list[SearchResult]) -> str:
        """Format passages for LLM evaluation."""
        parts = []
        for i, result in enumerate(results, 1):
            # Truncate long passages
            content = result.content[:500]
            if len(result.content) > 500:
                content += "..."
            parts.append(f"[Passage {i}]\n{content}")

        return "\n\n".join(parts)

    def _parse_scores(self, content: str, expected_count: int) -> list[float]:
        """Parse scores from LLM response."""
        try:
            # Clean up response
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    line for line in lines if not line.startswith("```")
                )

            data = json.loads(content)
            scores = data.get("scores", [])

            # Validate and normalize scores
            validated = []
            for s in scores[:expected_count]:
                try:
                    score = float(s)
                    validated.append(max(0.0, min(10.0, score)))
                except (TypeError, ValueError):
                    validated.append(5.0)  # Default middle score

            # Pad if needed
            while len(validated) < expected_count:
                validated.append(5.0)

            return validated

        except json.JSONDecodeError:
            logger.warning("Failed to parse reranker scores, using defaults")
            return [5.0] * expected_count
