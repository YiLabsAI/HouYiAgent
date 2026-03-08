"""Generate answer from search results using LLM."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from houyi.rag.generation.streaming_helpers import (
    RAG_ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    estimate_stream_confidence,
)
from houyi.rag.types import SearchResult

if TYPE_CHECKING:
    from houyi.adapters.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Synthesize answer from search results using LLM.

    Follows RAG best practices:
    - Cite sources inline with [1], [2] markers
    - Handle conflicting information
    - Indicate confidence level
    - Support streaming for long answers
    """

    SYSTEM_PROMPT = RAG_ANSWER_SYSTEM_PROMPT

    _UNCERTAINTY_PHRASES = [
        "not enough information",
        "cannot find",
        "no relevant",
        "unable to determine",
        "insufficient information",
    ]

    def __init__(
        self,
        adapter: LLMAdapter,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> None:
        """Initialize answer generator.

        Args:
            adapter: LLM adapter for making API calls
            max_tokens: Maximum tokens for generated answer
            temperature: LLM temperature
        """
        self._adapter = adapter
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate(
        self,
        query: str,
        results: list[SearchResult],
        include_sources: bool = True,
    ) -> tuple[str, float]:
        """Generate answer from search results.

        Args:
            query: User's question
            results: Search results to use as context
            include_sources: Whether to include source citations

        Returns:
            Tuple of (answer_text, confidence_score)
        """
        from houyi.adapters.llm.base import LLMMessage, MessageRole

        if not results:
            return "No relevant information found in knowledge base.", 0.0

        prompt = build_answer_prompt(query, results)

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
            LLMMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await self._adapter.chat(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            answer = response.content
            confidence = estimate_stream_confidence(
                answer,
                results,
                uncertainty_phrases=self._UNCERTAINTY_PHRASES,
            )

            return answer, confidence
        except Exception as e:
            logger.error("Answer generation failed: %s", e)
            return self._fallback_answer(results), 0.3

    def _fallback_answer(self, results: list[SearchResult]) -> str:
        """Generate a simple fallback answer by concatenating results."""
        if not results:
            return "No relevant information found."

        parts = []
        for i, result in enumerate(results[:3], 1):
            if result.content:
                parts.append(f"[{i}] {result.content.strip()[:500]}")

        if not parts:
            return "No relevant information found."

        return "\n\n".join(parts)
