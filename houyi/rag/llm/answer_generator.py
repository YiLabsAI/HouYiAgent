"""Generate answer from search results using LLM."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from houyi.rag.types import SearchResult

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Synthesize answer from search results using LLM.

    Follows RAG best practices:
    - Cite sources inline with [1], [2] markers
    - Handle conflicting information
    - Indicate confidence level
    - Support streaming for long answers
    """

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.

Guidelines:
- Answer ONLY based on the provided context
- Cite sources using [1], [2], etc. markers
- If the context doesn't contain enough information, say so clearly
- Be concise but complete
- If sources conflict, mention the discrepancy
- Use the same language as the user's question

Format your answer as:
<answer text with [1][2] citations>

Sources:
[1] <source description>
[2] <source description>
..."""

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
        from houyi.llm.base import LLMMessage, MessageRole

        if not results:
            return "未在知识库中找到相关信息。", 0.0

        context = self._format_context(results)

        prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the context above."""

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
            confidence = self._estimate_confidence(answer, results)

            return answer, confidence
        except Exception as e:
            logger.error("Answer generation failed: %s", e)
            return self._fallback_answer(results), 0.3

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results as numbered context blocks."""
        blocks = []

        for i, result in enumerate(results[:10], 1):  # Limit to top 10
            source_info = ""
            if result.source:
                source_info = f" (from: {result.source.file_path})"

            block = f"[{i}]{source_info}:\n{result.content.strip()}"
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)

    def _estimate_confidence(
        self,
        answer: str,
        results: list[SearchResult],
    ) -> float:
        """Estimate confidence based on answer quality indicators."""
        confidence = 0.5  # Base confidence

        # Check if answer contains citations
        citation_count = sum(1 for i in range(1, 11) if f"[{i}]" in answer)
        if citation_count > 0:
            confidence += 0.1 * min(citation_count, 3)

        # Check result quality
        high_score_count = sum(1 for r in results if r.score > 0.7)
        if high_score_count > 0:
            confidence += 0.1 * min(high_score_count, 3)

        # Penalize if answer indicates uncertainty
        uncertainty_phrases = [
            "not enough information",
            "cannot find",
            "no relevant",
            "未找到",
            "无法确定",
            "信息不足",
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _fallback_answer(self, results: list[SearchResult]) -> str:
        """Generate a simple fallback answer by concatenating results."""
        if not results:
            return "未找到相关信息。"

        parts = []
        for i, result in enumerate(results[:3], 1):
            if result.content:
                parts.append(f"[{i}] {result.content.strip()[:500]}")

        if not parts:
            return "未找到相关信息。"

        return "\n\n".join(parts)
