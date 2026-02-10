"""Streaming support for RAG responses.

Provides async generators and SSE formatting for streaming answers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter

from houyi.rag.types import SearchResult

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Types of streaming events."""

    START = "start"  # Stream started
    CHUNK = "chunk"  # Content chunk
    SOURCE = "source"  # Source citation
    METADATA = "metadata"  # Metadata update
    ERROR = "error"  # Error occurred
    END = "end"  # Stream ended


@dataclass
class StreamEvent:
    """A single streaming event."""

    event_type: StreamEventType
    data: str | dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        if isinstance(self.data, dict):
            data_str = json.dumps(self.data, ensure_ascii=False)
        else:
            data_str = self.data

        return f"event: {self.event_type.value}\ndata: {data_str}\n\n"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event": self.event_type.value,
            "data": self.data,
            "metadata": self.metadata,
        }


class StreamingAnswerGenerator:
    """Generate streaming answers from search results.

    Yields chunks as they are generated for real-time display.
    """

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.

Guidelines:
- Answer ONLY based on the provided context
- Cite sources using [1], [2], etc. markers
- If the context doesn't contain enough information, say so clearly
- Be concise but complete
- If sources conflict, mention the discrepancy
- Use the same language as the user's question"""

    def __init__(
        self,
        adapter: LLMAdapter,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> None:
        """Initialize streaming answer generator.

        Args:
            adapter: LLM adapter with stream_chat support
            max_tokens: Maximum tokens for generated answer
            temperature: LLM temperature
        """
        self._adapter = adapter
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def stream_generate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AsyncIterator[StreamEvent]:
        """Generate streaming answer from search results.

        Args:
            query: User's question
            results: Search results to use as context

        Yields:
            StreamEvent objects
        """
        from houyi.llm.base import LLMMessage, MessageRole

        # Emit start event with sources
        sources = []
        for i, result in enumerate(results[:10], 1):
            source_info = {
                "index": i,
                "file_path": result.source.file_path if result.source else "",
                "snippet": result.content[:200] if result.content else "",
            }
            sources.append(source_info)

        yield StreamEvent(
            event_type=StreamEventType.START,
            data={"sources": sources, "query": query},
        )

        if not results:
            yield StreamEvent(
                event_type=StreamEventType.CHUNK,
                data="No relevant information found in knowledge base.",
            )
            yield StreamEvent(
                event_type=StreamEventType.END,
                data={"confidence": 0.0},
            )
            return

        # Format context
        context = self._format_context(results)

        prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the context above."""

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
            LLMMessage(role=MessageRole.USER, content=prompt),
        ]

        # Stream the response
        full_response = ""
        try:
            async for chunk in self._adapter.stream_chat(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ):
                full_response += chunk
                yield StreamEvent(
                    event_type=StreamEventType.CHUNK,
                    data=chunk,
                )

        except Exception as e:
            logger.error("Streaming generation failed: %s", e)
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e)},
            )

        # Estimate confidence
        confidence = self._estimate_confidence(full_response, results)

        # Emit end event
        yield StreamEvent(
            event_type=StreamEventType.END,
            data={
                "confidence": confidence,
                "total_length": len(full_response),
            },
        )

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results as numbered context blocks."""
        blocks = []

        for i, result in enumerate(results[:10], 1):
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
        confidence = 0.5

        # Check citations
        citation_count = sum(1 for i in range(1, 11) if f"[{i}]" in answer)
        if citation_count > 0:
            confidence += 0.1 * min(citation_count, 3)

        # Check result scores
        high_score_count = sum(1 for r in results if r.score > 0.7)
        if high_score_count > 0:
            confidence += 0.1 * min(high_score_count, 3)

        # Penalize uncertainty
        uncertainty_phrases = [
            "not enough information",
            "cannot find",
            "no relevant",
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))


async def stream_sse(
    events: AsyncIterator[StreamEvent],
) -> AsyncIterator[str]:
    """Convert stream events to SSE format.

    Args:
        events: Stream of events

    Yields:
        SSE-formatted strings
    """
    async for event in events:
        yield event.to_sse()
