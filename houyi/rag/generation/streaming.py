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
    from houyi.adapters.llm.base import LLMAdapter

from houyi.rag.generation.streaming_helpers import (
    RAG_ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    build_stream_sources,
    estimate_stream_confidence,
)
from houyi.rag.types import RetrievalResult, SearchResult

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


def build_non_streaming_events(result: RetrievalResult) -> list[StreamEvent]:
    """Build static stream events from an already materialized retrieval result.

    This keeps the event payload contract in the generation layer even when the
    caller falls back from true token streaming to a one-shot answer.
    """
    return [
        StreamEvent(
            event_type=StreamEventType.START,
            data={"sources": [source.file_path for source in result.sources]},
        ),
        StreamEvent(
            event_type=StreamEventType.CHUNK,
            data=result.answer,
        ),
        StreamEvent(
            event_type=StreamEventType.END,
            data={"confidence": result.confidence},
        ),
    ]


class StreamingAnswerGenerator:
    """Generate streaming answers from search results.

    Yields chunks as they are generated for real-time display.
    """

    SYSTEM_PROMPT = RAG_ANSWER_SYSTEM_PROMPT

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
        from houyi.adapters.llm.base import LLMMessage, MessageRole

        # Emit start event with sources
        yield StreamEvent(
            event_type=StreamEventType.START,
            data={"sources": build_stream_sources(results), "query": query},
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

        prompt = build_answer_prompt(query, results)

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
                content = chunk.content_delta
                full_response += content
                yield StreamEvent(
                    event_type=StreamEventType.CHUNK,
                    data=content,
                )

        except Exception as e:
            logger.error("Streaming generation failed: %s", e)
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e)},
            )

        # Estimate confidence
        confidence = estimate_stream_confidence(full_response, results)

        # Emit end event
        yield StreamEvent(
            event_type=StreamEventType.END,
            data={
                "confidence": confidence,
                "total_length": len(full_response),
            },
        )


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
