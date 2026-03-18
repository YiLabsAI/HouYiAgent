"""SSE Adapter: Server-Sent Events framing for chat streaming.

Converts LLM streaming output into SSE events following the design protocol:
- message.delta: incremental content/reasoning chunks
- message.finish: completion with usage stats
- message.error: error notification
- message.aborted: user-initiated abort
- context.usage: context window usage snapshot

SSE is the business interaction channel only; observability spans go through WS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from .chat_errors import build_transport_chat_error

logger = logging.getLogger(__name__)


class SSEEvent:
    """A single Server-Sent Event."""

    def __init__(self, event: str, data: dict[str, Any], event_id: str | None = None):
        self.event = event
        self.data = data
        self.event_id = event_id

    def encode(self) -> str:
        """Encode as SSE wire format."""
        lines = []
        if self.event_id:
            lines.append(f"id: {self.event_id}")
        lines.append(f"event: {self.event}")
        lines.append(f"data: {json.dumps(self.data, ensure_ascii=False)}")
        lines.append("")  # trailing newline
        return "\n".join(lines) + "\n"


async def stream_chat_sse(
    llm_stream: AsyncIterator[tuple[str, str | None]],
    message_id: str,
    model: str = "",
    context_usage: dict[str, Any] | None = None,
    usage: dict[str, Any] | Callable[[], dict[str, Any] | None] | None = None,
    finish_reason: str | Callable[[], str | None] | None = None,
    error_message_builder: Callable[[Exception], str | None] | None = None,
    public_error_builder: Callable[[Exception], str] | None = None,
) -> AsyncIterator[str]:
    """Convert LLM stream into SSE events.

    Args:
        llm_stream: AsyncIterator yielding (content_delta, reasoning_delta) tuples.
        message_id: ID of the assistant message being generated.
        model: Model name for metadata.
        context_usage: Optional context usage snapshot to send before streaming.
        finish_reason: Optional provider-reported finish reason or lazy resolver.

    Yields:
        SSE-encoded strings ready for HTTP response.
    """
    seq = 0

    # Send context usage snapshot first (if available)
    if context_usage:
        yield SSEEvent(
            event="context.usage",
            data={"message_id": message_id, "usage": context_usage},
        ).encode()

    total_content = ""
    total_reasoning = ""

    try:
        async for content_delta, reasoning_delta in llm_stream:
            seq += 1
            data: dict[str, Any] = {
                "message_id": message_id,
                "seq": seq,
            }

            if content_delta:
                data["content"] = content_delta
                total_content += content_delta

            if reasoning_delta:
                data["reasoning_content"] = reasoning_delta
                total_reasoning += reasoning_delta

            yield SSEEvent(
                event="message.delta",
                data=data,
                event_id=f"{message_id}-{seq}",
            ).encode()

        # Send finish event
        resolved_finish_reason = finish_reason() if callable(finish_reason) else finish_reason
        resolved_usage = usage() if callable(usage) else usage
        finish_data: dict[str, Any] = {
            "message_id": message_id,
            "model": model,
            "finish_reason": resolved_finish_reason or "stop",
            "total_chunks": seq,
            "content_length": len(total_content),
            "timestamp": time.time(),
        }
        if total_reasoning:
            finish_data["reasoning_length"] = len(total_reasoning)
        if isinstance(resolved_usage, dict) and resolved_usage:
            finish_data["usage"] = resolved_usage

        yield SSEEvent(event="message.finish", data=finish_data).encode()

    except asyncio.CancelledError:
        # User-initiated abort (AbortController)
        yield SSEEvent(
            event="message.aborted",
            data={
                "message_id": message_id,
                "reason": "user_abort",
                "chunks_sent": seq,
                "timestamp": time.time(),
            },
        ).encode()
        raise

    except Exception as e:
        logger.error("SSE stream error for message %s: %s", message_id, e, exc_info=True)
        fallback_error_message = error_message_builder(e) if error_message_builder else None
        transport_error = build_transport_chat_error(e)
        public_error_message = (
            public_error_builder(e)
            if public_error_builder
            else (fallback_error_message or transport_error["public_message"])
        )
        if fallback_error_message:
            seq += 1
            yield SSEEvent(
                event="message.delta",
                data={
                    "message_id": message_id,
                    "seq": seq,
                    "content": fallback_error_message,
                },
                event_id=f"{message_id}-{seq}",
            ).encode()
        yield SSEEvent(
            event="message.error",
            data={
                "message_id": message_id,
                "error": public_error_message,
                "error_code": transport_error["error_code"],
                "public_message": transport_error["public_message"],
                "retryable": transport_error["retryable"],
                "status_code": transport_error["status_code"],
                "provider_code": transport_error["provider_code"],
                "error_type": type(e).__name__,
                "chunks_sent": seq,
                "timestamp": time.time(),
            },
        ).encode()
