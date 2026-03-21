from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, cast

from houyi.adapters.llm import LLMAdapter, LLMMessage
from houyi.infrastructure.observability import Span, SpanType

from .sse_adapter import SSEEvent, stream_chat_sse

logger = logging.getLogger(__name__)

_TOOL_MARKER_RE = re.compile(
    r"\[tool call\]|\[tool_call\]|<tool_call\b[^>]*>[\s\S]*?</tool_call>|<tool_call\b[^>]*>|</tool_call>|<arg_[^>]+>[\s\S]*?</arg_[^>]+>|<arg_[^>]+>|</arg_[^>]+>|</?think>|<\|tool_calls_section_begin\|>|<\|tool_calls_section_end\|>|<\|tool_call_begin\|>|<\|tool_call_end\|>|<\|tool_call_argument_begin\|>|<\|tool_call_argument_end\|>|<\|tool_[^|]+\|>|(?:^|\n)\s*tool\s*:\s*[a-zA-Z_][\w.-]*\s*&args\s*:\s*[^\n]*",
    re.IGNORECASE,
)
_PLAIN_TEXT_TOOL_CARRIER_RE = re.compile(
    r"\btool\s*:\s*[a-zA-Z_][\w.-]*\s*&args\s*:",
    re.IGNORECASE,
)


def _sanitize_final_stream_text(raw: str | None) -> str:
    text = str(raw or "")
    if not text:
        return ""
    if _PLAIN_TEXT_TOOL_CARRIER_RE.search(text):
        return ""
    if not _TOOL_MARKER_RE.search(text):
        return text
    return (
        _TOOL_MARKER_RE.sub(" ", text)
        .replace("\r", "")
        .replace("\t", " ")
        .replace("  ", " ")
        .replace("\n\n\n", "\n\n")
        .strip()
    )


def _chunk_replay_text(text: str, *, target_size: int = 48) -> list[str]:
    normalized = str(text or "")
    if not normalized:
        return []
    if len(normalized) <= target_size:
        return [normalized]

    chunks: list[str] = []
    cursor = 0
    length = len(normalized)
    while cursor < length:
        end = min(length, cursor + target_size)
        if end < length:
            split_at = max(
                normalized.rfind("\n", cursor + 1, end + 1),
                normalized.rfind(" ", cursor + 1, end + 1),
                normalized.rfind("，", cursor + 1, end + 1),
                normalized.rfind("。", cursor + 1, end + 1),
                normalized.rfind("；", cursor + 1, end + 1),
                normalized.rfind("、", cursor + 1, end + 1),
                normalized.rfind(",", cursor + 1, end + 1),
                normalized.rfind(".", cursor + 1, end + 1),
            )
            if split_at > cursor:
                end = split_at + 1
        chunk = normalized[cursor:end]
        if chunk:
            chunks.append(chunk)
        cursor = end
    return chunks


@dataclass(slots=True)
class ReplayStreamCapture:
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FinalStreamCapture:
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    usage_payload: dict[str, Any] | None = None
    finish_reason: str | None = None
    generation_metadata: dict[str, Any] = field(default_factory=dict)


class AssistantResponseStreamer:
    """Streams replay or final LLM output and captures response artifacts."""

    def __init__(
        self,
        *,
        build_stream_error_content: Any,
        build_public_stream_error_message: Any,
        build_empty_stream_content: Any,
        extract_finish_reason: Any,
        finalize_stream_result: Any,
        json_safe: Any,
        normalize_chat_error: Any,
        normalize_usage_payload: Any,
        stage_span: Any,
    ) -> None:
        self._build_stream_error_content = build_stream_error_content
        self._build_public_stream_error_message = build_public_stream_error_message
        self._build_empty_stream_content = build_empty_stream_content
        self._extract_finish_reason = extract_finish_reason
        self._finalize_stream_result = finalize_stream_result
        self._json_safe = json_safe
        self._normalize_chat_error = normalize_chat_error
        self._normalize_usage_payload = normalize_usage_payload
        self._stage_span = stage_span

    async def stream_replay_chunks(
        self,
        *,
        replay_response: Any,
        assistant_message_id: str,
        model: str,
        context_usage: dict[str, Any],
        finish_reason: str | None,
    ) -> tuple[list[str], list[str], list[str]]:
        capture = ReplayStreamCapture()
        sse_chunks = [
            chunk
            async for chunk in self.iter_replay_chunks(
                replay_response=replay_response,
                assistant_message_id=assistant_message_id,
                model=model,
                context_usage=context_usage,
                finish_reason=finish_reason,
                capture=capture,
            )
        ]
        return sse_chunks, capture.content_parts, capture.reasoning_parts

    async def iter_replay_chunks(
        self,
        *,
        replay_response: Any,
        assistant_message_id: str,
        model: str,
        context_usage: dict[str, Any],
        finish_reason: str | None,
        capture: ReplayStreamCapture,
        stream_reasoning: bool = True,
    ) -> AsyncIterator[str]:
        replay_content = _sanitize_final_stream_text(
            str(getattr(replay_response, "content", "") or "")
        )
        replay_reasoning: str | None = None
        metadata = getattr(replay_response, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("reasoning_content") is not None:
            replay_reasoning = _sanitize_final_stream_text(
                str(metadata.get("reasoning_content") or "")
            )

        if replay_content:
            capture.content_parts.append(replay_content)
        if replay_reasoning:
            capture.reasoning_parts.append(replay_reasoning)
        logger.info(
            "Chat replay stream: message=%s model=%s content_len=%s reasoning_len=%s finish_reason=%s",
            assistant_message_id,
            model,
            len(replay_content),
            len(replay_reasoning or ""),
            finish_reason,
        )

        async def replay_stream() -> AsyncIterator[tuple[str, str | None]]:
            content_chunks = _chunk_replay_text(replay_content)
            reasoning_chunks = (
                _chunk_replay_text(replay_reasoning or "") if stream_reasoning else []
            )
            if not content_chunks and not reasoning_chunks:
                yield "", None
                return
            for content_chunk, reasoning_chunk in zip_longest(
                content_chunks, reasoning_chunks, fillvalue=""
            ):
                yield content_chunk or "", (reasoning_chunk or None)

        async for chunk in stream_chat_sse(
            llm_stream=replay_stream(),
            message_id=assistant_message_id,
            model=model,
            context_usage=context_usage,
            finish_reason=finish_reason,
        ):
            yield chunk

    async def stream_final_response(
        self,
        *,
        llm_adapter: LLMAdapter,
        llm_messages: list[dict[str, Any]],
        llm_kwargs: dict[str, Any],
        model: str,
        conversation_id: str,
        assistant_message_id: str,
        context_usage: dict[str, Any],
        chat_span: Span,
        require_visible_content: bool = False,
    ) -> tuple[
        list[str],
        list[str],
        list[str],
        dict[str, Any] | None,
        str | None,
        dict[str, Any],
    ]:
        capture = FinalStreamCapture()
        sse_chunks = [
            chunk
            async for chunk in self.iter_final_response(
                llm_adapter=llm_adapter,
                llm_messages=llm_messages,
                llm_kwargs=llm_kwargs,
                model=model,
                conversation_id=conversation_id,
                assistant_message_id=assistant_message_id,
                context_usage=context_usage,
                chat_span=chat_span,
                capture=capture,
                require_visible_content=require_visible_content,
            )
        ]
        return (
            sse_chunks,
            capture.content_parts,
            capture.reasoning_parts,
            capture.usage_payload,
            capture.finish_reason,
            capture.generation_metadata,
        )

    async def iter_final_response(
        self,
        *,
        llm_adapter: LLMAdapter,
        llm_messages: list[dict[str, Any]],
        llm_kwargs: dict[str, Any],
        model: str,
        conversation_id: str,
        assistant_message_id: str,
        context_usage: dict[str, Any],
        chat_span: Span,
        capture: FinalStreamCapture,
        require_visible_content: bool = False,
    ) -> AsyncIterator[str]:
        llm_input_chars = sum(len(str(message.get("content") or "")) for message in llm_messages)

        with self._stage_span(chat_span, "chat.stream.llm"):
            final_stream_kwargs = dict(llm_kwargs)
            final_stream_kwargs["parallel_tool_calls"] = False
            final_stream_kwargs["include_stream_usage"] = False
            llm_span = Span(
                name="llm.call",
                parent=chat_span,
                span_type=SpanType.LLM,
                model=model,
                attributes={
                    "llm.model": model,
                    "llm.message_count": len(llm_messages),
                    "llm.input_chars": llm_input_chars,
                },
            )
            stream_started_at = time.perf_counter()
            first_token_ms: float | None = None
            llm_chunk_count = 0
            stream_error_content: str | None = None
            stream_error_info: Any | None = None
            llm_stream = llm_adapter.stream_chat(
                messages=cast(list[LLMMessage | dict[str, Any]], llm_messages),
                model=model,
                tools=None,
                **final_stream_kwargs,
            )

            async def accumulating_stream() -> AsyncIterator[tuple[str, str | None]]:
                nonlocal first_token_ms
                async for chunk in llm_stream:
                    content_delta = _sanitize_final_stream_text(chunk.content_delta)
                    reasoning_delta = _sanitize_final_stream_text(chunk.reasoning_delta)
                    if first_token_ms is None and (content_delta or reasoning_delta):
                        first_token_ms = (time.perf_counter() - stream_started_at) * 1000
                        llm_span.set_attribute("chat.first_token_ms", round(first_token_ms, 2))
                    if content_delta:
                        capture.content_parts.append(content_delta)
                    if reasoning_delta:
                        capture.reasoning_parts.append(reasoning_delta)
                    yield content_delta, reasoning_delta

            def capture_stream_error(exc: Exception) -> str:
                nonlocal stream_error_content, stream_error_info
                stream_error_content = self._build_stream_error_content(exc)
                stream_error_info = self._normalize_chat_error(exc)
                logger.warning(
                    "Chat final stream error: conversation=%s message=%s model=%s error=%s",
                    conversation_id,
                    assistant_message_id,
                    model,
                    exc,
                )
                if stream_error_content:
                    capture.content_parts.append(stream_error_content)
                return stream_error_content or ""

            async for sse_chunk in stream_chat_sse(
                llm_stream=accumulating_stream(),
                message_id=assistant_message_id,
                model=model,
                context_usage=context_usage,
                usage=lambda: self._normalize_usage_payload(
                    self._json_safe(getattr(llm_adapter, "last_usage", None))
                ),
                finish_reason=lambda: self._extract_finish_reason(
                    getattr(llm_adapter, "last_finish_reason", None)
                ),
                error_message_builder=capture_stream_error,
                public_error_builder=self._build_public_stream_error_message,
            ):
                llm_chunk_count += 1
                yield sse_chunk

            generation_time_ms = (time.perf_counter() - stream_started_at) * 1000
            (
                capture.usage_payload,
                capture.finish_reason,
                capture.generation_metadata,
            ) = self._finalize_stream_result(
                llm_adapter=llm_adapter,
                llm_span=llm_span,
                first_token_ms=first_token_ms,
                generation_time_ms=generation_time_ms,
                chunk_count=llm_chunk_count,
            )
            logger.info(
                "Chat final stream complete: conversation=%s message=%s model=%s chunks=%s content_parts=%s reasoning_parts=%s first_token_ms=%s finish_reason=%s errored=%s",
                conversation_id,
                assistant_message_id,
                model,
                llm_chunk_count,
                len(capture.content_parts),
                len(capture.reasoning_parts),
                round(first_token_ms, 2) if isinstance(first_token_ms, (int, float)) else None,
                capture.finish_reason,
                bool(stream_error_content),
            )
            capture.generation_metadata["final_stream_message_count"] = len(llm_messages)
            capture.generation_metadata["final_stream_input_chars"] = llm_input_chars
            capture.generation_metadata["final_stream_chunk_count"] = llm_chunk_count
            if stream_error_content and not capture.finish_reason:
                capture.finish_reason = "error"
            if (
                not capture.content_parts
                and capture.reasoning_parts
                and not require_visible_content
            ):
                capture.generation_metadata["final_stream_status"] = "reasoning_only"
                capture.generation_metadata["final_stream_empty_visible_output"] = False
            if not capture.content_parts and (
                not capture.reasoning_parts or require_visible_content
            ):
                empty_stream_content = self._build_empty_stream_content()
                capture.content_parts.append(empty_stream_content)
                capture.finish_reason = capture.finish_reason or "error"
                capture.generation_metadata["final_stream_status"] = "empty_visible_output"
                capture.generation_metadata["final_stream_empty_visible_output"] = True
                logger.warning(
                    "Chat final stream returned no visible deltas: conversation=%s, message=%s, model=%s, finish_reason=%s, chunk_count=%s, message_count=%s, usage=%s",
                    conversation_id,
                    assistant_message_id,
                    model,
                    capture.finish_reason,
                    llm_chunk_count,
                    len(llm_messages),
                    capture.usage_payload,
                )
                yield SSEEvent(
                    event="message.delta",
                    data={
                        "message_id": assistant_message_id,
                        "seq": llm_chunk_count + 1,
                        "content": empty_stream_content,
                    },
                    event_id=f"{assistant_message_id}-{llm_chunk_count + 1}",
                ).encode()
            if stream_error_content:
                capture.generation_metadata["final_stream_status"] = "error"
                capture.generation_metadata["final_stream_empty_visible_output"] = False
                if stream_error_info is not None:
                    capture.generation_metadata["final_stream_error_code"] = getattr(
                        stream_error_info, "error_code", None
                    )
                    capture.generation_metadata["final_stream_error_category"] = getattr(
                        stream_error_info, "category", None
                    )
                    capture.generation_metadata["final_stream_error_retryable"] = getattr(
                        stream_error_info, "retryable", None
                    )
            else:
                capture.generation_metadata.setdefault("final_stream_status", "completed")
                capture.generation_metadata.setdefault("final_stream_empty_visible_output", False)
