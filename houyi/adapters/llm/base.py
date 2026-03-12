"""Base classes for LLM adapters.

Defines the unified contract that all LLM provider adapters must implement:
- ``chat()`` — non-streaming completion with tool calling support
- ``stream_chat()`` — streaming completion yielding
  ``StreamChunk`` objects

Provider-specific adapters (OpenAI, Anthropic, SiliconFlow, Vertex AI, etc.)
inherit from ``LLMAdapter`` and implement the two abstract methods.

``StreamResponse`` wraps the stream and performs base-layer tool_calls
delta accumulation. Its ``__anext__`` returns ``StreamChunk`` objects.
"""

from __future__ import annotations

import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared defaults (single source of truth for all adapters)
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE: float = 0.7
"""Default sampling temperature used across all LLM adapters."""


@dataclass(slots=True)
class StreamChunk:
    """Structured streaming delta emitted by ``stream_chat()``.

    Attributes:
        content_delta: Incremental text content for this chunk.
        reasoning_delta: Incremental reasoning content when supported.
        tool_calls_delta: Incremental OpenAI-style tool call delta payload.
    """

    content_delta: str = ""
    reasoning_delta: str | None = None
    tool_calls_delta: list[dict] | None = None


class MessageRole(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessage(BaseModel):
    """Message in LLM conversation."""

    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: str | None = Field(default=None, description="Optional name (for tool messages)")
    tool_calls: list[dict] | None = Field(default=None, description="Tool calls (for assistant)")


class LLMResponse(BaseModel):
    """Response from LLM.

    A unified response structure for non-streaming chat completions.
    Holds the full response content, any tool calls requested by the model,
    finish reason, token usage, and extensible metadata.
    """

    content: str = Field(..., description="Response content")
    tool_calls: list[dict] = Field(default_factory=list, description="Tool calls if any")
    finish_reason: str = Field(..., description="Finish reason (stop, tool_calls, length, etc.)")
    usage: dict[str, Any] = Field(default_factory=dict, description="Token usage")
    model: str = Field(..., description="Model used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @classmethod
    def from_openai(cls, response: Any) -> LLMResponse:
        """Create from OpenAI SDK response object."""
        choice = response.choices[0]
        message = choice.message

        return cls(
            content=message.content or "",
            tool_calls=[tc.model_dump() for tc in (message.tool_calls or [])],
            finish_reason=choice.finish_reason,
            usage=_normalize_usage(response.usage),
            model=response.model,
        )

    @classmethod
    def from_anthropic(cls, response: Any) -> LLMResponse:
        """Create from Anthropic SDK response object."""
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input,
                        },
                    }
                )

        return cls(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            model=response.model,
        )

    @classmethod
    def from_raw_dict(cls, data: dict, model_fallback: str = "unknown") -> LLMResponse:
        """Create from a raw OpenAI-compatible JSON dict.

        Handles nested/non-int usage values (e.g. Vertex AI metadata) by
        filtering to int-only keys.
        """
        choices = data.get("choices", [])
        if not choices:
            return cls(
                content="",
                tool_calls=[],
                finish_reason="error",
                usage=_normalize_usage(data.get("usage", {})),
                model=data.get("model", model_fallback),
            )

        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        reasoning_content = msg.get("reasoning_content")

        raw_tool_calls = msg.get("tool_calls", [])
        tool_calls = _parse_tool_calls(raw_tool_calls)

        metadata: dict[str, Any] = {}
        if isinstance(reasoning_content, str) and reasoning_content:
            metadata["reasoning_content"] = reasoning_content

        return cls(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choices[0].get("finish_reason", "stop"),
            usage=_normalize_usage(data.get("usage", {})),
            model=data.get("model", model_fallback),
            metadata=metadata,
        )


class StreamResponse:
    """Wrapper for streaming chat responses.

    Receives ``StreamChunk`` objects from ``stream_chat()`` and
    performs **base-layer tool_calls delta accumulation** so that
    individual adapters don't need to repeat that logic.

    ``__anext__`` returns ``StreamChunk`` so callers can consume
    named fields (`content_delta`, `reasoning_delta`, etc.) and
    remain forward-compatible with added streaming metadata.

    After iteration, call ``finalize(adapter)`` to pull ``usage``,
    ``finish_reason``, and ``model`` from the adapter, then access
    ``stream.tool_calls``, ``stream.usage``, etc.

    Usage::

        stream = StreamResponse(adapter.stream_chat(messages, tools=tools))
        async for chunk in stream:
            print(chunk.content_delta, end="")
        stream.finalize(adapter)
        if stream.tool_calls:
            print("Tool calls:", stream.tool_calls)
    """

    def __init__(self, inner: AsyncIterator[StreamChunk]) -> None:
        self._inner = inner
        self.tool_calls: list[dict] = []
        self.usage: dict[str, Any] = {}
        self.finish_reason: str | None = None
        self.model: str | None = None
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_call_accum: dict[int, dict] = {}

    def __aiter__(self) -> StreamResponse:
        return self

    async def __anext__(self) -> StreamChunk:
        chunk = await self._inner.__anext__()
        if chunk.content_delta:
            self._content_parts.append(chunk.content_delta)
        if chunk.reasoning_delta:
            self._reasoning_parts.append(chunk.reasoning_delta)
        if chunk.tool_calls_delta:
            for tc in chunk.tool_calls_delta:
                idx = tc.get("index", 0)
                if idx not in self._tool_call_accum:
                    self._tool_call_accum[idx] = {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = self._tool_call_accum[idx]
                if tc.get("id"):
                    entry["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    entry["function"]["name"] = func["name"]
                if func.get("arguments"):
                    entry["function"]["arguments"] += func["arguments"]
        return chunk

    @property
    def accumulated_content(self) -> str:
        """Full accumulated content after iteration."""
        return "".join(self._content_parts)

    @property
    def accumulated_reasoning(self) -> str:
        """Full accumulated reasoning after iteration."""
        return "".join(self._reasoning_parts)

    def finalize(self, adapter: Any = None) -> None:
        """Finalize after the stream is fully consumed.

        1. Parse accumulated tool_calls arguments from JSON strings.
        2. If *adapter* is provided, pull ``last_usage``,
           ``last_finish_reason``, and ``model`` from it.
        """
        # Parse accumulated tool_calls
        self.tool_calls = []
        for idx in sorted(self._tool_call_accum):
            tc = self._tool_call_accum[idx]
            args_str = tc["function"]["arguments"]
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                tc["function"]["arguments"] = json.loads(args_str)
            self.tool_calls.append(tc)

        if adapter is not None:
            self.usage = getattr(adapter, "last_usage", None) or {}
            self.finish_reason = getattr(adapter, "last_finish_reason", None)
            self.model = getattr(adapter, "model", None)

    def to_response(self) -> LLMResponse:
        """Convert accumulated stream data to an LLMResponse."""
        return LLMResponse(
            content=self.accumulated_content,
            tool_calls=self.tool_calls,
            finish_reason=self.finish_reason or "stop",
            usage=self.usage,
            model=self.model or "unknown",
        )


class LLMAdapter(ABC):
    """Base class for LLM adapters.

    Adapters provide a unified interface for different LLM providers.

    Subclasses MUST implement:
    - ``chat()``        — non-streaming, returns ``LLMResponse``
    - ``stream_chat()`` — streaming, yields ``StreamChunk`` objects

    The optional ``stream_completion()`` convenience method wraps a prompt
    as a user message and delegates to ``stream_chat()``.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat completion with optional tool calling.

        Args:
            messages: Conversation messages (LLMMessage or plain dicts).
            tools: Available tools in OpenAI function-calling format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters (model, tool_choice, etc.).

        Returns:
            Complete LLM response.
        """
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion.

        Args:
            messages: Conversation messages (LLMMessage or plain dicts).
            tools: Available tools in OpenAI function-calling format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters (model, enable_reasoning,
                      thinking_budget, etc.).

        Yields:
            ``StreamChunk`` objects.
            - ``reasoning_delta`` is ``None`` for models without reasoning.
            - ``tool_calls_delta`` is ``None`` unless this chunk carries
              tool_call incremental data (OpenAI delta format).
        """
        ...
        # Need at least one yield to satisfy the async generator type
        yield StreamChunk()  # pragma: no cover

    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a single-prompt completion.

        Convenience wrapper: builds a ``[{role: user, content: prompt}]``
        message list and delegates to ``stream_chat()``.

        Yields ``StreamChunk`` objects for consistency with ``stream_chat()``.

        Args:
            prompt: Input prompt text.
            model: Model name (passed through to stream_chat via kwargs).
            **kwargs: Additional provider-specific parameters.

        Yields:
            ``StreamChunk`` objects.
        """
        messages: list[dict] = [{"role": "user", "content": prompt}]
        async for chunk in self.stream_chat(messages, model=model, **kwargs):  # type: ignore[arg-type]
            yield chunk

    def _normalize_messages(self, messages: list[LLMMessage | dict]) -> list[dict]:
        """Normalize messages to plain dict format.

        Converts ``LLMMessage`` instances to dicts while passing through
        existing dicts unchanged.

        Args:
            messages: Messages as LLMMessage or dict.

        Returns:
            List of normalized message dicts.
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                msg_dict: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
                if msg.name:
                    msg_dict["name"] = msg.name
                if msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                normalized.append(msg_dict)
            else:
                normalized.append(msg)
        return normalized

    @staticmethod
    def _coerce_message_content_to_text(value: Any) -> str:
        """Coerce message content into string for strict OpenAI-style APIs."""
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        if isinstance(value, list):
            text_parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @staticmethod
    def _sanitize_function_call_payload(function_payload: Any) -> dict[str, Any] | None:
        """Sanitize function-call payload to provider-safe string fields."""
        if not isinstance(function_payload, dict):
            return None
        normalized = dict(function_payload)
        arguments = normalized.get("arguments")
        if not isinstance(arguments, str):
            try:
                normalized["arguments"] = json.dumps(arguments, ensure_ascii=False)
            except TypeError:
                normalized["arguments"] = str(arguments)
        if normalized.get("name") is not None:
            normalized["name"] = str(normalized["name"])
        return normalized

    @staticmethod
    def _sanitize_tool_call(tool_call: Any) -> dict[str, Any] | None:
        """Sanitize one provider-compatible tool-call object."""
        if not isinstance(tool_call, dict):
            return None
        normalized = dict(tool_call)
        function_payload = LLMAdapter._sanitize_function_call_payload(normalized.get("function"))
        if function_payload is not None:
            normalized["function"] = function_payload
        return normalized

    @staticmethod
    def _sanitize_messages(
        messages: list[dict],
        *,
        enforce_string_content: bool = True,
        enforce_tool_call_arguments: bool = True,
    ) -> list[dict]:
        """Normalize provider-compatible request fields to safe strings."""
        sanitized: list[dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            normalized = dict(msg)
            if enforce_string_content:
                normalized["content"] = LLMAdapter._coerce_message_content_to_text(
                    normalized.get("content")
                )

            if enforce_tool_call_arguments and isinstance(normalized.get("tool_calls"), list):
                fixed_calls: list[dict] = []
                for call in normalized["tool_calls"]:
                    fixed = LLMAdapter._sanitize_tool_call(call)
                    if fixed is not None:
                        fixed_calls.append(fixed)
                normalized["tool_calls"] = fixed_calls

            sanitized.append(normalized)

        return sanitized


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_usage(raw: Any) -> dict[str, Any]:
    """Normalize provider usage payloads into a JSON-safe nested dict."""
    source = _extract_usage_source(raw)
    normalized = _normalize_usage_value(source)
    return normalized if isinstance(normalized, dict) else {}


def _normalize_usage_value(value: Any) -> Any:
    scalar = _normalize_usage_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, dict):
        return _normalize_usage_dict(value)

    nested_source = _extract_nested_usage_source(value)
    if nested_source:
        return _normalize_usage_dict(nested_source)

    dumped = _maybe_dump_usage_object(value)
    if isinstance(dumped, dict):
        return _normalize_usage_dict(dumped)
    return None


def _extract_usage_source(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw

    source = _extract_known_usage_fields(raw)
    if source:
        return source

    dumped = _maybe_dump_usage_object(raw)
    return dumped if isinstance(dumped, dict) else {}


def _extract_known_usage_fields(raw: Any) -> dict[str, Any]:
    source: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_details",
        "prompt_tokens_details",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ):
        value = getattr(raw, key, None)
        if value is not None:
            source[key] = value
    return source


def _normalize_usage_scalar(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _normalize_usage_dict(value: dict[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized_item = _normalize_usage_value(item)
        if normalized_item is not None:
            normalized[str(key)] = normalized_item
    return normalized


def _extract_nested_usage_source(value: Any) -> dict[str, int]:
    nested_source: dict[str, int] = {}
    for key in ("reasoning_tokens", "cached_tokens"):
        nested_value = getattr(value, key, None)
        normalized_value = _normalize_usage_scalar(nested_value)
        if normalized_value is not None:
            nested_source[key] = normalized_value
    return nested_source


def _maybe_dump_usage_object(value: Any) -> dict[str, Any] | None:
    for attr in ("model_dump", "dict"):
        dumper = getattr(value, attr, None)
        if callable(dumper):
            dumped = dumper(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
    return None


def _parse_tool_calls(raw_tool_calls: list[dict]) -> list[dict]:
    """Normalize raw tool_calls from an OpenAI-compatible response.

    Parses JSON-string arguments into dicts for convenience.
    """
    import json

    tool_calls = []
    for tc in raw_tool_calls:
        if not isinstance(tc, dict):
            continue

        normalized_tc = dict(tc)
        raw_func = tc.get("function")
        func = raw_func if isinstance(raw_func, dict) else {}
        normalized_func = dict(func)

        args = normalized_func.get("arguments", "")
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                args = json.loads(args)

        normalized_func["arguments"] = args
        normalized_func["name"] = str(normalized_func.get("name", ""))
        normalized_tc["function"] = normalized_func
        normalized_tc.setdefault("id", "")
        normalized_tc.setdefault("type", "function")

        tool_calls.append(normalized_tc)
    return tool_calls
