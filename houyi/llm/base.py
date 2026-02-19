"""Base classes for LLM adapters.

Defines the unified contract that all LLM provider adapters must implement:
- ``chat()`` — non-streaming completion with tool calling support
- ``stream_chat()`` — streaming completion yielding (content, reasoning) tuples

Provider-specific adapters (OpenAI, Anthropic, SiliconFlow, Vertex AI, etc.)
inherit from ``LLMAdapter`` and implement the two abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared defaults (single source of truth for all adapters)
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE: float = 0.7
"""Default sampling temperature used across all LLM adapters."""


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
    usage: dict[str, int] = Field(default_factory=dict, description="Token usage")
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
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
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
                usage=_sanitize_usage(data.get("usage", {})),
                model=data.get("model", model_fallback),
            )

        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""

        raw_tool_calls = msg.get("tool_calls", [])
        tool_calls = _parse_tool_calls(raw_tool_calls)

        return cls(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choices[0].get("finish_reason", "stop"),
            usage=_sanitize_usage(data.get("usage", {})),
            model=data.get("model", model_fallback),
        )


class StreamResponse:
    """Wrapper for streaming chat responses (Scheme C).

    Accumulates tool_calls and usage as the stream is consumed.
    Compatible with the ``async for content, reasoning in stream`` pattern
    while also providing post-iteration access to accumulated metadata.

    Usage::

        stream = StreamResponse(adapter.stream_chat(messages, tools=tools))
        async for content, reasoning in stream:
            print(content, end="")
        # After iteration:
        if stream.tool_calls:
            print("Tool calls:", stream.tool_calls)
        print("Usage:", stream.usage)
        print("Finish reason:", stream.finish_reason)
    """

    def __init__(self, inner: AsyncIterator[tuple[str, str | None]]) -> None:
        self._inner = inner
        self.tool_calls: list[dict] = []
        self.usage: dict[str, int] = {}
        self.finish_reason: str | None = None
        self.model: str | None = None
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> tuple[str, str | None]:
        return await self._inner.__anext__()

    @property
    def accumulated_content(self) -> str:
        """Full accumulated content after iteration."""
        return "".join(self._content_parts)

    @property
    def accumulated_reasoning(self) -> str:
        """Full accumulated reasoning after iteration."""
        return "".join(self._reasoning_parts)

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
    - ``stream_chat()`` — streaming, yields ``(content_delta, reasoning_delta)``

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
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Streaming chat completion.

        Args:
            messages: Conversation messages (LLMMessage or plain dicts).
            tools: Available tools in OpenAI function-calling format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters (model, enable_reasoning,
                      thinking_budget, etc.).

        Yields:
            ``(content_delta, reasoning_delta)`` tuples.
            ``reasoning_delta`` is ``None`` for models without reasoning support.
        """
        ...
        # Need at least one yield to satisfy the async generator type
        yield ("", None)  # pragma: no cover

    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream a single-prompt completion.

        Convenience wrapper: builds a ``[{role: user, content: prompt}]``
        message list and delegates to ``stream_chat()``.

        Args:
            prompt: Input prompt text.
            model: Model name (passed through to stream_chat via kwargs).
            **kwargs: Additional provider-specific parameters.

        Yields:
            ``(content_delta, reasoning_delta)`` tuples.
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sanitize_usage(raw: dict | None) -> dict[str, int]:
    """Filter usage dict to int-only values.

    Some providers (Vertex AI) include nested objects in usage
    (e.g. ``completion_tokens_details``). ``LLMResponse.usage`` expects
    ``dict[str, int]``, so we filter out non-int values.
    """
    if not raw:
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, int)}


def _parse_tool_calls(raw_tool_calls: list[dict]) -> list[dict]:
    """Normalize raw tool_calls from an OpenAI-compatible response.

    Parses JSON-string arguments into dicts for convenience.
    """
    import json

    tool_calls = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        tool_calls.append(
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                },
            }
        )
    return tool_calls
