"""Tool-call adapter interfaces and response normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from houyi.adapters.llm.base import LLMResponse


class ToolCallAdapter(Protocol):
    """Protocol for adapters returning OpenAI-compatible tool calls."""

    async def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class ToolCallAdapterError:
    """Normalized adapter error payload."""

    error_type: str
    message: str
    retryable: bool = False


def normalize_adapter_response(response: Any) -> LLMResponse:
    """Normalize provider responses into LLMResponse."""

    if isinstance(response, LLMResponse):
        return response
    if hasattr(response, "choices"):
        return LLMResponse.from_openai(response)
    if hasattr(response, "content") and hasattr(response, "stop_reason"):
        return LLMResponse.from_anthropic(response)
    raise TypeError("Unsupported adapter response type")


def normalize_adapter_error(error: Exception) -> ToolCallAdapterError:
    """Normalize adapter errors for unified handling."""

    error_type = error.__class__.__name__
    retryable = error_type.lower() in {"timeout", "ratelimiterror", "temporarilyunavailable"}
    return ToolCallAdapterError(error_type=error_type, message=str(error), retryable=retryable)
