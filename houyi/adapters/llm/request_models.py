"""Transport-agnostic request models for the OpenAI-compatible adapter family.

All business-semantic chat parameters must first be normalized into a
transport-agnostic request model, and only then consumed by transport-specific
encoders or executors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OpenAICompatRequest:
    """Normalized business-semantic request for OpenAI-compatible adapters."""

    model: str
    messages: list[dict[str, Any]]
    temperature: float
    tools: list[dict[str, Any]] | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    enable_streaming: bool = False
    enable_thinking: bool = False
    thinking_budget: int | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        enable_streaming: bool = False,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> OpenAICompatRequest:
        payload = dict(extra_kwargs or {})
        top_p = payload.pop("top_p", None)
        top_k = payload.pop("top_k", None)
        frequency_penalty = payload.pop("frequency_penalty", None)
        tool_choice = payload.pop("tool_choice", None)
        enable_thinking = bool(
            payload.pop("enable_thinking", payload.pop("enable_reasoning", False))
        )
        thinking_budget = payload.pop("thinking_budget", payload.pop("reasoning_budget", None))
        return cls(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=enable_streaming,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            extra_kwargs=payload,
        )

    @property
    def enable_reasoning(self) -> bool:
        return self.enable_thinking

    @property
    def reasoning_budget(self) -> int | None:
        return self.thinking_budget

    def to_sdk_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
        }
        if self.enable_streaming:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.tools:
            kwargs["tools"] = self.tools
        if self.tool_choice is not None:
            kwargs["tool_choice"] = self.tool_choice
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        kwargs.update(self.extra_kwargs)
        return kwargs

    def to_httpx_payload(self) -> dict[str, Any]:
        payload = self.to_sdk_kwargs()
        payload.setdefault("stream", self.enable_streaming)
        return payload
