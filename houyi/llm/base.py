"""Base classes for LLM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
    """Response from LLM."""

    content: str = Field(..., description="Response content")
    tool_calls: list[dict] = Field(default_factory=list, description="Tool calls if any")
    finish_reason: str = Field(..., description="Finish reason (stop, tool_calls, length, etc.)")
    usage: dict[str, int] = Field(default_factory=dict, description="Token usage")
    model: str = Field(..., description="Model used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @classmethod
    def from_openai(cls, response: Any) -> LLMResponse:
        """Create from OpenAI response."""
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
        """Create from Anthropic response."""
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    }
                })

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


class LLMAdapter(ABC):
    """Base class for LLM adapters.

    Adapters provide a unified interface for different LLM providers.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion.

        Args:
            messages: Conversation messages
            tools: Available tools (OpenAI function calling format)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming chat completion.

        Args:
            messages: Conversation messages
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Response chunks
        """
        pass

    def _normalize_messages(
        self,
        messages: list[LLMMessage | dict]
    ) -> list[dict]:
        """Normalize messages to dict format.

        Args:
            messages: Messages as LLMMessage or dict

        Returns:
            Normalized message dicts
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                msg_dict = {"role": msg.role.value, "content": msg.content}
                if msg.name:
                    msg_dict["name"] = msg.name
                if msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                normalized.append(msg_dict)
            else:
                normalized.append(msg)
        return normalized
