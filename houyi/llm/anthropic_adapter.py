"""Anthropic (Claude) LLM adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.config.env_config import ENV_ANTHROPIC_API_KEY
from houyi.llm.base import DEFAULT_TEMPERATURE, LLMAdapter, LLMMessage, LLMResponse, StreamChunk


class AnthropicAdapter(LLMAdapter):
    """Anthropic (Claude) LLM adapter.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, and other Anthropic models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        """Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
        """
        self.api_key = api_key or os.getenv(ENV_ANTHROPIC_API_KEY)
        self.model = model

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        # Lazy import to avoid requiring anthropic package if not used
        try:
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic>=0.18.0"
            ) from e

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with Anthropic.

        Args:
            messages: Conversation messages
            tools: Available tools (OpenAI format, will be converted)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters

        Returns:
            LLM response
        """
        # Normalize messages
        normalized_messages = self._normalize_messages(messages)

        # Extract system message (Anthropic requires separate system parameter)
        system_message = None
        filtered_messages = []
        for msg in normalized_messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }

        if system_message:
            params["system"] = system_message

        if tools:
            # Convert OpenAI tool format to Anthropic format
            params["tools"] = self._convert_tools_to_anthropic(tools)

        # Add any additional parameters
        params.update(kwargs)

        # Make API call
        response = await self.client.messages.create(**params)

        return LLMResponse.from_anthropic(response)

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion with Anthropic.

        Yields:
            ``StreamChunk`` objects.
            Anthropic tool_use arrives as complete blocks (not deltas),
            so ``tool_calls_delta`` is always ``None``.
        """
        normalized_messages = self._normalize_messages(messages)

        system_message = None
        filtered_messages = []
        for msg in normalized_messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }

        if system_message:
            params["system"] = system_message

        if tools:
            params["tools"] = self._convert_tools_to_anthropic(tools)

        params.update(kwargs)

        self.last_usage: dict[str, int] = {}
        self.last_finish_reason: str | None = None

        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(content_delta=text)

            final_message = await stream.get_final_message()

            if final_message.stop_reason:
                self.last_finish_reason = final_message.stop_reason

            if final_message.usage:
                self.last_usage = {
                    "prompt_tokens": final_message.usage.input_tokens,
                    "completion_tokens": final_message.usage.output_tokens,
                    "total_tokens": (
                        final_message.usage.input_tokens + final_message.usage.output_tokens
                    ),
                }

    def _convert_tools_to_anthropic(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool format to Anthropic format.

        Args:
            tools: Tools in OpenAI format

        Returns:
            Tools in Anthropic format
        """
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
        return anthropic_tools
