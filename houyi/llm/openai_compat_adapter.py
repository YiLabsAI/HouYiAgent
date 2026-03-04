"""OpenAI-compatible adapter for OpenAI-style providers."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.config.env_config import ENV_OPENAI_API_KEY, ENV_OPENAI_BASE_URL, ENV_OPENAI_ORG
from houyi.llm.base import DEFAULT_TEMPERATURE, LLMAdapter, LLMMessage, LLMResponse, StreamChunk


class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible providers (OpenAI-style APIs)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
        organization: str | None = None,
        default_headers: dict[str, str] | None = None,
        strict_message_string_contract: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv(ENV_OPENAI_API_KEY)
        self.model = model
        self.base_url = base_url or os.getenv(ENV_OPENAI_BASE_URL)
        self.organization = organization or os.getenv(ENV_OPENAI_ORG)
        self.default_headers = default_headers or {}
        self.strict_message_string_contract = strict_message_string_contract

        if not self.api_key:
            raise ValueError(
                "OpenAI-compatible API key not provided. "
                "Set OPENAI_API_KEY or pass api_key parameter."
            )

        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                default_headers=self.default_headers or None,
            )
        except ImportError as exc:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai>=1.0.0"
            ) from exc

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        normalized_messages = self._normalize_messages(messages)
        if self.strict_message_string_contract:
            normalized_messages = self._sanitize_messages(normalized_messages)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        params.update(kwargs)

        response = await self.client.chat.completions.create(**params)
        return LLMResponse.from_openai(response)

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion.

        Yields:
            ``StreamChunk`` objects.
        """
        normalized_messages = self._normalize_messages(messages)
        if self.strict_message_string_contract:
            normalized_messages = self._sanitize_messages(normalized_messages)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        params.update(kwargs)

        stream = await self.client.chat.completions.create(**params)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield StreamChunk(content_delta=chunk.choices[0].delta.content)
