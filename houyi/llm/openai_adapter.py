"""OpenAI LLM adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.config.env_config import ENV_OPENAI_API_KEY
from houyi.llm.base import DEFAULT_TEMPERATURE, LLMAdapter, LLMMessage, LLMResponse
from houyi.llm.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    RetryController,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """OpenAI LLM adapter.

    Supports GPT-4, GPT-3.5, and other OpenAI models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
    ):
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: Optional base URL for API
        """
        self.api_key = api_key or os.getenv(ENV_OPENAI_API_KEY)
        self.model = model
        self.base_url = base_url
        self._retry_policy = RetryPolicy(
            total_retries=DEFAULT_MAX_RETRIES,
            status_retries=DEFAULT_MAX_RETRIES,
            backoff_base=DEFAULT_RETRY_BASE_DELAY,
            backoff_cap=DEFAULT_RETRY_MAX_DELAY,
        )

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Lazy import to avoid requiring openai package if not used
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=DEFAULT_MAX_RETRIES,
            )
        except ImportError as e:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai>=1.0.0"
            ) from e

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        return None

    def _build_retry_controller(self) -> RetryController:
        return RetryController(self._retry_policy)

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with OpenAI.

        Args:
            messages: Conversation messages
            tools: Available tools (OpenAI function calling format)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters

        Returns:
            LLM response
        """
        # Normalize messages
        normalized_messages = self._normalize_messages(messages)

        # Build request parameters
        params = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": temperature,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        if tools:
            params["tools"] = tools

        # Add any additional parameters
        params.update(kwargs)

        retry_controller = self._build_retry_controller()
        while True:
            try:
                response = await self.client.chat.completions.create(**params)
                break
            except Exception as exc:
                status_code = self._extract_status_code(exc)
                decision = None
                if status_code is not None:
                    headers = getattr(getattr(exc, "response", None), "headers", {})
                    decision = retry_controller.on_status_code(
                        status_code,
                        method="POST",
                        headers=headers,
                    )
                if decision is None:
                    decision = retry_controller.on_transport_exception(exc, method="POST")

                if decision.retry:
                    logger.warning(
                        "OpenAI chat retry: bucket=%s used=%d/%d wait=%.2fs status=%s error=%s",
                        decision.bucket,
                        retry_controller.retries_used,
                        retry_controller.policy.total_retries,
                        decision.delay_seconds,
                        status_code,
                        exc,
                    )
                    await asyncio.sleep(decision.delay_seconds)
                    continue
                raise

        return LLMResponse.from_openai(response)

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Streaming chat completion with OpenAI.

        Args:
            messages: Conversation messages
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters

        Yields:
            ``(content_delta, None)`` tuples (OpenAI does not expose reasoning).
        """
        normalized_messages = self._normalize_messages(messages)

        params = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        if tools:
            params["tools"] = tools

        params.update(kwargs)

        retry_controller = self._build_retry_controller()
        while True:
            emitted = False
            try:
                stream = await self.client.chat.completions.create(**params)
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        emitted = True
                        yield (chunk.choices[0].delta.content, None)
                return
            except Exception as exc:
                if emitted:
                    raise

                status_code = self._extract_status_code(exc)
                decision = None
                if status_code is not None:
                    headers = getattr(getattr(exc, "response", None), "headers", {})
                    decision = retry_controller.on_status_code(
                        status_code,
                        method="POST",
                        headers=headers,
                    )
                if decision is None:
                    decision = retry_controller.on_transport_exception(exc, method="POST")

                if decision.retry:
                    logger.warning(
                        "OpenAI stream retry: bucket=%s used=%d/%d wait=%.2fs status=%s error=%s",
                        decision.bucket,
                        retry_controller.retries_used,
                        retry_controller.policy.total_retries,
                        decision.delay_seconds,
                        status_code,
                        exc,
                    )
                    await asyncio.sleep(decision.delay_seconds)
                    continue
                raise
