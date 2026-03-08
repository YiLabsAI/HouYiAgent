"""OpenAI LLM adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import (
    DEFAULT_TEMPERATURE,
    LLMAdapter,
    LLMMessage,
    LLMResponse,
    StreamChunk,
)
from houyi.adapters.llm.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    RetryController,
    RetryDecision,
    RetryPolicy,
)
from houyi.infrastructure.config.env_config import ENV_OPENAI_API_KEY

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """OpenAI LLM adapter.

    Supports GPT-4 and other OpenAI models.
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
            model: Model name (e.g., "gpt-4")
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

    def _build_retry_decision(
        self,
        *,
        retry_controller: RetryController,
        exc: Exception,
    ) -> tuple[int | None, RetryDecision]:
        status_code = self._extract_status_code(exc)
        if status_code is not None:
            headers = getattr(getattr(exc, "response", None), "headers", {})
            return status_code, retry_controller.on_status_code(
                status_code,
                method="POST",
                headers=headers,
            )
        return status_code, retry_controller.on_transport_exception(exc, method="POST")

    async def _maybe_retry(
        self,
        *,
        retry_controller: RetryController,
        exc: Exception,
        action: str,
    ) -> bool:
        status_code, decision = self._build_retry_decision(
            retry_controller=retry_controller,
            exc=exc,
        )
        if not decision.retry:
            return False
        logger.warning(
            "OpenAI %s retry: bucket=%s used=%d/%d wait=%.2fs status=%s error=%s",
            action,
            decision.bucket,
            retry_controller.retries_used,
            retry_controller.policy.total_retries,
            decision.delay_seconds,
            status_code,
            exc,
        )
        await asyncio.sleep(decision.delay_seconds)
        return True

    def _update_stream_usage(self, chunk: Any) -> None:
        if not chunk.usage:
            return
        self.last_usage = {
            "prompt_tokens": chunk.usage.prompt_tokens or 0,
            "completion_tokens": chunk.usage.completion_tokens or 0,
            "total_tokens": chunk.usage.total_tokens or 0,
        }

    @staticmethod
    def _extract_tool_calls_delta(delta: Any) -> list[dict] | None:
        if not delta or not delta.tool_calls:
            return None
        tc_delta_list: list[dict] = []
        for tc_d in delta.tool_calls:
            tc_dict: dict[str, Any] = {"index": tc_d.index}
            if tc_d.id:
                tc_dict["id"] = tc_d.id
            if tc_d.function:
                func: dict[str, Any] = {}
                if tc_d.function.name:
                    func["name"] = tc_d.function.name
                if tc_d.function.arguments:
                    func["arguments"] = tc_d.function.arguments
                if func:
                    tc_dict["function"] = func
            tc_delta_list.append(tc_dict)
        return tc_delta_list

    def _build_stream_chunk(self, choice: Any) -> StreamChunk | None:
        delta = choice.delta
        if choice.finish_reason:
            self.last_finish_reason = choice.finish_reason

        tc_delta_list = self._extract_tool_calls_delta(delta)
        content = delta.content if delta else None
        if not content and not tc_delta_list:
            return None
        return StreamChunk(
            content_delta=content or "",
            reasoning_delta=None,
            tool_calls_delta=tc_delta_list,
        )

    def _build_chat_params(
        self,
        *,
        messages: list[LLMMessage | dict],
        temperature: float,
        max_tokens: int | None,
        tools: list[dict] | None,
        stream: bool,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": self._normalize_messages(messages),
            "temperature": temperature,
        }
        if stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        params.update(extra_kwargs)
        return params

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
        params = self._build_chat_params(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=False,
            extra_kwargs=kwargs,
        )

        retry_controller = self._build_retry_controller()
        while True:
            try:
                response = await self.client.chat.completions.create(**params)
                break
            except Exception as exc:
                if await self._maybe_retry(
                    retry_controller=retry_controller,
                    exc=exc,
                    action="chat",
                ):
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
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion with OpenAI.

        Yields:
            ``StreamChunk`` objects.
            ``tool_calls_delta`` is populated with OpenAI delta dicts when present.
        """
        params = self._build_chat_params(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=True,
            extra_kwargs=kwargs,
        )

        self.last_usage = {}
        self.last_finish_reason = None

        retry_controller = self._build_retry_controller()
        while True:
            emitted = False
            try:
                stream = await self.client.chat.completions.create(**params)
                async for chunk in stream:
                    self._update_stream_usage(chunk)

                    if not chunk.choices:
                        continue

                    stream_chunk = self._build_stream_chunk(chunk.choices[0])
                    if stream_chunk is None:
                        continue

                    emitted = emitted or bool(stream_chunk.content_delta)
                    yield stream_chunk

                return
            except Exception as exc:
                if emitted:
                    raise
                if await self._maybe_retry(
                    retry_controller=retry_controller,
                    exc=exc,
                    action="stream",
                ):
                    continue
                raise
