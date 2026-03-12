"""SiliconFlow provider adapter.

Strategy: prefer the OpenAI client path and fall back to raw ``httpx`` SSE.
The client path provides retries, typed errors, streaming support, and usage tracking.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import (
    DEFAULT_TEMPERATURE,
    LLMMessage,
    LLMResponse,
    StreamChunk,
)
from houyi.adapters.llm.openai_compat_base import OpenAICompatAdapterBase
from houyi.adapters.llm.request_models import OpenAICompatRequest
from houyi.adapters.llm.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    RetryController,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


def _format_siliconflow_http_error(status_code: int) -> str:
    if status_code == 400:
        return "SiliconFlow rejected the request as invalid. Please retry or adjust the request payload."
    if status_code == 401:
        return "SiliconFlow authentication failed. Check the configured API key and retry."
    if status_code == 403:
        return "SiliconFlow rejected the request due to missing permissions. Check the configured account access and retry."
    if status_code == 404:
        return "SiliconFlow could not find the requested model or endpoint. Check the configured model and base URL."
    if status_code == 429:
        return "SiliconFlow is temporarily rate limited. Please retry in a moment."
    if status_code >= 500:
        return "SiliconFlow is temporarily unavailable. Please retry in a moment."
    return f"SiliconFlow request failed with HTTP {status_code}. Please retry in a moment."


def _tool_delta_impl(choice: Any) -> list[dict] | None:
    delta = choice.delta
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


class SiliconFlowAdapter(OpenAICompatAdapterBase):
    """Adapter for the SiliconFlow provider.

    Strategy: prefer the OpenAI client path and fall back to raw ``httpx`` SSE.
    The client path provides retries, typed errors, streaming support, and usage
    tracking.
    """

    _OPENAI_READY: bool | None = None

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        strict_message_string_contract: bool = True,
    ):
        from houyi.infrastructure.config.env_config import EnvConfig

        _env = EnvConfig.get()

        self.api_key = api_key or _env.siliconflow_api_key
        self.base_url = base_url or _env.siliconflow_base_url
        self.default_model = default_model or _env.deepseek_model
        self.strict_message_string_contract = strict_message_string_contract
        self.model = self.default_model  # uniform interface for callers
        self.last_usage: dict[str, Any] | None = None
        self.last_finish_reason: str | None = None

        if not self.api_key:
            logger.debug("SILICONFLOW_API_KEY not set, will use mock responses")

        if SiliconFlowAdapter._OPENAI_READY is None:
            try:
                import openai  # noqa: F401

                SiliconFlowAdapter._OPENAI_READY = True
                logger.info("openai client available — using client mode")
            except ImportError:
                SiliconFlowAdapter._OPENAI_READY = False
                logger.warning(
                    "openai client not installed — falling back to httpx raw SSE. "
                    "Install with: pip install 'houyi[model-adapters]' or pip install openai"
                )

    # ── chat() — non-streaming ────────────────────────────────────

    def _sanitize_request_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure OpenAI-compatible request fields are string-typed.

        SiliconFlow rejects non-string `messages[*].content` and may also reject
        non-string `assistant.tool_calls[*].function.arguments`.
        """
        sanitized = self._sanitize_messages(messages)
        for message in sanitized:
            if (
                message.get("role") == "assistant"
                and isinstance(message.get("tool_calls"), list)
                and "reasoning_content" not in message
            ):
                # SiliconFlow thinking mode may require this field to be present
                # on assistant tool-call messages even when empty.
                message["reasoning_content"] = ""
        return sanitized

    def _resolve_transport(self, request: OpenAICompatRequest) -> str:
        route = str(os.getenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "auto") or "auto").strip().lower()
        if route in {"sdk", "httpx"}:
            return route
        if request.enable_streaming:
            return "httpx"
        if SiliconFlowAdapter._OPENAI_READY:
            return "sdk"
        return "httpx"

    @classmethod
    def _prepare_messages_for_request(
        cls,
        request: OpenAICompatRequest,
    ) -> list[dict[str, Any]]:
        messages = [dict(message) for message in request.messages]
        if request.tools is not None:
            return messages

        prepared: list[dict[str, Any]] = []
        for message in messages:
            normalized = dict(message)
            if (
                str(normalized.get("role") or "") == "assistant"
                and isinstance(normalized.get("tool_calls"), list)
                and normalized.get("content") == ""
            ):
                normalized["content"] = "[tool call]"
            prepared.append(normalized)
        return prepared

    def _prepare_request_for_provider(
        self,
        request: OpenAICompatRequest,
    ) -> OpenAICompatRequest:
        # SiliconFlow rejects empty assistant content on tool-call turns when no
        # tool schema is present, so normalize those turns before either path encodes.
        return self._copy_request(
            request,
            messages=self._prepare_messages_for_request(request),
        )

    def _build_reasoning_extra_body(
        self,
        request: OpenAICompatRequest,
    ) -> dict[str, object] | None:
        if not request.enable_thinking or request.thinking_budget is None:
            return None
        logger.debug("Thinking enabled with thinking_budget=%d", request.thinking_budget)
        return {"thinking_budget": request.thinking_budget}

    def _new_client(self) -> Any:
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )

    async def _close_client(self, client: Any) -> None:
        await client.close()

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat completion with tool calling support."""
        request = self._build_request(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_streaming=False,
            kwargs=dict(kwargs),
        )

        if not self.api_key:
            return LLMResponse(
                content="Mock response (no API key)",
                tool_calls=[],
                finish_reason="stop",
                usage={},
                model=request.model,
            )

        return await super()._chat(request)

    def _get_httpx_chat_proxy_url(self) -> str | None:
        from houyi.infrastructure.net.proxy import detect_proxy

        return detect_proxy()

    def _chat_transport(
        self,
        *,
        exc: Exception,
        retry_controller: RetryController,
    ) -> tuple[bool, float]:
        decision = retry_controller.on_transport_exception(exc, method="POST")
        if decision.retry:
            logger.warning(
                "SiliconFlow chat transport error: bucket=%s retry=%d/%d wait=%.2fs error=%s",
                decision.bucket,
                retry_controller.retries_used,
                retry_controller.policy.total_retries,
                decision.delay_seconds,
                exc,
            )
        return decision.retry, decision.delay_seconds

    async def _chat_status(
        self,
        *,
        response: Any,
        retry_controller: RetryController,
    ) -> tuple[bool, Exception | None]:
        if response.status_code not in retry_controller.policy.status_forcelist:
            return False, None

        decision = retry_controller.on_status_code(
            response.status_code,
            method="POST",
            headers=response.headers,
        )
        if not decision.retry:
            return False, None

        last_error = RuntimeError(f"SiliconFlow HTTP {response.status_code}: {response.text[:500]}")
        logger.warning(
            "SiliconFlow chat HTTP %d retry=%d/%d wait=%.2fs",
            response.status_code,
            retry_controller.retries_used,
            retry_controller.policy.total_retries,
            decision.delay_seconds,
        )
        await asyncio.sleep(decision.delay_seconds)
        return True, last_error

    @staticmethod
    def _parse_httpx_chat_response(response: Any) -> dict[str, Any]:
        if response.status_code >= 400:
            raise RuntimeError(_format_siliconflow_http_error(int(response.status_code)))
        return response.json()

    def _chat_retry(self) -> RetryController:
        return self._new_retry_controller(status_only=True)

    @staticmethod
    def _new_retry_controller(*, status_only: bool) -> RetryController:
        return RetryController(
            RetryPolicy(
                total_retries=DEFAULT_MAX_RETRIES,
                connect_retries=None if status_only else DEFAULT_MAX_RETRIES,
                read_retries=None if status_only else DEFAULT_MAX_RETRIES,
                status_retries=DEFAULT_MAX_RETRIES,
                backoff_base=DEFAULT_RETRY_BASE_DELAY,
                backoff_cap=DEFAULT_RETRY_MAX_DELAY,
            )
        )

    # ── stream_chat() — streaming ─────────────────────────────────

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion with full message list.

        Yields ``StreamChunk`` objects.
        """
        request = self._build_request(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_streaming=True,
            kwargs=dict(kwargs),
        )

        if not self.api_key:
            logger.info("Using mock streaming (no API key)")
            last_content = ""
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    last_content = msg.get("content", "")
                    break
            words = f"Mock response from {request.model}: {last_content[:50]}...".split()
            for word in words:
                yield StreamChunk(content_delta=word + " ")
            return

        async for chunk in self._stream_chat(request):
            yield chunk

    _tool_delta = staticmethod(_tool_delta_impl)

    def _build_stream_chunk(
        self,
        *,
        choice: Any,
    ) -> tuple[StreamChunk | None, int, int]:
        # Reuse the base content/reasoning assembly, then layer SiliconFlow's
        # incremental tool-call deltas on top of that shared chunk model.
        base_chunk, content_inc, reasoning_inc = super()._build_stream_chunk(choice=choice)
        tc_delta_list = self._tool_delta(choice)
        if base_chunk is None and not tc_delta_list:
            return None, content_inc, reasoning_inc

        return (
            StreamChunk(
                content_delta=base_chunk.content_delta if base_chunk is not None else "",
                reasoning_delta=base_chunk.reasoning_delta if base_chunk is not None else None,
                tool_calls_delta=tc_delta_list,
            ),
            content_inc,
            reasoning_inc,
        )

    async def _get_stream_client(self, params: dict[str, Any]) -> Any:
        client = self._new_client()
        logger.debug("Streaming to %s model=%s", self.base_url, params.get("model"))
        return client

    def _log_stream(
        self,
        *,
        params: dict[str, Any],
        chunk_count: int,
        reasoning_count: int,
    ) -> None:
        # Keep stream completion logging here so provider-specific counters stay visible.
        logger.debug(
            "Stream completed: %d content, %d reasoning chunks, finish=%s, usage=%s",
            chunk_count,
            reasoning_count,
            self.last_finish_reason,
            self.last_usage,
        )

    async def _stream_status(
        self,
        *,
        response: Any,
        retry_controller: RetryController,
    ) -> bool:
        # SiliconFlow returns provider-specific error payloads, so status handling stays local.
        error_text = await self._read_httpx_error_text(response)
        should_retry, decision = self._retry_httpx_stream_status(
            response=response,
            retry_controller=retry_controller,
        )
        if should_retry and decision is not None:
            logger.warning(
                "httpx API error %d (retry bucket=%s used=%d/%d wait=%.2fs): %s",
                response.status_code,
                decision.bucket,
                retry_controller.retries_used,
                retry_controller.policy.total_retries,
                decision.delay_seconds,
                error_text,
            )
            await asyncio.sleep(decision.delay_seconds)
            return True
        logger.error("httpx API error %d: %s", response.status_code, error_text)
        raise RuntimeError(_format_siliconflow_http_error(int(response.status_code)))

    def _stream_retry(self) -> RetryController:
        return self._new_retry_controller(status_only=False)
