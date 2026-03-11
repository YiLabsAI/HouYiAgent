"""SiliconFlow / OpenAI-compatible LLM adapter.

Supports any OpenAI-compatible endpoint: SiliconFlow, DeepSeek, vLLM, Ollama, etc.

Strategy: ``openai`` SDK first, ``httpx`` raw SSE as fallback.
The openai SDK provides automatic retries, proper error types, streaming support,
and token usage tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    SSE_DATA_PREFIX,
    SSE_DONE_SIGNAL,
    USAGE_KEY_COMPLETION_TOKENS,
    USAGE_KEY_PROMPT_TOKENS,
    USAGE_KEY_TOTAL_TOKENS,
    RetryController,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


class SiliconFlowAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible APIs (SiliconFlow, DeepSeek, vLLM, Ollama, etc.).

    Strategy: openai SDK first, httpx raw SSE as fallback.
    The openai SDK provides automatic retries, proper error types, streaming
    support, and token usage tracking.
    """

    _SDK_AVAILABLE: bool | None = None

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
        self.last_usage: dict[str, int] | None = None
        self.last_finish_reason: str | None = None
        self._sdk_client: object | None = None

        if not self.api_key:
            logger.debug("SILICONFLOW_API_KEY not set, will use mock responses")

        if SiliconFlowAdapter._SDK_AVAILABLE is None:
            try:
                import openai  # noqa: F401

                SiliconFlowAdapter._SDK_AVAILABLE = True
                logger.info("openai SDK available — using SDK mode (recommended)")
            except ImportError:
                SiliconFlowAdapter._SDK_AVAILABLE = False
                logger.warning(
                    "openai SDK not installed — falling back to httpx raw SSE. "
                    "Install with: pip install 'houyi[model-adapters]' or pip install openai"
                )

    # ── chat() — non-streaming ────────────────────────────────────

    @classmethod
    def _sanitize_chat_messages(cls, messages: list[dict]) -> list[dict]:
        """Ensure OpenAI-compatible request fields are string-typed.

        SiliconFlow rejects non-string `messages[*].content` and may also reject
        non-string `assistant.tool_calls[*].function.arguments`.
        """
        sanitized = cls._sanitize_messages(messages)
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

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat completion with tool calling support."""
        model = kwargs.pop("model", None) or self.default_model
        normalized = self._normalize_messages(messages)
        if self.strict_message_string_contract:
            normalized = self._sanitize_chat_messages(normalized)

        if not self.api_key:
            return LLMResponse(
                content="Mock response (no API key)",
                tool_calls=[],
                finish_reason="stop",
                usage={},
                model=model,
            )

        if SiliconFlowAdapter._SDK_AVAILABLE:
            return await self._chat_via_sdk(
                normalized, model, tools, temperature, max_tokens, **kwargs
            )
        return await self._chat_via_httpx(
            normalized, model, tools, temperature, max_tokens, **kwargs
        )

    async def _chat_via_sdk(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat via openai SDK."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            params["tool_choice"] = tool_choice
        for k, v in kwargs.items():
            if v is not None and k not in params:
                params[k] = v

        try:
            response = await client.chat.completions.create(**params)
            result = LLMResponse.from_openai(response)
            self.last_usage = result.usage
            return result
        finally:
            await client.close()

    def _build_httpx_chat_body(
        self,
        *,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools

        tool_choice = extra_kwargs.pop("tool_choice", None)
        if tool_choice:
            body["tool_choice"] = tool_choice
        return body

    async def _maybe_retry_httpx_chat_transport_error(
        self,
        *,
        exc: Exception,
        retry_controller: RetryController,
    ) -> bool:
        decision = retry_controller.on_transport_exception(exc, method="POST")
        if not decision.retry:
            return False

        logger.warning(
            "SiliconFlow chat transport error: bucket=%s retry=%d/%d wait=%.2fs error=%s",
            decision.bucket,
            retry_controller.retries_used,
            retry_controller.policy.total_retries,
            decision.delay_seconds,
            exc,
        )
        await asyncio.sleep(decision.delay_seconds)
        return True

    async def _maybe_retry_httpx_chat_status(
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
            error_text = response.text[:1000]
            raise RuntimeError(f"SiliconFlow HTTP {response.status_code}: {error_text}")
        return response.json()

    async def _chat_via_httpx(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat via httpx fallback with proxy + retry."""
        import httpx

        from houyi.infrastructure.net.proxy import detect_proxy

        url = self.base_url.rstrip("/") + "/chat/completions"
        body = self._build_httpx_chat_body(
            messages=messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=kwargs,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        proxy_url = detect_proxy()
        retry_controller = RetryController(
            RetryPolicy(
                total_retries=DEFAULT_MAX_RETRIES,
                status_retries=DEFAULT_MAX_RETRIES,
                backoff_base=DEFAULT_RETRY_BASE_DELAY,
                backoff_cap=DEFAULT_RETRY_MAX_DELAY,
            )
        )
        last_error: Exception | None = None

        while True:
            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url,
                    timeout=httpx.Timeout(60.0, connect=10.0),
                ) as client:
                    resp = await client.post(url, json=body, headers=headers)
            except httpx.TransportError as exc:
                if await self._maybe_retry_httpx_chat_transport_error(
                    exc=exc,
                    retry_controller=retry_controller,
                ):
                    last_error = exc
                    continue
                raise

            should_retry, maybe_error = await self._maybe_retry_httpx_chat_status(
                response=resp,
                retry_controller=retry_controller,
            )
            if should_retry:
                last_error = maybe_error
                continue

            data = self._parse_httpx_chat_response(resp)
            break

        if not data:
            raise last_error or RuntimeError("SiliconFlow chat: max retries exhausted")

        result = LLMResponse.from_raw_dict(data, model_fallback=model)
        self.last_usage = result.usage
        return result

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
        model = kwargs.pop("model", None) or self.default_model
        enable_reasoning = kwargs.pop("enable_reasoning", False)
        thinking_budget = kwargs.pop("thinking_budget", None)
        normalized = self._normalize_messages(messages)
        if self.strict_message_string_contract:
            normalized = self._sanitize_chat_messages(normalized)

        if not self.api_key:
            logger.info("Using mock streaming (no API key)")
            last_content = ""
            for msg in reversed(normalized):
                if msg.get("role") == "user":
                    last_content = msg.get("content", "")
                    break
            words = f"Mock response from {model}: {last_content[:50]}...".split()
            for word in words:
                yield StreamChunk(content_delta=word + " ")
            return

        if SiliconFlowAdapter._SDK_AVAILABLE:
            async for chunk in self._stream_via_sdk(
                normalized, model, enable_reasoning, thinking_budget, **kwargs
            ):
                yield chunk
        else:
            async for chunk in self._stream_via_httpx(
                normalized, model, enable_reasoning, thinking_budget, **kwargs
            ):
                yield chunk

    def _build_sdk_stream_kwargs(
        self,
        *,
        messages: list[dict],
        model: str,
        enable_reasoning: bool,
        thinking_budget: int | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        extra_body: dict[str, object] = {}
        if enable_reasoning and thinking_budget:
            extra_body["thinking_budget"] = thinking_budget
            logger.info("Reasoning enabled with thinking_budget=%d", thinking_budget)

        tools = extra_kwargs.pop("tools", None)
        sdk_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            sdk_kwargs["tools"] = tools

        tool_choice = extra_kwargs.pop("tool_choice", None)
        if tool_choice:
            sdk_kwargs["tool_choice"] = tool_choice

        if extra_body:
            sdk_kwargs["extra_body"] = extra_body

        for key, value in extra_kwargs.items():
            if value is not None and key not in sdk_kwargs:
                sdk_kwargs[key] = value
        return sdk_kwargs

    @staticmethod
    def _extract_sdk_tool_calls_delta(choice: Any) -> list[dict] | None:
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

    def _build_sdk_stream_chunk(
        self,
        *,
        choice: Any,
    ) -> tuple[StreamChunk | None, int, int]:
        delta = choice.delta
        if choice.finish_reason:
            self.last_finish_reason = choice.finish_reason

        tc_delta_list = self._extract_sdk_tool_calls_delta(choice)
        content = delta.content if delta else None
        reasoning = getattr(delta, "reasoning_content", None)
        content_inc = int(isinstance(content, str) and bool(content))
        reasoning_inc = int(isinstance(reasoning, str) and bool(reasoning))

        if not content_inc and not reasoning_inc and not tc_delta_list:
            return None, content_inc, reasoning_inc

        return (
            StreamChunk(
                content_delta=content or "",
                reasoning_delta=reasoning if isinstance(reasoning, str) else None,
                tool_calls_delta=tc_delta_list,
            ),
            content_inc,
            reasoning_inc,
        )

    async def _stream_via_sdk(
        self,
        messages: list[dict],
        model: str,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via openai SDK (preferred path).

        Yields raw tool_calls delta for StreamResponse base-layer accumulation.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )

        sdk_kwargs = self._build_sdk_stream_kwargs(
            messages=messages,
            model=model,
            enable_reasoning=enable_reasoning,
            thinking_budget=thinking_budget,
            extra_kwargs=kwargs,
        )

        logger.info("SDK streaming to %s model=%s", self.base_url, model)
        chunk_count = 0
        reasoning_count = 0
        self.last_usage = None
        self.last_finish_reason: str | None = None  # type: ignore[no-redef]

        try:
            stream = await client.chat.completions.create(**sdk_kwargs)
            async for chunk in stream:
                if chunk.usage:
                    self.last_usage = {
                        USAGE_KEY_PROMPT_TOKENS: chunk.usage.prompt_tokens or 0,
                        USAGE_KEY_COMPLETION_TOKENS: chunk.usage.completion_tokens or 0,
                        USAGE_KEY_TOTAL_TOKENS: chunk.usage.total_tokens or 0,
                    }

                if not chunk.choices:
                    continue

                stream_chunk, content_inc, reasoning_inc = self._build_sdk_stream_chunk(
                    choice=chunk.choices[0]
                )
                chunk_count += content_inc
                reasoning_count += reasoning_inc
                if stream_chunk is not None:
                    yield stream_chunk

            logger.info(
                "SDK stream completed: %d content, %d reasoning chunks, finish=%s, usage=%s",
                chunk_count,
                reasoning_count,
                self.last_finish_reason,
                self.last_usage,
            )
        finally:
            await client.close()

    def _build_httpx_stream_payload(
        self,
        *,
        messages: list[dict],
        model: str,
        enable_reasoning: bool,
        thinking_budget: int | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        extra_body: dict[str, object] = {}
        if enable_reasoning and thinking_budget:
            extra_body["thinking_budget"] = thinking_budget
            logger.info("Reasoning enabled with thinking_budget=%d", thinking_budget)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            payload["extra_body"] = extra_body
        for key, value in extra_kwargs.items():
            if value is not None:
                payload[key] = value
        return payload

    async def _handle_httpx_stream_error_response(
        self,
        *,
        response: Any,
        retry_controller: RetryController,
    ) -> bool:
        error_body = await response.aread()
        error_text = error_body.decode("utf-8", errors="replace")[:2000]
        if response.status_code in retry_controller.policy.status_forcelist:
            decision = retry_controller.on_status_code(
                response.status_code,
                method="POST",
                headers=response.headers,
            )
            if decision.retry:
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
        response.raise_for_status()
        return False

    def _parse_httpx_sse_line(self, line: str) -> tuple[dict[str, Any] | None, bool]:
        if not line or not line.startswith(SSE_DATA_PREFIX):
            return None, False

        data = line[len(SSE_DATA_PREFIX) :].strip()
        if not data:
            return None, False
        if data == SSE_DONE_SIGNAL:
            return None, True

        try:
            event = json.loads(data)
        except Exception:
            logger.debug("Failed to decode SSE chunk: %r", data)
            return None, False

        if not isinstance(event, dict):
            return None, False
        return event, False

    def _build_httpx_stream_chunk(
        self,
        event: dict[str, Any],
    ) -> tuple[StreamChunk | None, int, int]:
        usage_data = event.get("usage")
        if isinstance(usage_data, dict):
            self.last_usage = {
                USAGE_KEY_PROMPT_TOKENS: usage_data.get(USAGE_KEY_PROMPT_TOKENS, 0),
                USAGE_KEY_COMPLETION_TOKENS: usage_data.get(USAGE_KEY_COMPLETION_TOKENS, 0),
                USAGE_KEY_TOTAL_TOKENS: usage_data.get(USAGE_KEY_TOTAL_TOKENS, 0),
            }

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None, 0, 0

        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            self.last_finish_reason = finish_reason

        delta = choice.get("delta")
        delta_payload = delta if isinstance(delta, dict) else {}
        message = choice.get("message")
        message_payload = message if isinstance(message, dict) else {}

        content = delta_payload.get("content")
        if not isinstance(content, str) or not content:
            fallback_content = message_payload.get("content")
            if isinstance(fallback_content, str):
                content = fallback_content

        reasoning = delta_payload.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            fallback_reasoning = message_payload.get("reasoning_content")
            if isinstance(fallback_reasoning, str):
                reasoning = fallback_reasoning

        content_inc = int(isinstance(content, str) and bool(content))
        reasoning_inc = int(isinstance(reasoning, str) and bool(reasoning))
        if not content_inc and not reasoning_inc:
            return None, content_inc, reasoning_inc

        return (
            StreamChunk(
                content_delta=content or "",
                reasoning_delta=reasoning if isinstance(reasoning, str) else None,
            ),
            content_inc,
            reasoning_inc,
        )

    async def _stream_via_httpx(
        self,
        messages: list[dict],
        model: str,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Fallback: stream via httpx raw SSE when openai SDK is not installed."""
        import httpx

        from houyi.infrastructure.net.proxy import detect_proxy

        retry_controller = RetryController(
            RetryPolicy(
                total_retries=DEFAULT_MAX_RETRIES,
                connect_retries=DEFAULT_MAX_RETRIES,
                read_retries=DEFAULT_MAX_RETRIES,
                status_retries=DEFAULT_MAX_RETRIES,
                backoff_base=DEFAULT_RETRY_BASE_DELAY,
                backoff_cap=DEFAULT_RETRY_MAX_DELAY,
            )
        )
        proxy_url = detect_proxy()

        logger.info("httpx fallback streaming to %s model=%s", self.base_url, model)

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = self._build_httpx_stream_payload(
            messages=messages,
            model=model,
            enable_reasoning=enable_reasoning,
            thinking_budget=thinking_budget,
            extra_kwargs=kwargs,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        chunk_count = 0
        reasoning_count = 0
        self.last_usage = None

        while True:
            http_client = httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(300.0, connect=10.0, read=300.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            try:
                async with http_client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400 and await self._handle_httpx_stream_error_response(
                        response=resp,
                        retry_controller=retry_controller,
                    ):
                        continue

                    async for line in resp.aiter_lines():
                        event, done = self._parse_httpx_sse_line(line)
                        if done:
                            break
                        if event is None:
                            continue

                        stream_chunk, content_inc, reasoning_inc = self._build_httpx_stream_chunk(
                            event
                        )
                        chunk_count += content_inc
                        reasoning_count += reasoning_inc
                        if stream_chunk is not None:
                            yield stream_chunk

                logger.info(
                    "httpx stream completed: %d content, %d reasoning chunks, usage=%s",
                    chunk_count,
                    reasoning_count,
                    self.last_usage,
                )
                return
            except httpx.TransportError as exc:
                decision = retry_controller.on_transport_exception(exc, method="POST")
                if decision.retry:
                    logger.warning(
                        "httpx stream transport error: bucket=%s used=%d/%d wait=%.2fs error=%s",
                        decision.bucket,
                        retry_controller.retries_used,
                        retry_controller.policy.total_retries,
                        decision.delay_seconds,
                        exc,
                    )
                    await asyncio.sleep(decision.delay_seconds)
                    continue
                logger.error("httpx fallback transport error: %s", exc, exc_info=True)
                raise
            finally:
                await http_client.aclose()
