"""SiliconFlow / OpenAI-compatible LLM adapter.

Supports any OpenAI-compatible endpoint: SiliconFlow, DeepSeek, vLLM, Ollama, etc.

Strategy: ``openai`` SDK first, ``httpx`` raw SSE as fallback.
The openai SDK provides automatic retries, proper error types, streaming support,
and token usage tracking.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from houyi.llm.base import DEFAULT_TEMPERATURE, LLMAdapter, LLMMessage, LLMResponse
from houyi.llm.retry import (
    DEFAULT_MAX_RETRIES,
    SSE_DONE_SIGNAL,
    USAGE_KEY_COMPLETION_TOKENS,
    USAGE_KEY_PROMPT_TOKENS,
    USAGE_KEY_TOTAL_TOKENS,
    exponential_backoff,
    is_retryable_status,
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
    ):
        from houyi.config.env_config import EnvConfig

        _env = EnvConfig.get()

        self.api_key = api_key or _env.siliconflow_api_key
        self.base_url = base_url or _env.siliconflow_base_url
        self.default_model = default_model or _env.deepseek_model
        self.model = self.default_model  # uniform interface for callers
        self.last_usage: dict[str, int] | None = None
        self.last_tool_calls: list[dict] = []
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

        params: dict[str, object] = {
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
        import urllib.request

        import httpx

        url = self.base_url.rstrip("/") + "/chat/completions"
        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            body["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        proxies = urllib.request.getproxies()
        proxy_url = proxies.get("https") or proxies.get("http")

        max_retries = DEFAULT_MAX_RETRIES
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(60.0, connect=10.0),
            ) as client:
                resp = await client.post(url, json=body, headers=headers)

                if is_retryable_status(resp.status_code) and attempt < max_retries:
                    logger.warning(
                        "SiliconFlow chat HTTP %d (retry %d/%d)",
                        resp.status_code,
                        attempt + 1,
                        max_retries,
                    )
                    last_error = RuntimeError(
                        f"SiliconFlow HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                    await exponential_backoff(attempt)
                    continue

                if resp.status_code >= 400:
                    error_text = resp.text[:1000]
                    raise RuntimeError(f"SiliconFlow HTTP {resp.status_code}: {error_text}")

                data = resp.json()
                break
        else:
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
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream chat completion with full message list.

        Yields ``(content_delta, reasoning_delta)`` tuples.
        ``reasoning_delta`` is populated for DeepSeek models that support reasoning.
        """
        model = kwargs.pop("model", None) or self.default_model
        enable_reasoning = kwargs.pop("enable_reasoning", False)
        thinking_budget = kwargs.pop("thinking_budget", None)
        normalized = self._normalize_messages(messages)

        if not self.api_key:
            logger.info("Using mock streaming (no API key)")
            last_content = ""
            for msg in reversed(normalized):
                if msg.get("role") == "user":
                    last_content = msg.get("content", "")
                    break
            words = f"Mock response from {model}: {last_content[:50]}...".split()
            for word in words:
                yield (word + " ", None)
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

    async def _stream_via_sdk(
        self,
        messages: list[dict],
        model: str,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream via openai SDK (preferred path).

        Accumulates tool_calls from delta chunks (OpenAI streaming protocol).
        After iteration, ``self.last_tool_calls`` contains accumulated tool calls.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )

        extra_body: dict[str, object] = {}
        if enable_reasoning and thinking_budget:
            extra_body["thinking_budget"] = thinking_budget
            logger.info("Reasoning enabled with thinking_budget=%d", thinking_budget)

        tools = kwargs.pop("tools", None)
        sdk_kwargs: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            sdk_kwargs["tools"] = tools
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            sdk_kwargs["tool_choice"] = tool_choice
        if extra_body:
            sdk_kwargs["extra_body"] = extra_body
        for k, v in kwargs.items():
            if v is not None and k not in sdk_kwargs:
                sdk_kwargs[k] = v

        logger.info("SDK streaming to %s model=%s", self.base_url, model)
        chunk_count = 0
        reasoning_count = 0
        self.last_usage = None
        self.last_tool_calls: list[dict] = []  # type: ignore[no-redef]
        self.last_finish_reason: str | None = None  # type: ignore[no-redef]

        # Accumulator for streaming tool_calls (OpenAI delta protocol)
        tool_call_accum: dict[int, dict] = {}

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

                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    self.last_finish_reason = choice.finish_reason

                # Accumulate tool_calls from delta (OpenAI streaming protocol)
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_call_accum[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments

                content = delta.content if delta else None
                reasoning = getattr(delta, "reasoning_content", None)

                if isinstance(content, str) and content:
                    chunk_count += 1
                if isinstance(reasoning, str) and reasoning:
                    reasoning_count += 1

                if (isinstance(content, str) and content) or (
                    isinstance(reasoning, str) and reasoning
                ):
                    yield (content or "", reasoning if isinstance(reasoning, str) else None)

            # Finalize accumulated tool calls
            if tool_call_accum:
                for idx in sorted(tool_call_accum):
                    tc = tool_call_accum[idx]
                    args_str = tc["function"]["arguments"]
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        tc["function"]["arguments"] = json.loads(args_str)
                    self.last_tool_calls.append(tc)

            logger.info(
                "SDK stream completed: %d content, %d reasoning chunks, "
                "%d tool_calls, finish=%s, usage=%s",
                chunk_count,
                reasoning_count,
                len(self.last_tool_calls),
                self.last_finish_reason,
                self.last_usage,
            )
        finally:
            await client.close()

    async def _stream_via_httpx(
        self,
        messages: list[dict],
        model: str,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Fallback: stream via httpx raw SSE when openai SDK is not installed."""
        import httpx

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0, read=300.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        extra_body: dict[str, object] = {}
        if enable_reasoning and thinking_budget:
            extra_body["thinking_budget"] = thinking_budget
            logger.info("Reasoning enabled with thinking_budget=%d", thinking_budget)

        logger.info("httpx fallback streaming to %s model=%s", self.base_url, model)

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            payload["extra_body"] = extra_body
        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        chunk_count = 0
        reasoning_count = 0
        self.last_usage = None

        try:
            async with http_client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    logger.error(
                        "httpx API error %d: %s",
                        resp.status_code,
                        error_body.decode("utf-8", errors="replace")[:2000],
                    )
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data:
                        continue
                    if data == SSE_DONE_SIGNAL:
                        break

                    try:
                        event = json.loads(data)
                    except Exception:
                        logger.debug("Failed to decode SSE chunk: %r", data)
                        continue

                    if isinstance(event, dict) and "usage" in event:
                        usage_data = event["usage"]
                        if isinstance(usage_data, dict):
                            self.last_usage = {
                                USAGE_KEY_PROMPT_TOKENS: usage_data.get(USAGE_KEY_PROMPT_TOKENS, 0),
                                USAGE_KEY_COMPLETION_TOKENS: usage_data.get(
                                    USAGE_KEY_COMPLETION_TOKENS, 0
                                ),
                                USAGE_KEY_TOTAL_TOKENS: usage_data.get(USAGE_KEY_TOTAL_TOKENS, 0),
                            }

                    choices = event.get("choices") if isinstance(event, dict) else None
                    if not isinstance(choices, list) or not choices:
                        continue

                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    if not isinstance(delta, dict):
                        continue

                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")

                    if isinstance(content, str) and content:
                        chunk_count += 1
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_count += 1

                    if (isinstance(content, str) and content) or (
                        isinstance(reasoning, str) and reasoning
                    ):
                        yield (content or "", reasoning if isinstance(reasoning, str) else None)

            logger.info(
                "httpx stream completed: %d content, %d reasoning chunks, usage=%s",
                chunk_count,
                reasoning_count,
                self.last_usage,
            )
        except Exception as e:
            logger.error("httpx fallback API error: %s", e, exc_info=True)
            raise
        finally:
            await http_client.aclose()
