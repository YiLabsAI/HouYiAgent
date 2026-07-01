from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import (
    DEFAULT_TEMPERATURE,
    LLMAdapter,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    _normalize_usage,
)
from houyi.adapters.llm.request_models import OpenAICompatRequest
from houyi.adapters.llm.retry import RetryController
from houyi.infrastructure.config.env_config import ENV_PROXY_ENABLED

_ENV_OPENAI_COMPAT_TRANSPORT = "HOUYI_OPENAI_COMPAT_TRANSPORT"


def _normalize_transport_name(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "sdk":
        return "client"
    if normalized in {"client", "httpx"}:
        return normalized
    return ""


def _is_proxy_enabled() -> bool:
    return str(os.getenv(ENV_PROXY_ENABLED, "false") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class OpenAICompatAdapterBase(LLMAdapter):
    client: Any
    api_key: str | None
    base_url: str | None
    default_headers: dict[str, str]
    last_usage: dict[str, Any] | None
    last_finish_reason: str | None

    def _get_default_model(self) -> str:
        return str(getattr(self, "model", ""))

    def _sanitize_request_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._sanitize_messages(messages)

    def _build_reasoning_extra_body(
        self,
        request: OpenAICompatRequest,
    ) -> dict[str, object] | None:
        return None

    def _new_client(self) -> Any | None:
        return None

    def _prepare_request_for_provider(
        self,
        request: OpenAICompatRequest,
    ) -> OpenAICompatRequest:
        return request

    def _copy_request(
        self,
        request: OpenAICompatRequest,
        *,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> OpenAICompatRequest:
        return OpenAICompatRequest(
            model=request.model,
            messages=request.messages if messages is None else messages,
            temperature=request.temperature,
            tools=request.tools if tools is None else tools,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            max_tokens=request.max_tokens,
            tool_choice=request.tool_choice,
            enable_streaming=request.enable_streaming,
            include_stream_usage=request.include_stream_usage,
            enable_thinking=request.enable_thinking,
            thinking_budget=request.thinking_budget,
            transport=request.transport,
            extra_kwargs=dict(request.extra_kwargs),
        )

    def _build_request(
        self,
        *,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int | None,
        enable_streaming: bool,
        kwargs: dict[str, Any],
    ) -> OpenAICompatRequest:
        from houyi.adapters.llm.base import DEFAULT_MAX_TOKENS

        model = kwargs.pop("model", None) or self._get_default_model()
        normalized_messages = self._normalize_messages(messages)
        if getattr(self, "strict_message_string_contract", False):
            normalized_messages = self._sanitize_request_messages(normalized_messages)
        effective_max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        return OpenAICompatRequest.create(
            model=model,
            messages=normalized_messages,
            temperature=temperature,
            tools=tools,
            max_tokens=effective_max_tokens,
            enable_streaming=enable_streaming,
            extra_kwargs=kwargs,
        )

    def _resolve_transport(self, request: OpenAICompatRequest) -> str:
        request_transport = _normalize_transport_name(request.transport)
        if request_transport:
            return request_transport
        route = _normalize_transport_name(os.getenv(_ENV_OPENAI_COMPAT_TRANSPORT, "auto"))
        if route:
            return route
        return "client"

    def _with_extra_body(
        self,
        *,
        request: OpenAICompatRequest,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        extra_body = self._build_reasoning_extra_body(request)
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    def _encode_chat_request(self, request: OpenAICompatRequest) -> dict[str, Any]:
        request = self._prepare_request_for_provider(request)
        return self._with_extra_body(request=request, payload=request.to_sdk_kwargs())

    def _encode_chat_request_for_httpx(self, request: OpenAICompatRequest) -> dict[str, Any]:
        request = self._prepare_request_for_provider(request)
        payload = request.to_httpx_payload()
        payload["stream"] = False
        payload = self._with_extra_body(request=request, payload=payload)
        return {key: value for key, value in payload.items() if value is not None}

    def _encode_stream_request(self, request: OpenAICompatRequest) -> dict[str, Any]:
        request = self._prepare_request_for_provider(request)
        return self._with_extra_body(request=request, payload=request.to_sdk_kwargs())

    def _encode_stream_request_for_httpx(self, request: OpenAICompatRequest) -> dict[str, Any]:
        request = self._prepare_request_for_provider(request)
        payload = request.to_httpx_payload()
        payload["stream"] = True
        payload = self._with_extra_body(request=request, payload=payload)
        return {key: value for key, value in payload.items() if value is not None}

    async def _chat(self, request: OpenAICompatRequest) -> LLMResponse:
        transport = self._resolve_transport(request)
        if transport == "httpx":
            return await self._chat_request_httpx(request)
        return await self._chat_request(request)

    async def _chat_request(self, request: OpenAICompatRequest) -> LLMResponse:
        # Use provider hooks so adapters can swap client lifecycle without forking chat flow.
        client = await self._get_chat_client(request)
        try:
            response = await self._create_chat_response(
                request=request,
                client=client,
            )
            result = LLMResponse.from_openai(response)
            self.last_usage = result.usage
            self.last_finish_reason = result.finish_reason
            return result
        finally:
            await self._close_client(client)

    async def _get_chat_client(self, request: OpenAICompatRequest) -> Any:
        client = self._new_client()
        return self.client if client is None else client

    async def _create_chat_response(
        self,
        *,
        request: OpenAICompatRequest,
        client: Any,
    ) -> Any:
        return await client.chat.completions.create(**self._encode_chat_request(request))

    async def _close_client(self, client: Any) -> None:
        return None

    async def _chat_request_httpx(self, request: OpenAICompatRequest) -> LLMResponse:
        payload = self._normalize_httpx_payload(self._encode_chat_request_for_httpx(request))
        response = await self._execute_chat_httpx(payload, request=request)
        result = LLMResponse.from_raw_dict(
            self._parse_httpx_response(response),
            model_fallback=request.model,
        )
        self.last_usage = result.usage
        self.last_finish_reason = result.finish_reason
        return result

    async def _stream_chat(self, request: OpenAICompatRequest) -> AsyncIterator[StreamChunk]:
        transport = self._resolve_transport(request)
        if transport == "httpx":
            async for chunk in self._stream_request_httpx(request):
                yield chunk
            return
        async for chunk in self._stream_request(request):
            yield chunk

    async def _stream_request(
        self,
        request: OpenAICompatRequest,
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in self._execute_stream_request(self._encode_stream_request(request)):
            yield chunk

    def _build_stream_chunk(
        self,
        *,
        choice: Any,
    ) -> tuple[StreamChunk | None, int, int]:
        finish_reason = getattr(choice, "finish_reason", None)
        if isinstance(finish_reason, str) and finish_reason:
            self.last_finish_reason = finish_reason
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None)
        reasoning = getattr(delta, "reasoning_content", None)
        tool_calls_delta = self._extract_sdk_tool_calls_delta(delta)
        content_text = content if isinstance(content, str) and content else ""
        reasoning_text = reasoning if isinstance(reasoning, str) and reasoning else None
        if not content_text and not reasoning_text and not tool_calls_delta:
            return None, 0, 0
        return (
            StreamChunk(
                content_delta=content_text,
                reasoning_delta=reasoning_text,
                tool_calls_delta=tool_calls_delta,
            ),
            int(bool(content_text)),
            int(bool(reasoning_text)),
        )

    async def _execute_stream_request(
        self,
        params: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        chunk_count = 0
        reasoning_count = 0
        self.last_usage = None
        self.last_finish_reason = None
        # Stream setup/teardown is centralized here so providers only override narrow hooks.
        client = await self._get_stream_client(params)
        try:
            stream = await self._create_stream(params, client=client)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    self.last_usage = _normalize_usage(chunk.usage)
                if not chunk.choices:
                    continue
                stream_chunk, content_inc, reasoning_inc = self._build_stream_chunk(
                    choice=chunk.choices[0]
                )
                chunk_count += content_inc
                reasoning_count += reasoning_inc
                if stream_chunk is not None:
                    yield stream_chunk
            self._log_stream(
                params=params,
                chunk_count=chunk_count,
                reasoning_count=reasoning_count,
            )
        finally:
            await self._close_client(client)

    async def _get_stream_client(self, params: dict[str, Any]) -> Any:
        client = self._new_client()
        return self.client if client is None else client

    async def _create_stream(self, params: dict[str, Any], *, client: Any) -> Any:
        return await client.chat.completions.create(**params)

    def _log_stream(
        self,
        *,
        params: dict[str, Any],
        chunk_count: int,
        reasoning_count: int,
    ) -> None:
        return None

    @staticmethod
    def _extract_sdk_tool_calls_delta(delta: Any) -> list[dict[str, Any]] | None:
        raw_tool_calls = getattr(delta, "tool_calls", None)
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            return None
        tc_delta_list: list[dict[str, Any]] = []
        for index, tool_call in enumerate(raw_tool_calls):
            tc_delta: dict[str, Any] = {"index": getattr(tool_call, "index", index)}
            tool_call_id = getattr(tool_call, "id", None)
            if tool_call_id:
                tc_delta["id"] = tool_call_id
            tool_call_type = getattr(tool_call, "type", None)
            if tool_call_type:
                tc_delta["type"] = tool_call_type
            function_payload = getattr(tool_call, "function", None)
            if function_payload is not None:
                function_delta: dict[str, Any] = {}
                function_name = getattr(function_payload, "name", None)
                if function_name:
                    function_delta["name"] = function_name
                function_arguments = getattr(function_payload, "arguments", None)
                if function_arguments is not None:
                    function_delta["arguments"] = function_arguments
                if function_delta:
                    tc_delta["function"] = function_delta
            tc_delta_list.append(tc_delta)
        return tc_delta_list or None

    @staticmethod
    def _extract_httpx_tool_calls_delta(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return None
        tc_delta_list: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            tc_delta: dict[str, Any] = {"index": tool_call.get("index", index)}
            if tool_call.get("id"):
                tc_delta["id"] = tool_call["id"]
            if tool_call.get("type"):
                tc_delta["type"] = tool_call["type"]
            function_payload = tool_call.get("function")
            if isinstance(function_payload, dict):
                function_delta: dict[str, Any] = {}
                if function_payload.get("name"):
                    function_delta["name"] = function_payload["name"]
                if function_payload.get("arguments") is not None:
                    function_delta["arguments"] = function_payload["arguments"]
                if function_delta:
                    tc_delta["function"] = function_delta
            tc_delta_list.append(tc_delta)
        return tc_delta_list or None

    def _parse_httpx_sse_event(self, line: str) -> dict[str, Any] | None:
        if not line.startswith("data: "):
            return None
        data = line[6:].strip()
        if not data or data == "[DONE]":
            return None
        event = json.loads(data)
        return event if isinstance(event, dict) else None

    def _build_stream_chunk_from_httpx_event(
        self,
        event: dict[str, Any],
    ) -> StreamChunk | None:
        usage = event.get("usage")
        if isinstance(usage, dict):
            self.last_usage = _normalize_usage(usage)
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            self.last_finish_reason = finish_reason
        delta = choice.get("delta")
        delta_payload = delta if isinstance(delta, dict) else {}
        message = choice.get("message")
        message_payload = message if isinstance(message, dict) else {}
        content = delta_payload.get("content")
        if (not isinstance(content, str) or not content) and isinstance(
            message_payload.get("content"), str
        ):
            content = message_payload.get("content")
        reasoning = delta_payload.get("reasoning_content")
        if (not isinstance(reasoning, str) or not reasoning) and isinstance(
            message_payload.get("reasoning_content"), str
        ):
            reasoning = message_payload.get("reasoning_content")
        tool_calls_delta = self._extract_httpx_tool_calls_delta(delta_payload)
        if tool_calls_delta is None:
            tool_calls_delta = self._extract_httpx_tool_calls_delta(message_payload)
        content_text = content if isinstance(content, str) and content else ""
        reasoning_text = reasoning if isinstance(reasoning, str) and reasoning else None
        if content_text or reasoning_text or tool_calls_delta:
            return StreamChunk(
                content_delta=content_text,
                reasoning_delta=reasoning_text,
                tool_calls_delta=tool_calls_delta,
            )
        return None

    def _normalize_httpx_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def _get_httpx_endpoint(self) -> str:
        base_url = str(getattr(self, "base_url", "") or "")
        return base_url.rstrip("/") + "/chat/completions"

    def _get_httpx_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **getattr(self, "default_headers", {}),
        }

    def _get_httpx_proxy(self) -> str | None:
        return None

    def _get_httpx_retry_controller(self) -> RetryController | None:
        return None

    def _parse_httpx_response(self, response: Any) -> dict[str, Any]:
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _handle_httpx_transport_error(
        self,
        *,
        exc: Exception,
        retry_controller: RetryController,
        request: OpenAICompatRequest | None = None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[bool, float]:
        _ = request
        _ = payload
        decision = retry_controller.on_transport_exception(exc, method="POST")
        return decision.retry, decision.delay_seconds

    async def _handle_httpx_status(
        self,
        *,
        response: Any,
        retry_controller: RetryController,
    ) -> tuple[bool, Exception | None]:
        status_code = getattr(response, "status_code", 200)
        if status_code not in retry_controller.policy.status_forcelist:
            return False, None
        decision = retry_controller.on_status_code(
            status_code,
            method="POST",
            headers=getattr(response, "headers", {}),
        )
        if not decision.retry:
            return False, None
        return True, RuntimeError(f"HTTP {status_code}")

    async def _close_httpx_response(self, response: Any) -> None:
        close = getattr(response, "aclose", None)
        if callable(close):
            await close()

    async def _send_httpx_request(
        self, http_client: Any, *, url: str, payload: dict[str, Any]
    ) -> Any:
        return await http_client.post(url, json=payload, headers=self._get_httpx_headers())

    def _get_httpx_chat_timeout(self) -> Any:
        import httpx

        # 180s lets verbose models finish long structured answers without
        # being cut mid-generation; 60s truncated valid slow responses,
        # producing empty output. connect stays short to fail fast on
        # network issues rather than queueing.
        return httpx.Timeout(180.0, connect=10.0)

    async def _execute_chat_httpx(
        self,
        payload: dict[str, Any],
        *,
        request: OpenAICompatRequest | None = None,
    ) -> Any:
        import asyncio

        import httpx

        retry_controller = self._get_httpx_retry_controller()
        proxy_url = self._get_httpx_proxy()
        url = self._get_httpx_endpoint()
        last_error: Exception | None = None

        while True:
            response: Any | None = None
            try:
                # Retry state lives outside the HTTP client so each attempt can use a fresh connection.
                async with httpx.AsyncClient(
                    proxy=proxy_url,
                    timeout=self._get_httpx_chat_timeout(),
                ) as client:
                    response = await self._send_httpx_request(client, url=url, payload=payload)
            except httpx.TransportError as exc:
                if retry_controller is None:
                    raise
                should_retry, delay_seconds = self._handle_httpx_transport_error(
                    exc=exc,
                    retry_controller=retry_controller,
                    request=request,
                    payload=payload,
                )
                if should_retry:
                    last_error = exc
                    await asyncio.sleep(delay_seconds)
                    continue
                raise

            if retry_controller is not None:
                should_retry, last_error = await self._handle_httpx_status(
                    response=response,
                    retry_controller=retry_controller,
                )
                if should_retry:
                    await self._close_httpx_response(response)
                    if last_error is not None:
                        await asyncio.sleep(0)
                    continue

            if response is None:
                raise last_error or RuntimeError("HTTP request failed")
            return response

    def _parse_httpx_sse_line(self, line: str) -> tuple[dict[str, Any] | None, bool]:
        if not line or not line.startswith("data: "):
            return None, False
        data = line[6:].strip()
        if not data:
            return None, False
        if data == "[DONE]":
            return None, True
        try:
            event = json.loads(data)
        except Exception:
            return None, False
        return (event if isinstance(event, dict) else None), False

    def _build_httpx_stream_chunk(
        self,
        event: dict[str, Any],
    ) -> tuple[StreamChunk | None, int, int]:
        chunk = self._build_stream_chunk_from_httpx_event(event)
        if chunk is None:
            return None, 0, 0
        content_inc = int(bool(chunk.content_delta))
        reasoning_inc = int(bool(chunk.reasoning_delta))
        return chunk, content_inc, reasoning_inc

    def _stream_retry(self) -> RetryController | None:
        return None

    def _build_httpx_stream_url(self) -> str:
        base_url = str(getattr(self, "base_url", "") or "")
        return base_url.rstrip("/") + "/chat/completions"

    def _build_httpx_stream_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **getattr(self, "default_headers", {}),
        }

    def _get_httpx_stream_proxy_url(self) -> str | None:
        if not _is_proxy_enabled():
            return None
        from houyi.infrastructure.net.proxy import detect_proxy

        return detect_proxy()

    def _stream_transport(
        self,
        *,
        exc: Exception,
        retry_controller: RetryController,
    ) -> tuple[bool, float]:
        decision = retry_controller.on_transport_exception(exc, method="POST")
        return decision.retry, decision.delay_seconds

    async def _stream_status(
        self,
        *,
        response: Any,
        retry_controller: RetryController,
    ) -> bool:
        response.raise_for_status()
        return False

    @staticmethod
    async def _read_httpx_error_text(response: Any) -> str:
        error_body = await response.aread()
        return error_body.decode("utf-8", errors="replace")[:2000]

    def _retry_httpx_stream_status(
        self,
        *,
        response: Any,
        retry_controller: RetryController,
    ) -> tuple[bool, Any | None]:
        status_code = getattr(response, "status_code", 200)
        if status_code not in retry_controller.policy.status_forcelist:
            return False, None
        decision = retry_controller.on_status_code(
            status_code,
            method="POST",
            headers=getattr(response, "headers", {}),
        )
        return decision.retry, decision if decision.retry else None

    async def _stream_httpx_response_chunks(
        self,
        response: Any,
    ) -> AsyncIterator[StreamChunk]:
        async for line in response.aiter_lines():
            event, done = self._parse_httpx_sse_line(line)
            if done:
                break
            if event is None:
                continue
            stream_chunk, _, _ = self._build_httpx_stream_chunk(event)
            if stream_chunk is not None:
                yield stream_chunk

    async def _close_httpx_client(self, http_client: Any) -> None:
        close = getattr(http_client, "aclose", None)
        if callable(close):
            await close()

    async def _should_retry_httpx_error_response(
        self,
        *,
        response: Any,
        retry_controller: RetryController | None,
    ) -> bool:
        status_code = getattr(response, "status_code", 200)
        if status_code < 400:
            return False
        if retry_controller is None:
            response.raise_for_status()
            return False
        if await self._stream_status(
            response=response,
            retry_controller=retry_controller,
        ):
            return True
        response.raise_for_status()
        return False

    async def _execute_stream_request_httpx(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        import asyncio

        import httpx

        retry_controller = self._stream_retry()
        proxy_url = self._get_httpx_stream_proxy_url()
        url = self._build_httpx_stream_url()
        headers = self._build_httpx_stream_headers()
        self.last_usage = None

        while True:
            http_client = httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(300.0, connect=10.0, read=300.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            try:
                # Keep retry and SSE parsing in one place so adapters only customize event semantics.
                async with http_client.stream("POST", url, json=payload, headers=headers) as resp:
                    if await self._should_retry_httpx_error_response(
                        response=resp,
                        retry_controller=retry_controller,
                    ):
                        continue
                    async for stream_chunk in self._stream_httpx_response_chunks(resp):
                        yield stream_chunk
                return
            except httpx.TransportError as exc:
                if retry_controller is None:
                    raise
                should_retry, delay_seconds = self._stream_transport(
                    exc=exc,
                    retry_controller=retry_controller,
                )
                if not should_retry:
                    raise
                await asyncio.sleep(delay_seconds)
            finally:
                await self._close_httpx_client(http_client)

    async def _stream_request_httpx(
        self,
        request: OpenAICompatRequest,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._normalize_httpx_payload(self._encode_stream_request_for_httpx(request))
        async for chunk in self._execute_stream_request_httpx(payload):
            yield chunk

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        request = self._build_request(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_streaming=True,
            kwargs=dict(kwargs),
        )
        async for chunk in self._stream_chat(request):
            yield chunk
