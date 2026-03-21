"""Vertex AI adapter using httpx + JWT (zero SDK dependency).

Auth: reads ``GOOGLE_APPLICATION_CREDENTIALS`` (service account JSON),
signs a JWT with ``openssl`` subprocess, exchanges for ``access_token``,
then calls the Vertex AI OpenAI-compatible ``chat/completions`` endpoint.

This is the **fallback** adapter used when the ``google-genai`` SDK is not
installed.  The SDK-based ``GoogleVertexGeminiAdapter`` (in
``vertex_gemini_adapter.py``) is preferred when available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import (
    DEFAULT_TEMPERATURE,
    LLMResponse,
    StreamChunk,
)
from houyi.adapters.llm.openai_compat_base import OpenAICompatAdapterBase
from houyi.adapters.llm.request_models import OpenAICompatRequest
from houyi.adapters.llm.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    SSE_DATA_PREFIX,
    SSE_DONE_SIGNAL,
    USAGE_KEY_COMPLETION_TOKENS,
    USAGE_KEY_PROMPT_TOKENS,
    USAGE_KEY_TOTAL_TOKENS,
    VERTEX_MAX_OUTPUT_TOKENS,
    RetryController,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


class _VertexAuthError(RuntimeError):
    pass


class VertexAIAdapter(OpenAICompatAdapterBase):
    """Adapter for Google Gemini via Vertex AI OpenAI-compatible endpoint.

    Auth: reads GOOGLE_APPLICATION_CREDENTIALS (service account JSON),
    signs a JWT with openssl subprocess, exchanges for access_token,
    then calls the Vertex AI OpenAI-compatible chat/completions endpoint.

    Zero external dependencies — uses only stdlib + httpx.
    """

    def __init__(self) -> None:
        from houyi.infrastructure.config.env_config import EnvConfig

        _env = EnvConfig.get()

        self.default_model = _env.gemini_model
        self.model = self.default_model  # uniform interface for callers
        self.client = None
        self.api_key = None
        self.base_url = None
        self.default_headers: dict[str, str] = {}
        self.strict_message_string_contract = False
        self.last_usage: dict[str, int] | None = None
        self.last_finish_reason: str | None = None
        self._access_token: str | None = None
        self._token_expiry: float = 0
        self._sa: dict | None = None
        self._request_access_token: str | None = None

        sa_path = _env.google_credentials_path
        if sa_path and os.path.isfile(sa_path):
            try:
                with open(sa_path) as f:
                    self._sa = json.load(f)
                logger.info(
                    "Vertex AI: loaded service account %s (project=%s)",
                    self._sa.get("client_email", "?"),
                    self._sa.get("project_id", "?"),
                )
            except Exception as e:
                logger.error("Failed to load service account from %s: %s", sa_path, e)

        self.project_id = (
            self._sa.get("project_id") if self._sa else None
        ) or _env.google_project_id
        self.location = _env.google_location

        if not self.project_id:
            logger.warning("Vertex AI: no project_id found, will use mock responses")
        if not self._sa:
            logger.warning(
                "Vertex AI: GOOGLE_APPLICATION_CREDENTIALS not set, will use mock responses"
            )

    # ── URL construction ──────────────────────────────────────────

    def _get_openai_base_url(self) -> str:
        """Build the Vertex AI OpenAI-compatible base URL."""
        if self.location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{self.location}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1beta1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"endpoints/openapi"
        )

    # ── JWT signing ───────────────────────────────────────────────

    def _sign_jwt_with_openssl(self) -> str:
        """Create a signed JWT using openssl subprocess (no python crypto deps)."""
        import base64
        import subprocess
        import tempfile
        import time as _time

        assert self._sa is not None
        sa = self._sa
        now = int(_time.time())

        header_b64 = base64.urlsafe_b64encode(
            json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
        ).rstrip(b"=")
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(
                {
                    "iss": sa["client_email"],
                    "scope": "https://www.googleapis.com/auth/cloud-platform",
                    "aud": sa["token_uri"],
                    "iat": now,
                    "exp": now + 3600,
                }
            ).encode()
        ).rstrip(b"=")

        signing_input = header_b64 + b"." + payload_b64

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as tmp:
            pem_path = tmp.name
            tmp.write(sa["private_key"])
        try:
            proc = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", pem_path],
                input=signing_input,
                capture_output=True,
                timeout=5,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"openssl signing failed: {proc.stderr.decode()}")
        finally:
            os.unlink(pem_path)

        sig_b64 = base64.urlsafe_b64encode(proc.stdout).rstrip(b"=")
        return (header_b64 + b"." + payload_b64 + b"." + sig_b64).decode()

    # ── Token exchange ────────────────────────────────────────────

    async def _get_access_token(self) -> str | None:
        """Get a valid access token via JWT exchange.

        Uses httpx with system proxy detection so that the token exchange
        works even behind a proxy.
        """
        import time as _time

        import httpx

        from houyi.infrastructure.net.proxy import detect_proxy

        now = _time.time()
        if self._access_token and now < self._token_expiry - 60:
            return self._access_token

        if not self._sa:
            return None

        try:
            jwt_token = self._sign_jwt_with_openssl()

            form_data = {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt_token,
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

            while True:
                try:
                    async with httpx.AsyncClient(
                        proxy=proxy_url,
                        timeout=httpx.Timeout(20.0, connect=10.0),
                    ) as client:
                        resp = await client.post(
                            self._sa["token_uri"],
                            data=form_data,
                            headers={"Content-Type": "application/x-www-form-urlencoded"},
                        )
                except httpx.TransportError as exc:
                    decision = retry_controller.on_transport_exception(exc, method="POST")
                    if decision.retry:
                        logger.warning(
                            "Vertex token transport error: bucket=%s retry=%d/%d wait=%.2fs error=%s",
                            decision.bucket,
                            retry_controller.retries_used,
                            retry_controller.policy.total_retries,
                            decision.delay_seconds,
                            exc,
                        )
                        await asyncio.sleep(decision.delay_seconds)
                        continue
                    raise

                if resp.status_code == 401:
                    self._access_token = None
                    self._token_expiry = 0

                if resp.status_code in retry_controller.policy.status_forcelist:
                    decision = retry_controller.on_status_code(
                        resp.status_code,
                        method="POST",
                        headers=getattr(resp, "headers", {}),
                    )
                    if decision.retry:
                        logger.warning(
                            "Vertex token HTTP %d retry=%d/%d wait=%.2fs",
                            resp.status_code,
                            retry_controller.retries_used,
                            retry_controller.policy.total_retries,
                            decision.delay_seconds,
                        )
                        await asyncio.sleep(decision.delay_seconds)
                        continue

                break

            resp.raise_for_status()
            token_data = resp.json()

            self._access_token = token_data["access_token"]
            self._token_expiry = now + token_data.get("expires_in", 3600)
            logger.info(
                "Vertex AI access token refreshed (expires in %ds)",
                int(self._token_expiry - now),
            )
            return self._access_token
        except Exception as e:
            logger.error("Failed to get Vertex AI access token: %s", e)
            return None

    def _resolve_transport(self, request: Any) -> str:
        return "httpx"

    def _new_retry_controller(self, *, stream: bool = False) -> RetryController:
        policy_kwargs: dict[str, Any] = {
            "total_retries": DEFAULT_MAX_RETRIES,
            "status_retries": DEFAULT_MAX_RETRIES,
            "backoff_base": DEFAULT_RETRY_BASE_DELAY,
            "backoff_cap": DEFAULT_RETRY_MAX_DELAY,
        }
        if stream:
            policy_kwargs.update(
                {
                    "connect_retries": DEFAULT_MAX_RETRIES,
                    "read_retries": DEFAULT_MAX_RETRIES,
                }
            )
        return RetryController(RetryPolicy(**policy_kwargs))

    @staticmethod
    def _clamp_max_tokens(max_tokens: int | None) -> int | None:
        if max_tokens is None:
            return None
        if max_tokens > VERTEX_MAX_OUTPUT_TOKENS:
            logger.warning(
                "Clamping max_tokens from %d to %d for Vertex AI",
                max_tokens,
                VERTEX_MAX_OUTPUT_TOKENS,
            )
            return VERTEX_MAX_OUTPUT_TOKENS
        return max_tokens

    @classmethod
    def _build_vertex_chat_payload(
        cls,
        *,
        model: str,
        normalized_messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": f"google/{model}",
            "messages": normalized_messages,
            "stream": False,
            "temperature": temperature,
        }
        clamped_max_tokens = cls._clamp_max_tokens(max_tokens)
        if clamped_max_tokens is not None:
            body["max_tokens"] = clamped_max_tokens
        if tools:
            body["tools"] = tools
        tool_choice = extra_kwargs.pop("tool_choice", None)
        if tool_choice:
            body["tool_choice"] = tool_choice
        return body

    @staticmethod
    def _build_chat_body(
        *,
        model: str,
        normalized_messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        tools: list[dict] | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return VertexAIAdapter._build_vertex_chat_payload(
            model=model,
            normalized_messages=normalized_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            extra_kwargs=dict(extra_kwargs),
        )

    @classmethod
    def _build_vertex_stream_payload(
        cls,
        *,
        model: str,
        normalized_messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": f"google/{model}",
            "messages": normalized_messages,
            "stream": True,
        }
        if temperature != DEFAULT_TEMPERATURE:
            body["temperature"] = temperature
        clamped_max_tokens = cls._clamp_max_tokens(max_tokens)
        if clamped_max_tokens is not None:
            body["max_tokens"] = clamped_max_tokens
        if "top_p" in extra_kwargs and extra_kwargs["top_p"] is not None:
            body["top_p"] = extra_kwargs["top_p"]
        if "stop" in extra_kwargs and extra_kwargs["stop"] is not None:
            body["stop"] = extra_kwargs["stop"]
        if extra_kwargs.get("enable_reasoning", False) or extra_kwargs.get(
            "enable_thinking", False
        ):
            body["reasoning_effort"] = "high"
            logger.info("Gemini reasoning enabled: reasoning_effort=high")
        return body

    @staticmethod
    def _build_stream_body(
        *,
        model: str,
        normalized_messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return VertexAIAdapter._build_vertex_stream_payload(
            model=model,
            normalized_messages=normalized_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=dict(extra_kwargs),
        )

    def _reset_token_if_unauthorized(self, status_code: int) -> None:
        if status_code == 401:
            self._access_token = None
            self._token_expiry = 0
            self._request_access_token = None

    async def _retry_or_raise_transport(
        self,
        *,
        retry_controller: RetryController,
        exc: Exception,
        label: str,
    ) -> bool:
        decision = retry_controller.on_transport_exception(exc, method="POST")
        if not decision.retry:
            return False
        logger.warning(
            "%s transport error: bucket=%s used=%d/%d wait=%.2fs error=%s",
            label,
            decision.bucket,
            retry_controller.retries_used,
            retry_controller.policy.total_retries,
            decision.delay_seconds,
            exc,
        )
        await asyncio.sleep(decision.delay_seconds)
        return True

    async def _retry_or_raise_http_status(
        self,
        *,
        retry_controller: RetryController,
        status_code: int,
        headers: Any,
        label: str,
    ) -> bool:
        if status_code not in retry_controller.policy.status_forcelist:
            return False
        decision = retry_controller.on_status_code(
            status_code,
            method="POST",
            headers=headers,
        )
        if not decision.retry:
            return False
        logger.warning(
            "%s HTTP %d retry=%d/%d wait=%.2fs",
            label,
            status_code,
            retry_controller.retries_used,
            retry_controller.policy.total_retries,
            decision.delay_seconds,
        )
        await asyncio.sleep(decision.delay_seconds)
        return True

    def _update_stream_usage_from_event(self, event: dict[str, Any]) -> None:
        usage_data = event.get("usage")
        if not isinstance(usage_data, dict):
            return
        self.last_usage = {
            USAGE_KEY_PROMPT_TOKENS: usage_data.get(USAGE_KEY_PROMPT_TOKENS, 0),
            USAGE_KEY_COMPLETION_TOKENS: usage_data.get(USAGE_KEY_COMPLETION_TOKENS, 0),
            USAGE_KEY_TOTAL_TOKENS: usage_data.get(USAGE_KEY_TOTAL_TOKENS, 0),
        }

    @staticmethod
    def _parse_sse_event(line: str) -> tuple[dict[str, Any] | None, bool]:
        if not line.startswith(SSE_DATA_PREFIX):
            return None, False
        payload = line[len(SSE_DATA_PREFIX) :].strip()
        if payload == SSE_DONE_SIGNAL:
            return None, True
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return None, False
        if not isinstance(event, dict):
            return None, False
        return event, False

    @staticmethod
    def _extract_tool_calls_delta(delta: dict[str, Any]) -> list[dict[str, Any]] | None:
        tool_calls = delta.get("tool_calls")
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
                tc_delta["function"] = {}
                if function_payload.get("name"):
                    tc_delta["function"]["name"] = function_payload["name"]
                if function_payload.get("arguments") is not None:
                    tc_delta["function"]["arguments"] = function_payload["arguments"]
            tc_delta_list.append(tc_delta)
        return tc_delta_list or None

    @staticmethod
    def _extract_stream_chunk(event: dict[str, Any]) -> StreamChunk | None:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        delta = choices[0].get("delta", {})
        if not isinstance(delta, dict):
            return None
        content = delta.get("content", "")
        reasoning = delta.get("reasoning_content", "")
        tool_calls_delta = VertexAIAdapter._extract_tool_calls_delta(delta)
        if not content and not reasoning and not tool_calls_delta:
            return None
        return StreamChunk(
            content_delta=content or "",
            reasoning_delta=reasoning if reasoning else None,
            tool_calls_delta=tool_calls_delta,
        )

    def _encode_chat_request_for_httpx(self, request: Any) -> dict[str, Any]:
        return self._build_vertex_chat_payload(
            model=request.model,
            normalized_messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            extra_kwargs=dict(request.extra_kwargs, tool_choice=request.tool_choice),
        )

    def _encode_stream_request_for_httpx(self, request: Any) -> dict[str, Any]:
        return self._build_vertex_stream_payload(
            model=request.model,
            normalized_messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_kwargs={
                **dict(request.extra_kwargs),
                "top_p": request.top_p,
                "enable_thinking": request.enable_thinking,
            },
        )

    def _get_httpx_endpoint(self) -> str:
        return f"{self._get_openai_base_url()}/chat/completions"

    def _build_httpx_stream_url(self) -> str:
        return f"{self._get_openai_base_url()}/chat/completions"

    def _get_httpx_headers(self) -> dict[str, str]:
        token = self._request_access_token or ""
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _get_httpx_proxy(self) -> str | None:
        from houyi.infrastructure.net.proxy import detect_proxy

        return detect_proxy()

    def _get_httpx_retry_controller(self) -> RetryController:
        return self._new_retry_controller()

    def _get_httpx_stream_proxy_url(self) -> str | None:
        from houyi.infrastructure.net.proxy import detect_proxy

        return detect_proxy()

    async def _send_httpx_request(
        self, http_client: Any, *, url: str, payload: dict[str, Any]
    ) -> Any:
        access_token = await self._get_access_token()
        if not access_token:
            raise _VertexAuthError("Failed to authenticate with Vertex AI")
        self._request_access_token = access_token
        return await http_client.post(url, json=payload, headers=self._get_httpx_headers())

    async def _chat_request_httpx(self, request: Any) -> LLMResponse:
        if not self.project_id or not self._sa:
            return LLMResponse(
                content="Mock response (no project ID or service account)",
                tool_calls=[],
                finish_reason="stop",
                usage={},
                model=request.model,
            )
        try:
            return await super()._chat_request_httpx(request)
        except _VertexAuthError:
            return LLMResponse(
                content="",
                tool_calls=[],
                finish_reason="error",
                usage={},
                model=request.model,
                metadata={"error": "Failed to authenticate with Vertex AI"},
            )
        finally:
            self._request_access_token = None

    def _parse_httpx_response(self, response: Any) -> dict[str, Any]:
        status_code = int(getattr(response, "status_code", 200))
        self._reset_token_if_unauthorized(status_code)
        if status_code >= 400:
            raise self._wrap_httpx_status_error(response)
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _wrap_httpx_status_error(self, response: Any) -> RuntimeError:
        status_code = int(getattr(response, "status_code", 500))
        body_text = str(getattr(response, "text", "") or "")[:2000]
        return RuntimeError(f"Vertex AI HTTP {status_code}: {body_text}")

    def _handle_httpx_transport_error(
        self,
        *,
        exc: Exception,
        retry_controller: RetryController,
        request: OpenAICompatRequest | None = None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[bool, float]:
        decision = retry_controller.on_transport_exception(exc, method="POST")
        if decision.retry:
            logger.warning(
                "Vertex AI chat transport error: bucket=%s used=%d/%d wait=%.2fs error=%s",
                decision.bucket,
                retry_controller.retries_used,
                retry_controller.policy.total_retries,
                decision.delay_seconds,
                exc,
            )
        return decision.retry, decision.delay_seconds

    async def _handle_httpx_status(
        self,
        *,
        response: Any,
        retry_controller: RetryController,
    ) -> tuple[bool, Exception | None]:
        status_code = int(getattr(response, "status_code", 200))
        self._reset_token_if_unauthorized(status_code)
        if status_code not in retry_controller.policy.status_forcelist:
            return False, None
        decision = retry_controller.on_status_code(
            status_code,
            method="POST",
            headers=getattr(response, "headers", {}),
        )
        if not decision.retry:
            return False, None
        logger.warning(
            "Vertex AI chat HTTP %d retry=%d/%d wait=%.2fs",
            status_code,
            retry_controller.retries_used,
            retry_controller.policy.total_retries,
            decision.delay_seconds,
        )
        return True, RuntimeError(f"Vertex AI HTTP {status_code}")

    def _build_stream_chunk_from_httpx_event(
        self,
        event: dict[str, Any],
    ) -> StreamChunk | None:
        self._update_stream_usage_from_event(event)
        choices = event.get("choices")
        if isinstance(choices, list) and choices:
            finish_reason = choices[0].get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                self.last_finish_reason = finish_reason
        return self._extract_stream_chunk(event)

    async def _handle_stream_error_response(
        self,
        *,
        response: Any,
        retry_controller: RetryController,
        body: dict[str, Any],
    ) -> tuple[bool, Exception | None]:
        error_body = await response.aread()
        error_text = error_body.decode()[:2000]
        status = response.status_code

        self._reset_token_if_unauthorized(status)
        if await self._retry_or_raise_http_status(
            retry_controller=retry_controller,
            status_code=status,
            headers=getattr(response, "headers", {}),
            label="Vertex AI stream",
        ):
            return True, RuntimeError(f"Vertex AI HTTP {status}: {error_text}")

        logger.error(
            "Vertex AI HTTP %d (non-retryable): %s\nRequest body: %s",
            status,
            error_text,
            json.dumps(body)[:500],
        )
        raise RuntimeError(f"Vertex AI HTTP {status}: {error_text}")

    async def _execute_stream_request_httpx(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        import httpx

        retry_controller = self._stream_retry() or self._new_retry_controller(stream=True)
        proxy_url = self._get_httpx_stream_proxy_url()
        url = self._build_httpx_stream_url()
        self.last_usage = None
        self.last_finish_reason = None

        while True:
            access_token = await self._get_access_token()
            if not access_token:
                raise _VertexAuthError("Failed to authenticate with Vertex AI")
            self._request_access_token = access_token
            http_client = httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(300.0, connect=10.0, read=300.0),
            )
            try:
                async with http_client.stream(
                    "POST",
                    url,
                    headers=self._build_httpx_stream_headers(),
                    json=payload,
                ) as resp:
                    if resp.status_code >= 400:
                        should_retry, _maybe_error = await self._handle_stream_error_response(
                            response=resp,
                            retry_controller=retry_controller
                            or self._new_retry_controller(stream=True),
                            body=payload,
                        )
                        if should_retry:
                            continue
                    async for stream_chunk in super()._stream_httpx_response_chunks(resp):
                        yield stream_chunk
                return
            except httpx.TransportError as exc:
                if retry_controller is None:
                    logger.error("Vertex AI stream transport error (non-retryable): %s", exc)
                    raise
                if await self._retry_or_raise_transport(
                    retry_controller=retry_controller,
                    exc=exc,
                    label="Vertex AI stream",
                ):
                    continue
                logger.error("Vertex AI stream transport error (non-retryable): %s", exc)
                raise
            finally:
                self._request_access_token = None
                await http_client.aclose()

    async def _stream_request_httpx(
        self,
        request: Any,
    ) -> AsyncIterator[StreamChunk]:
        if not self.project_id or not self._sa:
            logger.info("Using mock streaming (no project ID or service account)")
            words = f"Mock response from {request.model}: ...".split()
            for word in words:
                yield StreamChunk(content_delta=word + " ")
            return
        payload = self._encode_stream_request_for_httpx(request)
        try:
            chunk_count = 0
            async for chunk in self._execute_stream_request_httpx(payload):
                chunk_count += 1
                yield chunk
            self.last_finish_reason = self.last_finish_reason or "stop"
            logger.info(
                "Vertex AI stream completed: %d chunks, usage=%s",
                chunk_count,
                self.last_usage,
            )
        except _VertexAuthError:
            yield StreamChunk(content_delta="[Error: Failed to authenticate with Vertex AI]")
