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

from houyi.llm.base import DEFAULT_TEMPERATURE, LLMAdapter, LLMMessage, LLMResponse
from houyi.llm.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    SSE_DATA_PREFIX,
    SSE_DONE_SIGNAL,
    USAGE_KEY_COMPLETION_TOKENS,
    USAGE_KEY_PROMPT_TOKENS,
    USAGE_KEY_TOTAL_TOKENS,
    VERTEX_MAX_OUTPUT_TOKENS,
    exponential_backoff,
    is_retryable_status,
)

logger = logging.getLogger(__name__)


class VertexAIAdapter(LLMAdapter):
    """Adapter for Google Gemini via Vertex AI OpenAI-compatible endpoint.

    Auth: reads GOOGLE_APPLICATION_CREDENTIALS (service account JSON),
    signs a JWT with openssl subprocess, exchanges for access_token,
    then calls the Vertex AI OpenAI-compatible chat/completions endpoint.

    Zero external dependencies — uses only stdlib + httpx.
    """

    def __init__(self) -> None:
        from houyi.config.env_config import EnvConfig

        _env = EnvConfig.get()

        self.default_model = _env.gemini_model
        self.model = self.default_model  # uniform interface for callers
        self.last_usage: dict[str, int] | None = None
        self._access_token: str | None = None
        self._token_expiry: float = 0
        self._sa: dict | None = None

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

        from houyi.net.proxy import detect_proxy

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

            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(20.0, connect=10.0),
            ) as client:
                resp = await client.post(
                    self._sa["token_uri"],
                    data=form_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
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

    # ── chat() — non-streaming ────────────────────────────────────

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat completion via Vertex AI OpenAI-compatible endpoint."""
        model = kwargs.pop("model", None) or self.default_model
        normalized = self._normalize_messages(messages)

        if not self.project_id or not self._sa:
            return LLMResponse(
                content="Mock response (no project ID or service account)",
                tool_calls=[],
                finish_reason="stop",
                usage={},
                model=model,
            )

        base_url = self._get_openai_base_url()
        url = f"{base_url}/chat/completions"

        body: dict = {
            "model": f"google/{model}",
            "messages": normalized,
            "stream": False,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = min(max_tokens, VERTEX_MAX_OUTPUT_TOKENS)
        if tools:
            body["tools"] = tools
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            body["tool_choice"] = tool_choice

        import httpx

        from houyi.net.proxy import detect_proxy

        proxy_url = detect_proxy()

        max_retries = DEFAULT_MAX_RETRIES
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            access_token = await self._get_access_token()
            if not access_token:
                return LLMResponse(
                    content="",
                    tool_calls=[],
                    finish_reason="error",
                    usage={},
                    model=model,
                    metadata={"error": "Failed to authenticate with Vertex AI"},
                )

            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
            ) as client:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )

                if resp.status_code == 401:
                    self._access_token = None
                    self._token_expiry = 0

                if is_retryable_status(resp.status_code) and attempt < max_retries:
                    wait = DEFAULT_RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Vertex AI chat HTTP %d (retry %d/%d, wait %.1fs)",
                        resp.status_code,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    last_error = RuntimeError(
                        f"Vertex AI HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code >= 400:
                    error_text = resp.text[:1000]
                    raise RuntimeError(f"Vertex AI HTTP {resp.status_code}: {error_text}")

                data = resp.json()
                break
        else:
            raise last_error or RuntimeError("Vertex AI chat: max retries exhausted")

        result = LLMResponse.from_raw_dict(data, model_fallback=model)
        self.last_usage = dict(result.usage)
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
        """Stream chat completion via Vertex AI OpenAI-compatible endpoint.

        Includes exponential-backoff retry for transient errors (429, 500, etc.).
        Non-retryable errors (400, 401, 403, 404) fail immediately.
        On 401, the cached access token is invalidated and re-fetched on next retry.
        """
        model = kwargs.pop("model", None) or self.default_model
        self.last_usage = None
        normalized = self._normalize_messages(messages)

        if not self.project_id or not self._sa:
            logger.info("Using mock streaming (no project ID or service account)")
            words = f"Mock response from {model}: ...".split()
            for word in words:
                yield (word + " ", None)
            return

        base_url = self._get_openai_base_url()
        url = f"{base_url}/chat/completions"

        supported_keys = {"temperature", "max_tokens", "top_p", "stop"}
        body: dict = {
            "model": f"google/{model}",
            "messages": normalized,
            "stream": True,
        }
        if temperature != DEFAULT_TEMPERATURE:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = min(max_tokens, VERTEX_MAX_OUTPUT_TOKENS)
        for k, v in kwargs.items():
            if k in supported_keys and v is not None:
                if k == "max_tokens" and isinstance(v, int) and v > VERTEX_MAX_OUTPUT_TOKENS:
                    logger.warning(
                        "Clamping max_tokens from %d to %d for Vertex AI",
                        v,
                        VERTEX_MAX_OUTPUT_TOKENS,
                    )
                    v = VERTEX_MAX_OUTPUT_TOKENS
                body[k] = v

        enable_reasoning = kwargs.get("enable_reasoning", False)
        if enable_reasoning:
            body["reasoning_effort"] = "high"
            logger.info("Gemini reasoning enabled: reasoning_effort=high")

        import httpx

        from houyi.net.proxy import detect_proxy

        _proxy_url = detect_proxy()

        max_retries = DEFAULT_MAX_RETRIES
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            access_token = await self._get_access_token()
            if not access_token:
                yield ("[Error: Failed to authenticate with Vertex AI]", None)
                return

            logger.info(
                "Vertex AI request (attempt %d/%d): model=%s, url=%s, messages=%d",
                attempt + 1,
                max_retries + 1,
                model,
                url,
                len(normalized),
            )

            http_client = httpx.AsyncClient(
                proxy=_proxy_url,
                timeout=httpx.Timeout(300.0, connect=10.0, read=300.0),
            )
            try:
                async with http_client.stream(
                    "POST",
                    url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                ) as resp:
                    if resp.status_code >= 400:
                        error_body = await resp.aread()
                        error_text = error_body.decode()[:2000]
                        status = resp.status_code

                        if status == 401:
                            self._access_token = None
                            self._token_expiry = 0

                        if is_retryable_status(status) and attempt < max_retries:
                            logger.warning(
                                "Vertex AI HTTP %d (retryable, attempt %d/%d): %s",
                                status,
                                attempt + 1,
                                max_retries + 1,
                                error_text[:200],
                            )
                            last_error = Exception(f"Vertex AI HTTP {status}: {error_text}")
                            await exponential_backoff(attempt)
                            continue

                        logger.error(
                            "Vertex AI HTTP %d (non-retryable): %s\nRequest body: %s",
                            status,
                            error_text,
                            json.dumps(body)[:500],
                        )
                        raise RuntimeError(f"Vertex AI HTTP {status}: {error_text}")

                    chunk_count = 0
                    async for line in resp.aiter_lines():
                        if not line.startswith(SSE_DATA_PREFIX):
                            continue
                        payload = line[len(SSE_DATA_PREFIX) :].strip()
                        if payload == SSE_DONE_SIGNAL:
                            break
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        usage_data = event.get("usage")
                        if isinstance(usage_data, dict):
                            self.last_usage = {
                                USAGE_KEY_PROMPT_TOKENS: usage_data.get(USAGE_KEY_PROMPT_TOKENS, 0),
                                USAGE_KEY_COMPLETION_TOKENS: usage_data.get(
                                    USAGE_KEY_COMPLETION_TOKENS, 0
                                ),
                                USAGE_KEY_TOTAL_TOKENS: usage_data.get(USAGE_KEY_TOTAL_TOKENS, 0),
                            }

                        choices = event.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "")
                        if content or reasoning:
                            chunk_count += 1
                            yield (content or "", reasoning if reasoning else None)

                    logger.info(
                        "Vertex AI stream completed: %d chunks, usage=%s",
                        chunk_count,
                        self.last_usage,
                    )
                    return

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Vertex AI timeout (attempt %d/%d): %s",
                        attempt + 1,
                        max_retries + 1,
                        e,
                    )
                    await exponential_backoff(attempt)
                    continue
                logger.error("Vertex AI timeout after %d attempts: %s", max_retries + 1, e)
                raise
            except httpx.ConnectError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Vertex AI connection error (attempt %d/%d): %s",
                        attempt + 1,
                        max_retries + 1,
                        e,
                    )
                    await exponential_backoff(attempt)
                    continue
                logger.error("Vertex AI connection error after %d attempts: %s", max_retries + 1, e)
                raise
            except Exception as e:
                logger.error("Vertex AI API error: %s", e, exc_info=True)
                raise
            finally:
                await http_client.aclose()

        if last_error:
            raise last_error
