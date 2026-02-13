"""LLM Adapter for streaming output.

Supports:
- DeepSeek via SiliconFlow
- Google Gemini via Vertex AI
"""

import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from houyi.llm.models import PROVIDER_SILICONFLOW, PROVIDER_VERTEX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Implementation-detail constants (not part of public API)
# ---------------------------------------------------------------------------

# Retry / backoff
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_BASE_DELAY: float = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY: float = 10.0  # seconds

# Vertex AI model limit
VERTEX_MAX_OUTPUT_TOKENS: int = 65_536

# SSE protocol (OpenAI-compatible streaming)
SSE_DATA_PREFIX = "data: "
SSE_DONE_SIGNAL = "[DONE]"

# OpenAI-compatible usage response keys
USAGE_KEY_PROMPT_TOKENS = "prompt_tokens"
USAGE_KEY_COMPLETION_TOKENS = "completion_tokens"
USAGE_KEY_TOTAL_TOKENS = "total_tokens"

# ---------------------------------------------------------------------------
# Shared retry / backoff helpers (inspired by litellm's tenacity patterns)
# ---------------------------------------------------------------------------


def _is_retryable_status(status_code: int) -> bool:
    """Check if an HTTP status code is retryable."""
    return status_code in RETRYABLE_STATUS_CODES


async def _exponential_backoff(
    attempt: int, base: float = DEFAULT_RETRY_BASE_DELAY, cap: float = DEFAULT_RETRY_MAX_DELAY
) -> None:
    """Sleep with exponential backoff + jitter (full-jitter strategy)."""
    delay = min(base * (2**attempt), cap)
    jitter = random.uniform(0, delay)
    logger.info("Retry attempt %d: backing off %.2fs", attempt + 1, jitter)
    await asyncio.sleep(jitter)


class LLMAdapter(ABC):
    """Base class for LLM adapters."""

    @abstractmethod
    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion from LLM.

        Args:
            prompt: Input prompt
            model: Model name (optional, uses default if not specified)
            **kwargs: Additional model-specific parameters

        Yields:
            Text chunks from the LLM
        """
        pass

    async def stream_chat(
        self,
        messages: list[dict],
        model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream chat completion from LLM with full message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      e.g. [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            model: Model name (optional, uses default if not specified)
            **kwargs: Additional model-specific parameters (temperature, max_tokens, etc.)

        Yields:
            Tuples of (content_delta, reasoning_delta). reasoning_delta is None
            for models that don't support reasoning.
        """
        # Default implementation: extract last user message and delegate to stream_completion.
        # Subclasses should override for proper multi-turn support.
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        async for chunk in self.stream_completion(last_user, model=model, **kwargs):
            if isinstance(chunk, tuple):
                yield chunk
            else:
                yield (chunk, None)


class SiliconFlowAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible APIs (SiliconFlow, DeepSeek, vLLM, Ollama, etc.).

    Strategy: openai SDK first (industry standard), httpx raw SSE as fallback.
    The openai SDK provides automatic retries, proper error types, streaming
    support, and token usage tracking — matching LangChain/LlamaIndex patterns.
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
        self.last_usage: dict[str, int] | None = None
        self._sdk_client: object | None = None

        if not self.api_key:
            logger.warning("SILICONFLOW_API_KEY not set, will use mock responses")

        # Detect SDK availability once per class (not per instance)
        if SiliconFlowAdapter._SDK_AVAILABLE is None:
            try:
                import openai  # noqa: F401  # pylint: disable=unused-import

                SiliconFlowAdapter._SDK_AVAILABLE = True
                logger.info("openai SDK available — using SDK mode (recommended)")
            except ImportError:
                SiliconFlowAdapter._SDK_AVAILABLE = False
                logger.warning(
                    "openai SDK not installed — falling back to httpx raw SSE. "
                    "Install with: pip install 'houyi[model-adapters]' or pip install openai"
                )

    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream completion from OpenAI-compatible API.

        Returns:
            AsyncIterator of (content, reasoning_content) tuples
        """
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.stream_chat(
            messages,
            model=model,
            enable_reasoning=enable_reasoning,
            thinking_budget=thinking_budget,
            **kwargs,
        ):
            yield chunk

    async def stream_chat(
        self,
        messages: list[dict],
        model: str | None = None,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream chat completion with full message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model name (optional, uses default if not specified)
            enable_reasoning: Enable reasoning/thinking mode (DeepSeek extension)
            thinking_budget: Token budget for reasoning (DeepSeek extension)
            **kwargs: Additional model-specific parameters

        Yields:
            Tuples of (content_delta, reasoning_delta).
        """
        model = model or self.default_model

        if not self.api_key:
            logger.info("Using mock streaming (no API key)")
            last_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_content = msg.get("content", "")
                    break
            words = f"Mock response from {model}: {last_content[:50]}...".split()
            for word in words:
                yield (word + " ", None)
            return

        if SiliconFlowAdapter._SDK_AVAILABLE:
            async for chunk in self._stream_via_sdk(
                messages, model, enable_reasoning, thinking_budget, **kwargs
            ):
                yield chunk
        else:
            async for chunk in self._stream_via_httpx(
                messages, model, enable_reasoning, thinking_budget, **kwargs
            ):
                yield chunk

    async def _stream_via_sdk(
        self,
        messages: list[dict],
        model: str,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream via openai SDK (preferred path)."""
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

        sdk_kwargs: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            sdk_kwargs["extra_body"] = extra_body
        for k, v in kwargs.items():
            if v is not None and k not in sdk_kwargs:
                sdk_kwargs[k] = v

        logger.info("SDK streaming to %s model=%s", self.base_url, model)
        chunk_count = 0
        reasoning_count = 0
        self.last_usage = None

        try:
            stream = await client.chat.completions.create(**sdk_kwargs)
            async for chunk in stream:
                # Capture usage from final chunk
                if chunk.usage:
                    self.last_usage = {
                        USAGE_KEY_PROMPT_TOKENS: chunk.usage.prompt_tokens or 0,
                        USAGE_KEY_COMPLETION_TOKENS: chunk.usage.completion_tokens or 0,
                        USAGE_KEY_TOTAL_TOKENS: chunk.usage.total_tokens or 0,
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content = delta.content if delta else None
                # reasoning_content is a DeepSeek extension, access via getattr
                reasoning = getattr(delta, "reasoning_content", None)

                if isinstance(content, str) and content:
                    chunk_count += 1
                if isinstance(reasoning, str) and reasoning:
                    reasoning_count += 1

                if (isinstance(content, str) and content) or (
                    isinstance(reasoning, str) and reasoning
                ):
                    yield (content or "", reasoning if isinstance(reasoning, str) else None)

            logger.info(
                "SDK stream completed: %d content, %d reasoning chunks, usage=%s",
                chunk_count,
                reasoning_count,
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
        **kwargs,
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
                    # Read error body before raise_for_status so we can log it
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


class VertexAIAdapter(LLMAdapter):
    """Adapter for Google Gemini via Vertex AI OpenAI-compatible endpoint.

    Auth: reads GOOGLE_APPLICATION_CREDENTIALS (service account JSON),
    signs a JWT with openssl subprocess, exchanges for access_token,
    then calls the Vertex AI OpenAI-compatible chat/completions endpoint.

    Zero external dependencies — uses only stdlib + httpx.
    """

    def __init__(self):
        from houyi.config.env_config import EnvConfig

        _env = EnvConfig.get()

        self.default_model = _env.gemini_model
        self.last_usage: dict[str, int] | None = None
        self._access_token: str | None = None
        self._token_expiry: float = 0
        self._sa: dict | None = None

        # Load service account JSON
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

        # Read project/location from service account or env
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

    def _get_openai_base_url(self) -> str:
        """Build the Vertex AI OpenAI-compatible base URL.

        Uses v1beta1 API version. Supports both regional and global endpoints:
        - global: https://aiplatform.googleapis.com/v1beta1/projects/.../locations/global/endpoints/openapi
        - regional: https://{loc}-aiplatform.googleapis.com/v1beta1/projects/.../locations/{loc}/endpoints/openapi
        """
        if self.location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{self.location}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1beta1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"endpoints/openapi"
        )

    def _sign_jwt_with_openssl(self) -> str:
        """Create a signed JWT using openssl subprocess (no python crypto deps)."""
        import base64
        import subprocess
        import tempfile
        import time as _time

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

        # Write private key to temp file for openssl
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

    async def _get_access_token(self) -> str | None:
        """Get a valid access token by signing JWT and exchanging with Google OAuth."""
        import time as _time
        import urllib.parse
        import urllib.request

        now = _time.time()
        if self._access_token and now < self._token_expiry - 60:
            return self._access_token

        if not self._sa:
            return None

        try:
            jwt_token = self._sign_jwt_with_openssl()

            # Exchange JWT for access token
            data = urllib.parse.urlencode(
                {
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": jwt_token,
                }
            ).encode()
            req = urllib.request.Request(
                self._sa["token_uri"],
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                token_data = json.loads(resp.read())

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

    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion from Gemini via Vertex AI."""
        messages = [{"role": "user", "content": prompt}]
        async for content, _reasoning in self.stream_chat(messages, model=model, **kwargs):
            yield content

    async def stream_chat(
        self,
        messages: list[dict],
        model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream chat completion via Vertex AI OpenAI-compatible endpoint.

        Includes exponential-backoff retry for transient errors (429, 500, 502, 503, 504).
        Non-retryable errors (400, 401, 403, 404) fail immediately.
        On 401, the cached access token is invalidated and re-fetched on next retry.
        """
        model = model or self.default_model
        self.last_usage = None

        if not self.project_id or not self._sa:
            logger.info("Using mock streaming (no project ID or service account)")
            words = f"Mock response from {model}: ...".split()
            for word in words:
                yield (word + " ", None)
            return

        base_url = self._get_openai_base_url()
        url = f"{base_url}/chat/completions"

        # Filter and clamp kwargs for Vertex AI
        supported_keys = {"temperature", "max_tokens", "top_p", "stop"}
        body: dict = {
            "model": f"google/{model}",
            "messages": messages,
            "stream": True,
        }
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

        # Gemini thinking/reasoning support via OpenAI-compatible reasoning_effort.
        # Vertex AI maps "low"/"medium"/"high" to 1K/8K/24K thinking token budgets.
        # Note: Gemini 2.5 Pro always thinks (cannot be turned off).
        enable_reasoning = kwargs.get("enable_reasoning", False)
        if enable_reasoning:
            body["reasoning_effort"] = "high"
            logger.info("Gemini reasoning enabled: reasoning_effort=high")

        import httpx

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
                len(messages),
            )

            http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0, read=300.0))
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

                        # Invalidate token on auth errors so next retry gets a fresh one
                        if status == 401:
                            self._access_token = None
                            self._token_expiry = 0

                        if _is_retryable_status(status) and attempt < max_retries:
                            logger.warning(
                                "Vertex AI HTTP %d (retryable, attempt %d/%d): %s",
                                status,
                                attempt + 1,
                                max_retries + 1,
                                error_text[:200],
                            )
                            last_error = Exception(f"Vertex AI HTTP {status}: {error_text}")
                            await _exponential_backoff(attempt)
                            continue

                        logger.error(
                            "Vertex AI HTTP %d (non-retryable): %s\nRequest body: %s",
                            status,
                            error_text,
                            json.dumps(body)[:500],
                        )
                        raise RuntimeError(f"Vertex AI HTTP {status}: {error_text}")

                    # Successful response — stream SSE chunks
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

                        # Capture usage
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
                    return  # Success — exit retry loop

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Vertex AI timeout (attempt %d/%d): %s", attempt + 1, max_retries + 1, e
                    )
                    await _exponential_backoff(attempt)
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
                    await _exponential_backoff(attempt)
                    continue
                logger.error("Vertex AI connection error after %d attempts: %s", max_retries + 1, e)
                raise
            except Exception as e:
                logger.error("Vertex AI API error: %s", e, exc_info=True)
                raise
            finally:
                await http_client.aclose()

        # Exhausted all retries
        if last_error:
            raise last_error


class LLMAdapterFactory:
    """Factory for creating LLM adapters."""

    @staticmethod
    def create(provider: str | None = None) -> LLMAdapter:
        """Create an LLM adapter.

        Args:
            provider: Provider name (siliconflow, vertex, or None for default)

        Returns:
            LLM adapter instance
        """
        from houyi.config.env_config import EnvConfig

        provider = provider or EnvConfig.get().default_llm_provider

        if provider == PROVIDER_SILICONFLOW:
            return SiliconFlowAdapter()
        elif provider == PROVIDER_VERTEX:
            return VertexAIAdapter()
        else:
            logger.warning("Unknown provider %s, using SiliconFlow", provider)
            return SiliconFlowAdapter()
