"""Google Gemini adapter using google-genai SDK (REST-based).

Two API modes:

1. **Vertex AI** (``GOOGLE_CLOUD_PROJECT`` required) — routes to
   ``aiplatform.googleapis.com``.  Supports ALL models including
   preview.  Auth: ADC, service account, or GCP API key.
2. **Developer API** (``GOOGLE_API_KEY`` only) — routes to
   ``generativelanguage.googleapis.com``.  Auth: AI Studio API key
   (``AIza...``).

Uses the google-genai SDK which communicates via REST API,
making it compatible with HTTP proxies (unlike the gRPC-based vertexai SDK).

Proxy handling:
    Uses Python's ``urllib.request.getproxies()`` for cross-platform system
    proxy detection (macOS SystemConfiguration, Windows registry, Linux env
    vars).  When a proxy is detected, a custom ``httpx`` client with the
    proxy explicitly configured is injected into the ``genai.Client`` via
    ``http_options``, guaranteeing the proxy is used for all requests.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any
from urllib.request import getproxies

from houyi.config.env_config import _DEFAULT_GOOGLE_LOCATION
from houyi.llm.base import DEFAULT_TEMPERATURE, LLMAdapter, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

# ── Proxy detection (cross-platform) ─────────────────────────────────


def _detect_proxy() -> str | None:
    """Detect the system HTTPS/HTTP proxy.

    ``urllib.request.getproxies()`` is the standard cross-platform approach:

    * **macOS** — reads SystemConfiguration framework (same data as
      ``scutil --proxy``, no subprocess needed).
    * **Windows** — reads Internet Explorer / system registry settings.
    * **Linux** — reads ``HTTPS_PROXY`` / ``HTTP_PROXY`` / ``ALL_PROXY``
      environment variables.

    Returns the proxy URL (e.g. ``http://127.0.0.1:7890``) or ``None``.
    """
    proxies = getproxies()
    proxy_url = proxies.get("https") or proxies.get("http")
    if proxy_url:
        logger.debug("System proxy detected: %s (all: %s)", proxy_url, proxies)
    else:
        logger.debug("No system proxy detected (getproxies=%s)", proxies)
    return proxy_url


def _build_proxy_http_options(proxy_url: str) -> dict[str, Any]:
    """Build ``http_options`` dict with custom httpx clients that use *proxy_url*.

    Passing custom ``httpx`` clients via ``http_options`` to ``genai.Client``
    guarantees the proxy is used for every request — the SDK's default
    ``aiohttp`` streaming path is bypassed in favour of ``httpx``, which has
    more reliable HTTP CONNECT tunnel proxy support.
    """
    import httpx  # google-genai already depends on httpx

    return {
        "httpx_client": httpx.Client(
            proxy=proxy_url,
            follow_redirects=True,
        ),
        "httpx_async_client": httpx.AsyncClient(
            proxy=proxy_url,
            follow_redirects=True,
        ),
    }


class GoogleVertexGeminiAdapter(LLMAdapter):
    """Adapter for Google Gemini via google-genai SDK (REST).

    Two API modes:

    1. **Vertex AI** — ``GOOGLE_CLOUD_PROJECT`` set (auto-detected from
       service-account JSON or explicit env var).  Auth via ADC / SA /
       GCP API key.  Supports ALL models including preview.
    2. **Developer API** — only ``GOOGLE_API_KEY`` (``AIza...`` from
       AI Studio).  Routes to ``generativelanguage.googleapis.com``.

    Compatible with both execution engine (``stream_completion``) and
    chat service (``stream_chat`` yielding tuples).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        project: str | None = None,
        location: str = _DEFAULT_GOOGLE_LOCATION,
        credentials_path: str | None = None,
    ) -> None:
        self.model = model
        self.default_model = model  # alias used by execution engine
        self._api_key = api_key
        self.project = project
        self.location = location
        self.credentials_path = credentials_path
        self.last_usage: dict[str, int] | None = None

        try:
            from google import genai  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        # ── Proxy detection (cross-platform) ──────────────────────────
        self._proxy_url = _detect_proxy()
        http_options = _build_proxy_http_options(self._proxy_url) if self._proxy_url else None

        if self.project:
            # ── Vertex AI mode ───────────────────────────────────────
            # Auth: ADC / service account (GOOGLE_APPLICATION_CREDENTIALS).
            # AI Studio API keys (AIza...) are NOT used here — they only
            # work with the Developer API endpoint.
            if self.credentials_path:
                os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", self.credentials_path)

            self._client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                http_options=http_options,
            )
            self._auth_mode = "vertex_ai"
            logger.info(
                "Gemini: Vertex AI mode (project=%s, location=%s, model=%s, proxy=%s)",
                self.project,
                self.location,
                self.model,
                self._proxy_url or "none",
            )

        elif self._api_key:
            # ── Developer API mode ───────────────────────────────────
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=http_options,
            )
            self._auth_mode = "developer_api"
            logger.info(
                "Gemini: Developer API mode (model=%s, proxy=%s)",
                self.model,
                self._proxy_url or "none",
            )

        else:
            raise ValueError("Either GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY must be set for Gemini")

    @classmethod
    def from_env(cls) -> GoogleVertexGeminiAdapter:
        """Create from environment variables.

        If ``GOOGLE_CLOUD_PROJECT`` is available (from env var or
        auto-detected from service-account JSON) → Vertex AI mode.
        Otherwise falls back to Developer API with ``GOOGLE_API_KEY``.
        """
        from houyi.config.env_config import EnvConfig

        env = EnvConfig.get()
        model = env.gemini_model or "gemini-2.5-pro"
        api_key = env.google_api_key
        project = env.google_project
        location = env.google_location
        credentials_path = env.google_credentials_path

        if not project and not api_key:
            raise ValueError("Either GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY must be set for Gemini")

        return cls(
            model=model,
            api_key=api_key,
            project=project,
            location=location,
            credentials_path=credentials_path,
        )

    # ── Non-streaming (tool calling) ──────────────────────────────

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            from google.genai import types  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        normalized_messages = self._normalize_messages(messages)

        system_instruction = None
        contents: list[types.Content] = []
        for message in normalized_messages:
            if message["role"] == "system":
                system_instruction = message["content"]
                continue
            role = "user" if message["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message.get("content", ""))],
                )
            )

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            config_kwargs["tool_config"] = self._convert_tool_choice(tool_choice)

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise self._wrap_sdk_error(exc) from exc
        return self._normalize_response(response)

    # ── Streaming (execution engine + chat) ───────────────────────

    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream completion — delegates to ``stream_chat``.

        Yields ``(content_delta, reasoning_delta)`` tuples, compatible with
        the execution engine's ``async for content, reasoning in ...`` pattern.
        """
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.stream_chat(messages, model=model, **kwargs):  # type: ignore[arg-type]
            yield chunk

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream chat completion.

        Yields ``(content_delta, None)`` tuples (Gemini does not expose
        reasoning tokens in the streaming API).
        """
        try:
            from google.genai import types  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        normalized_messages = self._normalize_messages(messages)

        system_instruction = None
        contents: list[types.Content] = []
        for message in normalized_messages:
            if message["role"] == "system":
                system_instruction = message["content"]
                continue
            role = "user" if message["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message.get("content", ""))],
                )
            )

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                if chunk.text:
                    yield (chunk.text, None)
        except Exception as exc:
            raise self._wrap_sdk_error(exc) from exc

    # ── Response normalization ────────────────────────────────────

    def _normalize_response(self, response: Any) -> LLMResponse:
        content = ""
        tool_calls: list[dict[str, Any]] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content += part.text
                if part.function_call:
                    fc = part.function_call
                    args = fc.args
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls.append(
                        {
                            "id": f"call_{fc.name}",
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": args,
                            },
                        }
                    )

        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        self.last_usage = usage

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=usage,
            model=self.model,
        )

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool schema to Vertex function declarations."""
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            function_payload = tool.get("function") if isinstance(tool, dict) else None
            if not function_payload:
                continue
            declarations.append(
                {
                    "name": function_payload.get("name"),
                    "description": function_payload.get("description"),
                    "parameters": function_payload.get("parameters"),
                }
            )
        if not declarations:
            return []
        return [{"function_declarations": declarations}]

    def _convert_tool_choice(self, tool_choice: Any) -> dict[str, Any]:
        """Map OpenAI tool_choice to Vertex function calling config."""
        mode = "AUTO"
        if isinstance(tool_choice, dict):
            mode = "ANY"
        elif tool_choice in {"required", "any"}:
            mode = "ANY"
        elif tool_choice in {"none"}:
            mode = "NONE"
        return {"function_calling_config": {"mode": mode}}

    def _wrap_sdk_error(self, exc: Exception) -> Exception:
        """Wrap google-genai SDK errors with actionable diagnostic messages."""
        exc_str = str(exc)

        # 400 location not supported — Developer API blocks VPN/proxy/datacenter IPs
        if "location is not supported" in exc_str or "FAILED_PRECONDITION" in exc_str:
            if self._auth_mode == "developer_api":
                return RuntimeError(
                    f"Gemini Developer API: region not supported "
                    f"(model={self.model}). "
                    f"Google blocks requests from VPN/proxy/datacenter IPs "
                    f"and restricted regions. This cannot be bypassed with "
                    f"a proxy. "
                    f"Fix: switch to Vertex AI mode by setting "
                    f"GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS "
                    f"in .env. Vertex AI authenticates via service account "
                    f"and has no IP or region restrictions."
                )
            return RuntimeError(
                f"Gemini API: region not supported (auth_mode={self._auth_mode}, "
                f"model={self.model}). Check GOOGLE_CLOUD_LOCATION in .env."
            )

        # 401 UNAUTHENTICATED
        if "401" in exc_str or "UNAUTHENTICATED" in exc_str:
            if self._auth_mode == "developer_api":
                return RuntimeError(
                    f"Gemini Developer API 401 (model={self.model}). "
                    f"Check GOOGLE_API_KEY is a valid AI Studio key (AIza...). "
                    f"Or switch to Vertex AI: set GOOGLE_CLOUD_PROJECT + "
                    f"GOOGLE_APPLICATION_CREDENTIALS."
                )
            return RuntimeError(
                f"Gemini Vertex AI 401 (model={self.model}). "
                f"Check: (1) API key / SA credentials valid, (2) GOOGLE_CLOUD_PROJECT correct, "
                f"(3) Vertex AI API enabled in GCP project."
            )

        # 403 PERMISSION_DENIED
        if "403" in exc_str or "PERMISSION_DENIED" in exc_str:
            return RuntimeError(
                f"Gemini API 403 (auth_mode={self._auth_mode}, model={self.model}). "
                f"Check: Vertex AI API enabled + aiplatform.user role granted."
            )

        # 404 NOT_FOUND
        if "404" in exc_str or "NOT_FOUND" in exc_str:
            return RuntimeError(f"Gemini API 404 (model={self.model}). Check GEMINI_MODEL in .env.")

        # Pass through with context
        return RuntimeError(
            f"Gemini API error (auth_mode={self._auth_mode}, model={self.model}): {exc}"
        )
