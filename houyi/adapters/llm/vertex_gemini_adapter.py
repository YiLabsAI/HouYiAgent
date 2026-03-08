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
    Uses ``houyi.infrastructure.net.proxy.detect_proxy()`` for cross-platform system
    proxy detection (with optional ``HOUYI_PROXY_URL`` override).
    When a proxy is detected, a custom ``httpx`` client with the
    proxy explicitly configured is injected into the ``genai.Client`` via
    ``http_options``, guaranteeing the proxy is used for all requests.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from houyi.adapters.llm.base import (
    DEFAULT_TEMPERATURE,
    LLMAdapter,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    _sanitize_usage,
)
from houyi.infrastructure.config.env_config import _DEFAULT_GOOGLE_LOCATION
from houyi.infrastructure.net.proxy import detect_proxy

if TYPE_CHECKING:
    from google.genai.types import HttpOptionsDict

logger = logging.getLogger(__name__)


def _build_proxy_http_options(proxy_url: str) -> HttpOptionsDict:
    """Build ``http_options`` dict with custom httpx clients that use *proxy_url*.

    Passing custom ``httpx`` clients via ``http_options`` to ``genai.Client``
    guarantees the proxy is used for every request — the SDK's default
    ``aiohttp`` streaming path is bypassed in favour of ``httpx``, which has
    more reliable HTTP CONNECT tunnel proxy support.
    """
    import httpx  # google-genai already depends on httpx

    return cast(
        "HttpOptionsDict",
        {
            "httpx_client": httpx.Client(
                proxy=proxy_url,
                follow_redirects=True,
            ),
            "httpx_async_client": httpx.AsyncClient(
                proxy=proxy_url,
                follow_redirects=True,
            ),
        },
    )


class GoogleVertexGeminiAdapter(LLMAdapter):
    """Adapter for Google Gemini via google-genai SDK (REST).

    Two API modes:

    1. **Vertex AI** — ``GOOGLE_CLOUD_PROJECT`` set (auto-detected from
       service-account JSON or explicit env var).  Auth via ADC / SA /
       GCP API key.  Supports ALL models including preview.
    2. **Developer API** — only ``GOOGLE_API_KEY`` (``AIza...`` from
       AI Studio).  Routes to ``generativelanguage.googleapis.com``.

    Compatible with both execution engine (``stream_completion``) and
    chat service (``stream_chat`` yielding ``StreamChunk`` objects).
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
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        # ── Proxy detection (cross-platform) ──────────────────────────
        self._proxy_url = detect_proxy()
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
        from houyi.infrastructure.config.env_config import EnvConfig

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

    def _prepare_contents_and_system(
        self, types: Any, messages: list[LLMMessage | dict]
    ) -> tuple[str | None, list[Any]]:
        normalized_messages = self._normalize_messages(messages)
        system_instruction = None
        contents: list[Any] = []
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
        return system_instruction, contents

    def _build_generate_config(
        self,
        types: Any,
        *,
        temperature: float,
        max_tokens: int | None,
        tools: list[dict] | None,
        system_instruction: str | None,
        extra_kwargs: dict[str, Any],
    ) -> Any:
        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)
        tool_choice = extra_kwargs.pop("tool_choice", None)
        if tool_choice:
            config_kwargs["tool_config"] = self._convert_tool_choice(tool_choice)
        config_kwargs.update(
            {key: value for key, value in extra_kwargs.items() if value is not None}
        )
        return types.GenerateContentConfig(**config_kwargs)

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        system_instruction, contents = self._prepare_contents_and_system(types, messages)
        config = self._build_generate_config(
            types,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system_instruction=system_instruction,
            extra_kwargs=kwargs,
        )

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

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion.

        Yields:
            ``StreamChunk`` objects.
            Gemini function_calls are complete objects (not deltas),
            so ``tool_calls_delta`` is always ``None``.
        """
        try:
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        system_instruction, contents = self._prepare_contents_and_system(types, messages)
        config = self._build_generate_config(
            types,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system_instruction=system_instruction,
            extra_kwargs=kwargs,
        )

        self.last_usage = {}
        self.last_finish_reason = None

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                usage_metadata = getattr(chunk, "usage_metadata", None)
                if usage_metadata:
                    self.last_usage = {
                        "prompt_tokens": usage_metadata.prompt_token_count or 0,
                        "completion_tokens": usage_metadata.candidates_token_count or 0,
                        "total_tokens": usage_metadata.total_token_count or 0,
                    }

                text = getattr(chunk, "text", None)
                if text:
                    yield StreamChunk(content_delta=text)

            self.last_finish_reason = "stop"
        except Exception as exc:
            raise self._wrap_sdk_error(exc) from exc

    # ── Response normalization ────────────────────────────────────

    def _normalize_response(self, response: Any) -> LLMResponse:
        content = self._extract_response_content(response)
        tool_calls: list[dict[str, Any]] = []

        if response.candidates:
            candidate = response.candidates[0]
            candidate_content = getattr(candidate, "content", None)
            parts = getattr(candidate_content, "parts", None) or []
            part_text = "".join(part.text for part in parts if getattr(part, "text", None))
            if not content:
                content = self._extract_candidate_content(candidate) or part_text
            for part in parts:
                if part.text and not content:
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

        usage_metadata = getattr(response, "usage_metadata", None)
        usage = _sanitize_usage(
            {
                "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
                "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(usage_metadata, "total_token_count", None),
            }
        )

        self.last_usage = usage

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=usage,
            model=self.model,
        )

    def _extract_response_content(self, response: Any) -> str:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, str):
                return parsed
            return json.dumps(parsed, ensure_ascii=False)
        return str(getattr(response, "text", "") or "")

    def _extract_candidate_content(self, candidate: Any) -> str:
        candidate_text = getattr(candidate, "text", None)
        if candidate_text:
            return str(candidate_text)
        candidate_content = getattr(candidate, "content", None)
        parts = getattr(candidate_content, "parts", None) or []
        text_parts = [part.text for part in parts if getattr(part, "text", None)]
        return "".join(text_parts)

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
        if isinstance(tool_choice, dict) or tool_choice in {"required", "any"}:
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
