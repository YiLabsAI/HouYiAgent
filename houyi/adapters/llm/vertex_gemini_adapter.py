"""Google Gemini adapter using the primary google-genai transport (REST-based).

Two API modes:

1. **Vertex AI** (``GOOGLE_CLOUD_PROJECT`` required) — routes to
   ``aiplatform.googleapis.com``.  Supports ALL models including
   preview.  Auth: ADC, service account, or GCP API key.
2. **Developer API** (``GOOGLE_API_KEY`` only) — routes to
   ``generativelanguage.googleapis.com``.  Auth: AI Studio API key
   (``AIza...``).

Uses the default google-genai client over REST,
making it compatible with HTTP proxies (unlike the gRPC-based vertexai client).

Proxy handling:
    Uses ``houyi.infrastructure.net.proxy.detect_proxy()`` for cross-platform system
    proxy detection (with optional ``HOUYI_PROXY_URL`` override).
    When a proxy is detected, a custom ``httpx`` client with the
    proxy explicitly configured is injected into the ``genai.Client`` via
    ``http_options``, guaranteeing the proxy is used for all requests.
"""

from __future__ import annotations

import base64
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
    _normalize_usage,
)
from houyi.infrastructure.config.env_config import _DEFAULT_GOOGLE_LOCATION
from houyi.infrastructure.net.proxy import detect_proxy

if TYPE_CHECKING:
    from google.genai.types import HttpOptionsDict

logger = logging.getLogger(__name__)

_GEMINI_RETRY_ATTEMPTS = 5
_GEMINI_RETRY_INITIAL_DELAY = 1.0
_GEMINI_RETRY_MAX_DELAY = 60.0
_GEMINI_RETRY_EXP_BASE = 2.0
_GEMINI_RETRY_JITTER = 1.0
_GEMINI_RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]
_GEMINI_RATE_LIMIT_SIGNALS = ("429", "RESOURCE_EXHAUSTED")
_GEMINI_REGION_SIGNALS = ("location is not supported", "FAILED_PRECONDITION")
_GEMINI_AUTH_SIGNALS = ("401", "UNAUTHENTICATED")
_GEMINI_PERMISSION_SIGNALS = ("403", "PERMISSION_DENIED")
_GEMINI_NOT_FOUND_SIGNALS = ("404", "NOT_FOUND")


def _contains_error_signal(exc_str: str, signals: tuple[str, ...]) -> bool:
    return any(signal in exc_str for signal in signals)


def _classify_client_error(exc: Exception) -> str:
    exc_str = str(exc)
    if _contains_error_signal(exc_str, _GEMINI_RATE_LIMIT_SIGNALS):
        return "rate_limit"
    if _contains_error_signal(exc_str, _GEMINI_REGION_SIGNALS):
        return "region"
    if _contains_error_signal(exc_str, _GEMINI_AUTH_SIGNALS):
        return "auth"
    if _contains_error_signal(exc_str, _GEMINI_PERMISSION_SIGNALS):
        return "permission"
    if _contains_error_signal(exc_str, _GEMINI_NOT_FOUND_SIGNALS):
        return "not_found"
    return "unknown"


def _build_proxy_http_options(proxy_url: str) -> HttpOptionsDict:
    """Build ``http_options`` dict with custom httpx clients that use *proxy_url*.

    Passing custom ``httpx`` clients via ``http_options`` to ``genai.Client``
    guarantees the proxy is used for every request — the default
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


def _build_http_options(types: Any, *, proxy_url: str | None) -> Any:
    """Build client-level google-genai HttpOptions with explicit retry settings.

    The default client already has a tenacity-based retry layer. We configure it explicitly
    so stream/non-stream Gemini calls use the same documented retry envelope for
    429/5xx responses, while still preserving any custom proxy-backed httpx
    clients we inject for enterprise network environments.
    """
    http_options_kwargs: dict[str, Any] = {
        "retry_options": types.HttpRetryOptions(
            attempts=_GEMINI_RETRY_ATTEMPTS,
            initial_delay=_GEMINI_RETRY_INITIAL_DELAY,
            max_delay=_GEMINI_RETRY_MAX_DELAY,
            exp_base=_GEMINI_RETRY_EXP_BASE,
            jitter=_GEMINI_RETRY_JITTER,
            http_status_codes=list(_GEMINI_RETRY_STATUS_CODES),
        )
    }
    if proxy_url:
        http_options_kwargs.update(_build_proxy_http_options(proxy_url))
    return types.HttpOptions(**http_options_kwargs)


def _parse_json_object(raw_value: Any, fallback_key: str) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    try:
        parsed = json.loads(str(raw_value or ""))
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    return {fallback_key: str(raw_value or "")} if fallback_key else {}


def _fallback_tool_call_id(*, function_name: Any, index: int) -> str:
    normalized_name = str(function_name or "tool").strip() or "tool"
    return f"gemini_call_{index}_{normalized_name}"


def _reset_pending_tool_state() -> tuple[list[Any], list[str], list[str]]:
    return [], [], []


def _collect_active_tool_call_ids(
    message: dict[str, Any],
    tool_name_by_call_id: dict[str, str],
) -> list[str]:
    active_tool_call_ids: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = str(tool_call.get("id") or "")
        function_payload = tool_call.get("function")
        if not tool_call_id or not isinstance(function_payload, dict):
            continue
        tool_name = str(function_payload.get("name") or "")
        if tool_name:
            tool_name_by_call_id[tool_call_id] = tool_name
        active_tool_call_ids.append(tool_call_id)
    return active_tool_call_ids


def _append_content_turn(types: Any, contents: list[Any], *, role: str, parts: list[Any]) -> None:
    contents.append(types.Content(role=role, parts=parts))


def _is_thought_part(part: Any) -> bool:
    thought = getattr(part, "thought", None)
    if isinstance(thought, bool):
        return thought
    is_thought = getattr(part, "is_thought", None)
    if isinstance(is_thought, bool):
        return is_thought
    return False


def _extract_text_and_reasoning_from_parts(parts: list[Any]) -> tuple[str, str]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            text_value = str(text)
            if _is_thought_part(part):
                reasoning_parts.append(text_value)
            else:
                content_parts.append(text_value)
            continue

        code_execution_result = getattr(part, "code_execution_result", None)
        code_output = getattr(code_execution_result, "output", None)
        if isinstance(code_output, str) and code_output:
            content_parts.append(code_output)
            continue

        executable_code = getattr(part, "executable_code", None)
        code = getattr(executable_code, "code", None)
        if isinstance(code, str) and code:
            content_parts.append(code)
    return "".join(content_parts), "".join(reasoning_parts)


def _collect_unknown_part_fields(part: Any) -> list[str]:
    names: list[str] = []
    raw_dict = getattr(part, "__dict__", None)
    if isinstance(raw_dict, dict):
        for key, value in raw_dict.items():
            if value is None:
                continue
            if key in {
                "text",
                "function_call",
                "thought",
                "is_thought",
                "thought_signature",
                "thoughtSignature",
            }:
                continue
            names.append(str(key))
    payload = getattr(part, "payload", None)
    if isinstance(payload, dict):
        for key, value in payload.items():
            if value is None:
                continue
            if key in {"text", "function_call", "thought_signature"}:
                continue
            names.append(str(key))
    return sorted(set(names))


def _summarize_unknown_parts(parts: list[Any]) -> list[list[str]]:
    summaries: list[list[str]] = []
    for part in parts:
        names = _collect_unknown_part_fields(part)
        if names:
            summaries.append(names)
    return summaries


def _extract_stream_finish_reason(candidate: Any) -> str | None:
    finish_reason = getattr(candidate, "finish_reason", None)
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason.lower()
    if finish_reason is None:
        return None
    normalized = str(finish_reason).strip()
    return normalized.lower() if normalized else None


def _extract_chunk_candidate(chunk: Any) -> tuple[Any, list[Any]]:
    candidate = (getattr(chunk, "candidates", None) or [None])[0]
    candidate_content = getattr(candidate, "content", None)
    parts = getattr(candidate_content, "parts", None) or []
    return candidate, parts


def _has_function_call_parts(parts: list[Any]) -> bool:
    return any(getattr(part, "function_call", None) is not None for part in parts)


def _extract_thought_signature(part: Any) -> str | None:
    raw_signature = getattr(part, "thought_signature", None)
    if raw_signature is None:
        raw_signature = getattr(part, "thoughtSignature", None)
    if not raw_signature:
        return None
    if isinstance(raw_signature, bytes):
        return base64.b64encode(raw_signature).decode("ascii")
    if isinstance(raw_signature, str):
        return raw_signature
    return None


def _decode_thought_signature(raw_signature: Any) -> bytes | None:
    if isinstance(raw_signature, bytes):
        return raw_signature
    if not isinstance(raw_signature, str) or not raw_signature:
        return None
    try:
        return base64.b64decode(raw_signature)
    except Exception:
        return raw_signature.encode("utf-8")


def _convert_openai_tools_to_vertex(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _convert_openai_tool_choice_to_vertex(tool_choice: Any) -> dict[str, Any]:
    mode = "AUTO"
    if isinstance(tool_choice, dict) or tool_choice in {"required", "any"}:
        mode = "ANY"
    elif tool_choice in {"none"}:
        mode = "NONE"
    return {"function_calling_config": {"mode": mode}}


class GoogleVertexGeminiAdapter(LLMAdapter):
    """Adapter for Google Gemini via the primary google-genai client (REST).

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
        self.last_usage: dict[str, Any] | None = None

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Google GenAI client not installed. Install with: pip install google-genai"
            ) from exc

        # ── Proxy detection (cross-platform) ──────────────────────────
        self._proxy_url = detect_proxy()
        http_options = _build_http_options(types, proxy_url=self._proxy_url)

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

        retry_options = getattr(http_options, "retry_options", None)
        logger.info(
            "Gemini retry config: auth_mode=%s model=%s attempts=%s initial_delay=%s max_delay=%s exp_base=%s jitter=%s status_codes=%s",
            self._auth_mode,
            self.model,
            getattr(retry_options, "attempts", None),
            getattr(retry_options, "initial_delay", None),
            getattr(retry_options, "max_delay", None),
            getattr(retry_options, "exp_base", None),
            getattr(retry_options, "jitter", None),
            getattr(retry_options, "http_status_codes", None),
        )

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

    def _build_function_call_part(
        self,
        types: Any,
        function_payload: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> Any | None:
        tool_name = function_payload.get("name")
        if not tool_name:
            return None
        tool_args = _parse_json_object(function_payload.get("arguments"), "")
        thought_signature = function_payload.get("thought_signature")
        function_call = types.FunctionCall(
            id=str(tool_call_id or ""),
            name=str(tool_name),
            args=tool_args,
        )
        if thought_signature:
            signature_bytes = _decode_thought_signature(thought_signature)
            if signature_bytes:
                return types.Part(
                    function_call=function_call,
                    thought_signature=signature_bytes,
                )
        return types.Part(function_call=function_call)

    def _resolve_tool_call_id(
        self,
        *,
        raw_id: Any,
        function_name: Any,
        index: int,
    ) -> str:
        resolved = str(raw_id or "").strip()
        if resolved:
            return resolved
        return _fallback_tool_call_id(function_name=function_name, index=index)

    def _build_assistant_tool_call_parts(self, types: Any, message: dict[str, Any]) -> list[Any]:
        parts: list[Any] = []
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function_payload = tool_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            function_part = self._build_function_call_part(
                types,
                function_payload,
                str(tool_call.get("id") or ""),
            )
            if function_part is not None:
                parts.append(function_part)
        return parts

    def _build_tool_response_part(
        self,
        types: Any,
        message: dict[str, Any],
        tool_name_by_call_id: dict[str, str] | None = None,
    ) -> Any | None:
        tool_name = str(message.get("name") or "")
        if not tool_name:
            tool_call_id = str(message.get("tool_call_id") or "")
            if tool_call_id and tool_name_by_call_id:
                tool_name = str(tool_name_by_call_id.get(tool_call_id) or "")
        if not tool_name:
            return None
        tool_response = _parse_json_object(message.get("content", ""), "response")
        return types.Part(
            function_response=types.FunctionResponse(
                id=str(message.get("tool_call_id") or ""),
                name=tool_name,
                response=tool_response,
            )
        )

    def _build_content_parts(self, types: Any, message: dict[str, Any]) -> list[Any]:
        parts: list[Any] = []
        content_text = str(message.get("content", "") or "")
        if content_text:
            parts.append(types.Part.from_text(text=content_text))
        if message["role"] == "assistant":
            parts.extend(self._build_assistant_tool_call_parts(types, message))
        return parts or [types.Part.from_text(text="")]

    def _flush_pending_tool_responses(
        self,
        *,
        types: Any,
        contents: list[Any],
        pending_parts: list[Any],
        active_tool_call_ids: list[str],
        pending_tool_response_ids: list[str],
    ) -> tuple[list[Any], list[str], list[str]]:
        if not pending_parts:
            return _reset_pending_tool_state()
        if active_tool_call_ids and pending_tool_response_ids != active_tool_call_ids:
            logger.warning(
                "Gemini: dropping incomplete tool response turn (expected=%s, got=%s)",
                active_tool_call_ids,
                pending_tool_response_ids,
            )
            return _reset_pending_tool_state()
        contents.append(types.Content(role="user", parts=pending_parts))
        return _reset_pending_tool_state()

    def _queue_tool_response_part(
        self,
        *,
        types: Any,
        message: dict[str, Any],
        active_tool_call_ids: list[str],
        pending_tool_response_parts: list[Any],
        pending_tool_response_ids: list[str],
        tool_name_by_call_id: dict[str, str],
    ) -> tuple[list[Any], list[str]]:
        tool_call_id = str(message.get("tool_call_id") or "")
        if not active_tool_call_ids or not tool_call_id or tool_call_id not in active_tool_call_ids:
            logger.warning(
                "Gemini: dropping orphan tool response (tool_call_id=%s, active_ids=%s)",
                tool_call_id or "missing",
                active_tool_call_ids,
            )
            return pending_tool_response_parts, pending_tool_response_ids
        if tool_call_id in pending_tool_response_ids:
            logger.warning(
                "Gemini: dropping duplicate tool response (tool_call_id=%s)",
                tool_call_id,
            )
            return pending_tool_response_parts, pending_tool_response_ids
        tool_response_part = self._build_tool_response_part(
            types,
            message,
            tool_name_by_call_id,
        )
        if tool_response_part is None:
            return pending_tool_response_parts, pending_tool_response_ids
        return (
            [*pending_tool_response_parts, tool_response_part],
            [*pending_tool_response_ids, tool_call_id],
        )

    def _prepare_contents_and_system(
        self, types: Any, messages: list[LLMMessage | dict]
    ) -> tuple[str | None, list[Any]]:
        normalized_messages = self._normalize_messages(messages)
        system_instruction = None
        contents: list[Any] = []
        pending_tool_response_parts: list[Any] = []
        tool_name_by_call_id: dict[str, str] = {}
        active_tool_call_ids: list[str] = []
        pending_tool_response_ids: list[str] = []

        for message in normalized_messages:
            role = message["role"]
            if role == "system":
                system_instruction = message["content"]
                continue
            if role == "assistant":
                (
                    pending_tool_response_parts,
                    pending_tool_response_ids,
                    active_tool_call_ids,
                ) = self._flush_pending_tool_responses(
                    types=types,
                    contents=contents,
                    pending_parts=pending_tool_response_parts,
                    active_tool_call_ids=active_tool_call_ids,
                    pending_tool_response_ids=pending_tool_response_ids,
                )
                active_tool_call_ids = _collect_active_tool_call_ids(
                    message,
                    tool_name_by_call_id,
                )
                parts = self._build_content_parts(types, message)
                _append_content_turn(types, contents, role="model", parts=parts)
                continue
            if role == "tool":
                pending_tool_response_parts, pending_tool_response_ids = (
                    self._queue_tool_response_part(
                        types=types,
                        message=message,
                        active_tool_call_ids=active_tool_call_ids,
                        pending_tool_response_parts=pending_tool_response_parts,
                        pending_tool_response_ids=pending_tool_response_ids,
                        tool_name_by_call_id=tool_name_by_call_id,
                    )
                )
                continue

            (
                pending_tool_response_parts,
                pending_tool_response_ids,
                active_tool_call_ids,
            ) = self._flush_pending_tool_responses(
                types=types,
                contents=contents,
                pending_parts=pending_tool_response_parts,
                active_tool_call_ids=active_tool_call_ids,
                pending_tool_response_ids=pending_tool_response_ids,
            )
            normalized_role = "user" if role == "user" else "model"
            parts = self._build_content_parts(types, message)
            _append_content_turn(types, contents, role=normalized_role, parts=parts)
        self._flush_pending_tool_responses(
            types=types,
            contents=contents,
            pending_parts=pending_tool_response_parts,
            active_tool_call_ids=active_tool_call_ids,
            pending_tool_response_ids=pending_tool_response_ids,
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
        # API contract note:
        # ``extra_kwargs`` comes from the shared chat pipeline and may include
        # OpenAI/OpenAI-compatible request fields. Gemini's
        # ``types.GenerateContentConfig`` has a strict schema (extra_forbidden),
        # so we must explicitly strip non-Gemini keys before constructing config.
        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)
        enable_reasoning = bool(extra_kwargs.pop("enable_reasoning", False))
        thinking_budget = extra_kwargs.pop("thinking_budget", None)
        if enable_reasoning or thinking_budget is not None:
            thinking_config_kwargs: dict[str, Any] = {}
            if enable_reasoning:
                thinking_config_kwargs["include_thoughts"] = True
            if thinking_budget is not None:
                thinking_config_kwargs["thinking_budget"] = thinking_budget
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_config_kwargs)
        extra_kwargs.pop("model", None)
        extra_kwargs.pop("parallel_tool_calls", None)
        extra_kwargs.pop("max_parallel_calls", None)
        extra_kwargs.pop("prompt_cache_key", None)
        # Shared final-stream logic may inject OpenAI stream usage controls.
        # Gemini config does not support these keys, and forwarding them causes
        # pydantic ``extra_forbidden`` validation errors at runtime.
        extra_kwargs.pop("include_stream_usage", None)
        extra_kwargs.pop("stream_options", None)
        tool_choice = extra_kwargs.pop("tool_choice", None)
        if tool_choice:
            config_kwargs["tool_config"] = self._convert_tool_choice(tool_choice)
        elif not tools:
            config_kwargs["tool_config"] = self._convert_tool_choice("none")
        config_kwargs.update(
            {key: value for key, value in extra_kwargs.items() if value is not None}
        )
        return types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _extract_text_from_parts(parts: list[Any]) -> str:
        return "".join(
            str(getattr(part, "text", "") or "") for part in parts if getattr(part, "text", None)
        )

    def _update_stream_usage(self, chunk: Any) -> None:
        usage_metadata = getattr(chunk, "usage_metadata", None)
        if not usage_metadata:
            return
        self.last_usage = _normalize_usage(
            {
                "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
                "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(usage_metadata, "total_token_count", None),
                "thinking_tokens": getattr(usage_metadata, "thoughts_token_count", None),
                "cache_read_input_tokens": getattr(
                    usage_metadata, "cached_content_token_count", None
                ),
                "prompt_tokens_details": getattr(usage_metadata, "prompt_tokens_details", None),
            }
        )

    def _resolve_stream_text(
        self,
        *,
        chunk: Any,
        candidate: Any,
        parts: list[Any],
        request_model: str,
    ) -> tuple[str, str]:
        text, reasoning = _extract_text_and_reasoning_from_parts(parts)
        has_function_call_parts = _has_function_call_parts(parts)
        if not text and candidate is not None:
            candidate_text = getattr(candidate, "text", None)
            if isinstance(candidate_text, str) and candidate_text:
                text = candidate_text
        if not text and not reasoning and not has_function_call_parts and candidate is not None:
            finish_message = getattr(candidate, "finish_message", None)
            if isinstance(finish_message, str) and finish_message:
                text = finish_message
        if not text and not has_function_call_parts:
            chunk_text = getattr(chunk, "text", None)
            if isinstance(chunk_text, str) and chunk_text:
                text = chunk_text
        if not text and not reasoning and not has_function_call_parts and parts:
            unknown_part_fields = _summarize_unknown_parts(parts)
            if unknown_part_fields:
                logger.warning(
                    "Gemini stream chunk had no visible text/function_call but included unmapped parts: model=%s fields=%s",
                    request_model,
                    unknown_part_fields,
                )
        return text, reasoning

    def _build_stream_tool_calls_delta(self, parts: list[Any]) -> list[dict[str, Any]]:
        tool_calls_delta: list[dict[str, Any]] = []
        for index, part in enumerate(parts):
            function_call = getattr(part, "function_call", None)
            if not function_call:
                continue
            function_name = getattr(function_call, "name", "")
            tool_call_id = self._resolve_tool_call_id(
                raw_id=getattr(function_call, "id", ""),
                function_name=function_name,
                index=index,
            )
            args = getattr(function_call, "args", None)
            if isinstance(args, dict):
                args = json.dumps(args)
            function_payload: dict[str, Any] = {
                "name": function_name,
                "arguments": args if isinstance(args, str) else str(args or ""),
            }
            thought_signature = _extract_thought_signature(part)
            if thought_signature:
                function_payload["thought_signature"] = thought_signature
            tool_calls_delta.append(
                {
                    "index": index,
                    "id": tool_call_id,
                    "type": "function",
                    "function": function_payload,
                }
            )
        return tool_calls_delta

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
                "Google GenAI client not installed. Install with: pip install google-genai"
            ) from exc

        system_instruction, contents = self._prepare_contents_and_system(types, messages)
        request_model = kwargs.pop("model", None) or self.model
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
                model=request_model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise self._wrap_client_error(exc) from exc
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
                "Google GenAI client not installed. Install with: pip install google-genai"
            ) from exc

        system_instruction, contents = self._prepare_contents_and_system(types, messages)
        request_model = kwargs.pop("model", None) or self.model
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
                model=request_model,
                contents=contents,
                config=config,
            )
            chunk_count = 0
            saw_visible_output = False
            async for chunk in stream:
                chunk_count += 1
                self._update_stream_usage(chunk)
                candidate, parts = _extract_chunk_candidate(chunk)
                finish_reason = _extract_stream_finish_reason(candidate)
                if finish_reason:
                    self.last_finish_reason = finish_reason
                text, reasoning = self._resolve_stream_text(
                    chunk=chunk,
                    candidate=candidate,
                    parts=parts,
                    request_model=request_model,
                )
                if text:
                    saw_visible_output = True
                    yield StreamChunk(content_delta=text)
                if reasoning:
                    saw_visible_output = True
                    yield StreamChunk(reasoning_delta=reasoning)
                tool_calls_delta = self._build_stream_tool_calls_delta(parts)
                if tool_calls_delta:
                    saw_visible_output = True
                    yield StreamChunk(tool_calls_delta=tool_calls_delta)

            self.last_finish_reason = self.last_finish_reason or "stop"
            if not saw_visible_output:
                logger.warning(
                    "Gemini stream completed without visible output: model=%s finish_reason=%s chunk_count=%d usage=%s",
                    request_model,
                    self.last_finish_reason,
                    chunk_count,
                    self.last_usage,
                )
        except Exception as exc:
            raise self._wrap_client_error(exc) from exc

    # ── Response normalization ────────────────────────────────────

    def _extract_candidate_tool_calls(self, parts: list[Any]) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            function_call = getattr(part, "function_call", None)
            if not function_call:
                continue
            function_name = getattr(function_call, "name", "")
            tool_call_id = self._resolve_tool_call_id(
                raw_id=getattr(function_call, "id", ""),
                function_name=function_name,
                index=len(tool_calls),
            )
            args = getattr(function_call, "args", None)
            if isinstance(args, dict):
                args = json.dumps(args)
            function_payload: dict[str, Any] = {
                "name": function_name,
                "arguments": args,
            }
            thought_signature = _extract_thought_signature(part)
            if thought_signature:
                function_payload["thought_signature"] = thought_signature
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": function_payload,
                }
            )
        return tool_calls

    def _extract_candidate_response_data(
        self,
        *,
        candidate: Any,
        content: str,
    ) -> tuple[str, str, list[dict[str, Any]], str]:
        finish_reason = _extract_stream_finish_reason(candidate) or "stop"
        candidate_content = getattr(candidate, "content", None)
        parts = getattr(candidate_content, "parts", None) or []
        part_text, part_reasoning = _extract_text_and_reasoning_from_parts(parts)
        if not content:
            content = self._extract_candidate_content(candidate) or part_text
        return (
            content,
            part_reasoning or "",
            self._extract_candidate_tool_calls(parts),
            finish_reason,
        )

    def _normalize_response(self, response: Any) -> LLMResponse:
        content = self._extract_response_content(response)
        tool_calls: list[dict[str, Any]] = []
        reasoning_content = ""
        finish_reason = "stop"

        if response.candidates:
            content, reasoning_content, tool_calls, finish_reason = (
                self._extract_candidate_response_data(
                    candidate=response.candidates[0],
                    content=content,
                )
            )

        usage_metadata = getattr(response, "usage_metadata", None)
        usage = _normalize_usage(
            {
                "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
                "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(usage_metadata, "total_token_count", None),
                "thinking_tokens": getattr(usage_metadata, "thoughts_token_count", None),
                "cache_read_input_tokens": getattr(
                    usage_metadata, "cached_content_token_count", None
                ),
                "prompt_tokens_details": getattr(usage_metadata, "prompt_tokens_details", None),
            }
        )

        self.last_usage = usage

        metadata: dict[str, Any] = {}
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
            metadata=metadata,
        )

    def _extract_response_content(self, response: Any) -> str:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, str):
                return parsed
            return json.dumps(parsed, ensure_ascii=False)
        if getattr(response, "candidates", None):
            return ""
        return str(getattr(response, "text", "") or "")

    def _extract_candidate_content(self, candidate: Any) -> str:
        candidate_text = getattr(candidate, "text", None)
        if candidate_text:
            return str(candidate_text)
        finish_message = getattr(candidate, "finish_message", None)
        if finish_message:
            return str(finish_message)
        candidate_content = getattr(candidate, "content", None)
        parts = getattr(candidate_content, "parts", None) or []
        text_parts, _ = _extract_text_and_reasoning_from_parts(parts)
        return text_parts

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool schema to Vertex function declarations."""
        return _convert_openai_tools_to_vertex(tools)

    def _convert_tool_choice(self, tool_choice: Any) -> dict[str, Any]:
        """Map OpenAI tool_choice to Vertex function calling config."""
        return _convert_openai_tool_choice_to_vertex(tool_choice)

    def _wrap_client_error(self, exc: Exception) -> Exception:
        """Wrap client errors with actionable diagnostic messages."""
        error_kind = _classify_client_error(exc)

        if error_kind == "rate_limit":
            return RuntimeError(
                f"Gemini API rate limited (auth_mode={self._auth_mode}, model={self.model}). "
                f"The google-genai client may retry transient 429s automatically, but this request still failed after retry. "
                f"Please retry in a moment or reduce request rate/concurrency."
            )

        # 400 location not supported — Developer API blocks VPN/proxy/datacenter IPs
        if error_kind == "region":
            if self._auth_mode == "developer_api":
                return RuntimeError(
                    f"Developer API: region not supported "
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
        if error_kind == "auth":
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
        if error_kind == "permission":
            return RuntimeError(
                f"Gemini API 403 (auth_mode={self._auth_mode}, model={self.model}). "
                f"Check: Vertex AI API enabled + aiplatform.user role granted."
            )

        # 404 NOT_FOUND
        if error_kind == "not_found":
            return RuntimeError(f"Gemini API 404 (model={self.model}). Check GEMINI_MODEL in .env.")

        # Pass through with context
        return RuntimeError(
            f"Gemini API error (auth_mode={self._auth_mode}, model={self.model}): {exc}"
        )
