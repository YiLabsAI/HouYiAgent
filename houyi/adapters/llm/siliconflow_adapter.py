"""SiliconFlow provider adapter.

Strategy: prefer the OpenAI client path and fall back to raw ``httpx`` SSE.
The client path provides retries, typed errors, streaming support, and usage tracking.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.adapters.llm.base import (
    DEFAULT_TEMPERATURE,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    _parse_bracket_tool_calls,
)
from houyi.adapters.llm.models import (
    DEEPSEEK_R1,
    DEEPSEEK_V3,
    DEEPSEEK_V3_2,
    KIMI_K2_5,
    normalize_model_id,
)
from houyi.adapters.llm.openai_compat_base import (
    OpenAICompatAdapterBase,
    _is_proxy_enabled,
    _normalize_transport_name,
)
from houyi.adapters.llm.request_models import OpenAICompatRequest
from houyi.adapters.llm.retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    RetryController,
    RetryPolicy,
)

logger = logging.getLogger(__name__)

_HTTPX_CHAT_CONNECT_TIMEOUT_SECONDS = 10.0
_HTTPX_CHAT_READ_TIMEOUT_SECONDS = 30.0
_HTTPX_CHAT_TOTAL_RETRIES = 1


def _is_siliconflow_deepseek_r1(model: str | None) -> bool:
    return normalize_model_id(model or "") == normalize_model_id(DEEPSEEK_R1)


def _is_deepseek_tool_model(model: str | None) -> bool:
    normalized = normalize_model_id(model or "")
    return normalized in {
        normalize_model_id(DEEPSEEK_R1),
        normalize_model_id(DEEPSEEK_V3),
        normalize_model_id(DEEPSEEK_V3_2),
    }


def _looks_like_balance_error(error_text: str) -> bool:
    """Match SiliconFlow balance-exhaustion failures across inconsistent error shapes.

    SiliconFlow has returned both plain-text "INSUFFICIENT BALANCE" messages and
    JSON fragments containing provider code ``30001``. We keep this provider-
    specific detector narrow so retry/fallback logic can classify quota failures
    consistently until error normalization is centralized across adapters.
    """
    normalized = str(error_text or "").upper()
    return "INSUFFICIENT BALANCE" in normalized or '"CODE":30001' in normalized


def _project_function_payload(function_payload: Any) -> dict[str, Any] | None:
    if not isinstance(function_payload, dict):
        return None
    projected_function: dict[str, Any] = {}
    if function_payload.get("name") is not None:
        projected_function["name"] = str(function_payload.get("name"))
    arguments = function_payload.get("arguments")
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False)
        except TypeError:
            arguments = str(arguments)
    if arguments is not None:
        projected_function["arguments"] = arguments
    return projected_function or None


def _project_tool_call(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    projected_call: dict[str, Any] = {}
    if call.get("id") is not None:
        projected_call["id"] = str(call.get("id"))
    if call.get("type") is not None:
        projected_call["type"] = str(call.get("type"))
    projected_function = _project_function_payload(call.get("function"))
    if projected_function is not None:
        projected_call["function"] = projected_function
    return projected_call or None


def _project_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    projected: list[dict[str, Any]] = []
    for call in tool_calls:
        projected_call = _project_tool_call(call)
        if projected_call is not None:
            projected.append(projected_call)
    return projected


def _project_tools(tools: Any) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return projected
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if str(tool.get("type") or "") != "function":
            continue
        function_payload = tool.get("function")
        if not isinstance(function_payload, dict):
            continue
        projected_function: dict[str, Any] = {}
        if function_payload.get("name") is not None:
            projected_function["name"] = str(function_payload.get("name"))
        if function_payload.get("description") is not None:
            projected_function["description"] = str(function_payload.get("description"))
        parameters = function_payload.get("parameters")
        if isinstance(parameters, dict):
            projected_function["parameters"] = dict(parameters)
        if not projected_function:
            continue
        projected.append({"type": "function", "function": projected_function})
    return projected


def _build_assistant_tool_message(
    message: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    *,
    include_content: bool,
    include_reasoning_content: bool,
) -> dict[str, Any]:
    projected_message: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    if include_content:
        content = message.get("content")
        if content is not None:
            content_text = str(content)
            if content_text:
                projected_message["content"] = content_text
    if include_reasoning_content:
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None:
            reasoning_text = str(reasoning_content)
            if reasoning_text:
                projected_message["reasoning_content"] = reasoning_text
    return projected_message


def _build_assistant_message(
    message: dict[str, Any],
    *,
    include_reasoning_content: bool,
) -> dict[str, Any]:
    projected_message: dict[str, Any] = {
        "role": "assistant",
        "content": str(message.get("content") or ""),
    }
    if include_reasoning_content:
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None:
            reasoning_text = str(reasoning_content)
            if reasoning_text:
                projected_message["reasoning_content"] = reasoning_text
    return projected_message


def _stringify_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    except TypeError:
        return str(arguments if arguments is not None else {})


def _build_r1_replay_assistant_message(
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    lines = ["Tool calls requested:"]
    for index, call in enumerate(tool_calls, start=1):
        function_payload = call.get("function") if isinstance(call, dict) else None
        function_name = "tool"
        function_arguments: Any = {}
        if isinstance(function_payload, dict):
            if function_payload.get("name") is not None:
                function_name = str(function_payload.get("name"))
            function_arguments = function_payload.get("arguments")
        lines.append(f"{index}. {function_name} {_stringify_tool_arguments(function_arguments)}")
    return {
        "role": "assistant",
        "content": "\n".join(lines),
    }


def _build_r1_replay_tool_message(message: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(message.get("name") or "").strip()
    tool_call_id = str(message.get("tool_call_id") or "").strip()
    if tool_name and tool_call_id:
        header = f"Tool result for {tool_name} ({tool_call_id})"
    elif tool_name:
        header = f"Tool result for {tool_name}"
    elif tool_call_id:
        header = f"Tool result for {tool_call_id}"
    else:
        header = "Tool result"
    content = _stringify_minimal_tool_content(message.get("content"))
    return {
        "role": "user",
        "content": f"{header}:\n{content}" if content else f"{header}: [no output]",
    }


def _parse_embedded_json_string(content: str) -> Any | None:
    stripped = content.strip()
    if not stripped:
        return None
    with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
        return json.loads(stripped)
    return None


def _extract_minimal_dict_string(content: dict[str, Any]) -> str | None:
    for key in ("content", "message", "result", "data"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("content", "message", "result", "data"):
        value = content.get(key)
        if isinstance(value, (dict, list)):
            serialized = _stringify_minimal_tool_content(value)
            if serialized:
                return serialized
    return None


def _stringify_minimal_tool_content(content: Any) -> str:
    if isinstance(content, str):
        parsed = _parse_embedded_json_string(content)
        if parsed is not None:
            minimized = _stringify_minimal_tool_content(parsed)
            if minimized:
                return minimized
        return content
    if isinstance(content, dict):
        extracted = _extract_minimal_dict_string(content)
        if extracted:
            return extracted
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        if len(content) == 1:
            return _stringify_minimal_tool_content(content[0])
        return json.dumps(content, ensure_ascii=False)
    return str(content or "")


def _project_plain_message(normalized: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": str(normalized.get("content") or ""),
    }


def _project_r1_message(normalized: dict[str, Any], role: str) -> dict[str, Any]:
    if role == "assistant":
        projected_tool_calls = _project_tool_calls(normalized.get("tool_calls"))
        if projected_tool_calls:
            return _build_r1_replay_assistant_message(projected_tool_calls)
        return _build_assistant_message(
            normalized,
            include_reasoning_content=False,
        )
    if role == "tool":
        return _build_r1_replay_tool_message(normalized)
    return _project_plain_message(normalized, role)


def _project_standard_tool_message(normalized: dict[str, Any]) -> dict[str, Any]:
    tool_message = {
        "role": "tool",
        "content": _stringify_minimal_tool_content(normalized.get("content")),
    }
    tool_call_id = normalized.get("tool_call_id")
    if tool_call_id is not None:
        tool_message["tool_call_id"] = str(tool_call_id)
    return tool_message


def _project_standard_message(
    normalized: dict[str, Any],
    role: str,
    *,
    preserve_interleaved_reasoning: bool,
) -> dict[str, Any]:
    if role == "assistant":
        if isinstance(normalized.get("tool_calls"), list):
            projected_tool_calls = _project_tool_calls(normalized.get("tool_calls"))
            return _build_assistant_tool_message(
                normalized,
                projected_tool_calls,
                include_content=preserve_interleaved_reasoning,
                include_reasoning_content=preserve_interleaved_reasoning,
            )
        return _build_assistant_message(
            normalized,
            include_reasoning_content=preserve_interleaved_reasoning,
        )
    if role == "tool":
        return _project_standard_tool_message(normalized)
    return _project_plain_message(normalized, role)


def _project_messages(
    messages: list[dict[str, Any]],
    *,
    model: str | None,
) -> list[dict[str, Any]]:
    projected_messages: list[dict[str, Any]] = []
    is_r1 = _is_siliconflow_deepseek_r1(model)
    is_deepseek_tool_model = _is_deepseek_tool_model(model)
    # SiliconFlow DeepSeek (notably V3/V3.2) may reject OpenAI-style follow-up
    # tool transcripts that include assistant.tool_calls + tool role messages with
    # provider error `code=20015` (`"messages" in request are illegal.`).
    #
    # We keep first-round user/system messages in standard projection, but when
    # a tool transcript is present we convert assistant/tool turns into the same
    # replay-style shape used by R1. This preserves multi-round tool-loop ability
    # while keeping provider-specific behavior isolated to this adapter.
    #
    # If SiliconFlow later accepts standard OpenAI transcripts consistently, this
    # branch can be narrowed/removed with adapter regression updates.
    has_tool_transcript = any(str(message.get("role") or "") == "tool" for message in messages)
    preserve_interleaved_reasoning = not is_r1
    for message in messages:
        normalized = dict(message)
        role = str(normalized.get("role") or "")
        projected_messages.append(
            _project_r1_message(normalized, role)
            if is_r1
            or (is_deepseek_tool_model and has_tool_transcript and role in {"assistant", "tool"})
            else _project_standard_message(
                normalized,
                role,
                preserve_interleaved_reasoning=preserve_interleaved_reasoning,
            )
        )
    return projected_messages


def _summarize_payload_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    summary: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            summary.append({"index": index, "type": type(message).__name__})
            continue
        tool_calls = message.get("tool_calls")
        summary.append(
            {
                "index": index,
                "role": message.get("role"),
                "keys": sorted(message.keys()),
                "content_len": len(str(message.get("content") or "")),
                "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
                "tool_call_ids": [
                    str(call.get("id"))
                    for call in tool_calls
                    if isinstance(call, dict) and call.get("id") is not None
                ]
                if isinstance(tool_calls, list)
                else [],
                "tool_call_id": message.get("tool_call_id"),
            }
        )
    return summary


def _payload_chars(payload: dict[str, Any]) -> int:
    total = 0
    messages = payload.get("messages") or []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += len(json.dumps(content, ensure_ascii=False, default=str))
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                total += len(json.dumps(tool_calls, ensure_ascii=False, default=str))
    tools = payload.get("tools")
    if isinstance(tools, list):
        total += len(json.dumps(tools, ensure_ascii=False, default=str))
    return total


def _format_siliconflow_http_error(status_code: int, error_text: str = "") -> str:
    if status_code == 403 and _looks_like_balance_error(error_text):
        return (
            "SiliconFlow rejected the request because the configured account has "
            "insufficient balance or credits. Check the provider billing status "
            "or switch to another available model before retrying."
        )
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
        self.default_model = default_model or _env.siliconflow_model
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
                logger.debug("openai client available — using client mode")
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
        return sanitized

    def _resolve_transport(self, request: OpenAICompatRequest) -> str:
        request_transport = _normalize_transport_name(request.transport)
        if request_transport:
            logger.debug(
                "SiliconFlow transport resolved: model=%s transport=%s source=request enable_streaming=%s",
                request.model,
                request_transport,
                request.enable_streaming,
            )
            return request_transport
        route = _normalize_transport_name(os.getenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "auto"))
        if route:
            logger.debug(
                "SiliconFlow transport resolved: model=%s transport=%s source=env enable_streaming=%s",
                request.model,
                route,
                request.enable_streaming,
            )
            return route
        if request.enable_streaming:
            logger.debug(
                "SiliconFlow transport resolved: model=%s transport=httpx source=streaming enable_streaming=%s",
                request.model,
                request.enable_streaming,
            )
            return "httpx"
        if _is_deepseek_tool_model(request.model) and SiliconFlowAdapter._OPENAI_READY:
            logger.debug(
                "SiliconFlow transport resolved: model=%s transport=client source=deepseek_tool_model enable_streaming=%s",
                request.model,
                request.enable_streaming,
            )
            return "client"
        if SiliconFlowAdapter._OPENAI_READY:
            logger.debug(
                "SiliconFlow transport resolved: model=%s transport=client source=openai_ready enable_streaming=%s",
                request.model,
                request.enable_streaming,
            )
            return "client"
        logger.debug(
            "SiliconFlow transport resolved: model=%s transport=httpx source=fallback enable_streaming=%s",
            request.model,
            request.enable_streaming,
        )
        return "httpx"

    @classmethod
    def _prepare_messages_for_request(
        cls,
        request: OpenAICompatRequest,
    ) -> list[dict[str, Any]]:
        return [dict(message) for message in request.messages]

    def _prepare_request_for_provider(
        self,
        request: OpenAICompatRequest,
    ) -> OpenAICompatRequest:
        prepared_messages = self._prepare_messages_for_request(request)
        projected_tools = request.tools
        extra_kwargs = dict(request.extra_kwargs)
        tool_choice = request.tool_choice
        normalized_model = normalize_model_id(request.model or "")
        if _is_deepseek_tool_model(request.model):
            # SiliconFlow DeepSeek tool-loop first rounds are sensitive to extra
            # OpenAI-compatible fields and expanded tool schemas. Keep this
            # provider-specific shaping explicit here for now so the outbound
            # payload stays as close as possible to SiliconFlow's supported
            # function-calling subset; if more providers need the same behavior,
            # this should be generalized in the shared OpenAI-compatible layer.
            projected_tools = _project_tools(request.tools)
            prepared_messages = _project_messages(
                prepared_messages,
                model=request.model,
            )
            extra_kwargs.pop("parallel_tool_calls", None)
            extra_kwargs.pop("max_parallel_calls", None)
            extra_kwargs.pop("prompt_cache_key", None)
            tool_choice = None
        elif normalized_model == normalize_model_id(KIMI_K2_5) and tool_choice == "required":
            # SiliconFlow currently rejects tool_choice="required" for Kimi with
            # a provider 400. Keep this provider-local compatibility guard here
            # until request capability normalization is centralized.
            tool_choice = None
        prepared_request = self._copy_request(
            request,
            messages=prepared_messages,
            tools=projected_tools,
        )
        prepared_request.extra_kwargs = extra_kwargs
        prepared_request.tool_choice = tool_choice
        if _is_deepseek_tool_model(request.model):
            logger.debug(
                "SiliconFlow DeepSeek prepared payload summary: model=%s tool_choice=%s extra_keys=%s messages=%s",
                prepared_request.model,
                prepared_request.tool_choice,
                sorted(prepared_request.extra_kwargs.keys()),
                _summarize_payload_messages(
                    {
                        "messages": prepared_request.messages,
                    }
                ),
            )
        return prepared_request

    def _build_reasoning_extra_body(
        self,
        request: OpenAICompatRequest,
    ) -> dict[str, object] | None:
        if not request.enable_thinking or request.thinking_budget is None:
            return None
        logger.debug("Thinking enabled with thinking_budget=%d", request.thinking_budget)
        return {"thinking_budget": request.thinking_budget}

    _CLIENT_CONNECT_TIMEOUT = 10.0
    _CLIENT_READ_TIMEOUT = 240.0
    _CLIENT_MAX_RETRIES = 1

    def _new_client(self) -> Any:
        import httpx
        from openai import AsyncOpenAI

        http_client = httpx.AsyncClient(
            proxy=self._get_httpx_proxy(),
            timeout=httpx.Timeout(
                self._CLIENT_READ_TIMEOUT,
                connect=self._CLIENT_CONNECT_TIMEOUT,
                read=self._CLIENT_READ_TIMEOUT,
            ),
            trust_env=False,
        )
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self._CLIENT_READ_TIMEOUT,
            max_retries=self._CLIENT_MAX_RETRIES,
            http_client=http_client,
        )

    async def _close_client(self, client: Any) -> None:
        await client.close()

    async def _create_chat_response(
        self,
        *,
        request: OpenAICompatRequest,
        client: Any,
    ) -> Any:
        params = self._encode_chat_request(request)
        if _is_deepseek_tool_model(request.model):
            logger.debug(
                "SiliconFlow DeepSeek client create kwargs summary: model=%s tool_choice=%s extra_keys=%s messages=%s",
                params.get("model"),
                params.get("tool_choice"),
                sorted(key for key in params if key != "messages"),
                _summarize_payload_messages(params),
            )
        try:
            return await client.chat.completions.create(**params)
        except Exception:
            if _is_deepseek_tool_model(request.model):
                logger.error(
                    "SiliconFlow DeepSeek client create failed: model=%s tool_choice=%s extra_keys=%s messages=%s",
                    params.get("model"),
                    params.get("tool_choice"),
                    sorted(key for key in params if key != "messages"),
                    _summarize_payload_messages(params),
                    exc_info=True,
                )
            raise

    def _get_httpx_proxy(self) -> str | None:
        if not _is_proxy_enabled():
            return None
        from houyi.infrastructure.net.proxy import detect_proxy

        return detect_proxy()

    def _get_httpx_chat_timeout(self) -> Any:
        import httpx

        return httpx.Timeout(
            _HTTPX_CHAT_READ_TIMEOUT_SECONDS,
            connect=_HTTPX_CHAT_CONNECT_TIMEOUT_SECONDS,
            read=_HTTPX_CHAT_READ_TIMEOUT_SECONDS,
        )

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
                "SiliconFlow chat transport error: model=%s bucket=%s retry=%d/%d wait=%.2fs payload_chars=%s messages=%s error_type=%s error=%r",
                getattr(request, "model", None),
                decision.bucket,
                retry_controller.retries_used,
                retry_controller.policy.total_retries,
                decision.delay_seconds,
                _payload_chars(payload or {}),
                _summarize_payload_messages(payload or {}),
                type(exc).__name__,
                exc,
            )
        return decision.retry, decision.delay_seconds

    async def _handle_httpx_status(
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
    def _parse_httpx_response(response: Any) -> dict[str, Any]:
        if response.status_code >= 400:
            error_text = getattr(response, "text", "") or ""
            raise RuntimeError(
                _format_siliconflow_http_error(int(response.status_code), error_text)
            )
        return response.json()

    async def _chat_request_httpx(self, request: OpenAICompatRequest) -> LLMResponse:
        payload = self._normalize_httpx_payload(self._encode_chat_request_for_httpx(request))
        if _is_deepseek_tool_model(request.model):
            logger.debug(
                "SiliconFlow DeepSeek httpx request summary: model=%s tool_choice=%s payload_chars=%s extra_keys=%s messages=%s",
                payload.get("model"),
                payload.get("tool_choice"),
                _payload_chars(payload),
                sorted(key for key in payload if key != "messages"),
                _summarize_payload_messages(payload),
            )
        response = await self._execute_chat_httpx(payload, request=request)
        if response.status_code >= 400 and _is_deepseek_tool_model(request.model):
            # Keep 400 payload summaries for all SiliconFlow DeepSeek tool-loop
            # models while first-round compatibility issues are still being
            # narrowed down; V3.2 has shown the same failure family as R1.
            logger.error(
                "SiliconFlow DeepSeek httpx 400 payload summary: model=%s tool_choice=%s extra_keys=%s messages=%s raw_error=%s",
                payload.get("model"),
                payload.get("tool_choice"),
                sorted(key for key in payload if key != "messages"),
                _summarize_payload_messages(payload),
                (getattr(response, "text", "") or "")[:1000],
            )
        result = LLMResponse.from_raw_dict(
            self._parse_httpx_response(response),
            model_fallback=request.model,
        )
        if (
            normalize_model_id(request.model or "") == normalize_model_id(DEEPSEEK_V3_2)
            and not result.tool_calls
            and result.metadata.get("response_shape") == "empty_choices"
            and isinstance(request.tools, list)
            and len(request.tools) == 1
        ):
            function_payload = (
                request.tools[0].get("function") if isinstance(request.tools[0], dict) else None
            )
            function_name = (
                str(function_payload.get("name") or "").strip()
                if isinstance(function_payload, dict)
                else ""
            )
            if function_name:
                result.tool_calls = _parse_bracket_tool_calls(f"[tool:{function_name}]")
                if result.tool_calls:
                    result.finish_reason = "stop"
                    result.metadata.pop("response_shape", None)
        self.last_usage = result.usage
        self.last_finish_reason = result.finish_reason
        return result

    def _get_httpx_retry_controller(self) -> RetryController:
        return self._new_retry_controller(status_only=True)

    @staticmethod
    def _new_retry_controller(*, status_only: bool) -> RetryController:
        max_retries = _HTTPX_CHAT_TOTAL_RETRIES if status_only else DEFAULT_MAX_RETRIES
        return RetryController(
            RetryPolicy(
                total_retries=max_retries,
                connect_retries=None if status_only else DEFAULT_MAX_RETRIES,
                read_retries=None if status_only else DEFAULT_MAX_RETRIES,
                status_retries=max_retries,
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
            logger.debug("Using mock streaming (no API key)")
            last_content = ""
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    last_content = str(msg.get("content") or "")
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
        raise RuntimeError(_format_siliconflow_http_error(int(response.status_code), error_text))

    def _stream_retry(self) -> RetryController:
        return self._new_retry_controller(status_only=False)
