from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    raw: dict[str, Any]
    content: str = ""
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallTrace:
    tool_name: str | None = None
    requested_tool_name: str | None = None
    tool_call_id: str | None = None
    parallel_group_id: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    result: ToolResult = field(default_factory=lambda: ToolResult(raw={}))
    tool_override: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolError:
    tool_name: str | None = None
    requested_tool_name: str | None = None
    tool_call_id: str | None = None
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCallOutputPayload:
    type: str = "llm_response"
    content: str = ""
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    tool_errors: list[ToolError] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssembleResult:
    content: str
    output_payload: ToolCallOutputPayload
    messages_for_log: str | list[dict[str, Any]] = ""
    tool_call_rounds: int = 0
    normalized_tool_trace: list[ToolCallTrace] = field(default_factory=list)
    tool_errors: list[ToolError] = field(default_factory=list)


def _json_dumps_stable(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )


def build_llm_cache_key(
    *,
    adapter: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_kwargs: dict[str, Any] | None,
) -> str:
    """Build a deterministic cache key for an LLM chat call.

    The goal is stability rather than perfect uniqueness.
    """

    normalized_kwargs = dict(chat_kwargs or {})
    normalized_kwargs.pop("prompt_cache_key", None)

    payload = {
        "adapter": {
            "model": getattr(adapter, "model", None),
            "base_url": getattr(adapter, "base_url", None),
        },
        "messages": messages,
        "tools": tools,
        "chat_kwargs": normalized_kwargs,
    }

    digest = hashlib.sha256(_json_dumps_stable(payload).encode("utf-8")).hexdigest()
    return f"llm:{digest}"


def _normalize_tool_trace(
    tool_trace: list[dict[str, Any]],
) -> tuple[list[ToolCallTrace], list[ToolError]]:
    calls: list[ToolCallTrace] = []
    errors: list[ToolError] = []

    for entry in tool_trace:
        if not isinstance(entry, dict):
            continue

        result_payload = entry.get("result")
        if not isinstance(result_payload, dict):
            result_payload = {}

        raw_payload = result_payload.get("raw")
        if isinstance(raw_payload, dict):
            raw_dict = raw_payload
        else:
            raw_dict = {"result": raw_payload}

        is_error = bool(result_payload.get("is_error"))
        metadata = result_payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        call = ToolCallTrace(
            tool_name=entry.get("tool_name"),
            requested_tool_name=entry.get("requested_tool_name"),
            tool_call_id=entry.get("tool_call_id"),
            parallel_group_id=entry.get("parallel_group_id"),
            args=entry.get("args") if isinstance(entry.get("args"), dict) else {},  # type: ignore[arg-type]
            result=ToolResult(raw=raw_dict, is_error=is_error, metadata=metadata),
            tool_override=entry.get("tool_override")
            if isinstance(entry.get("tool_override"), dict)
            else None,
        )
        calls.append(call)

        if is_error:
            errors.append(
                ToolError(
                    tool_name=call.tool_name,
                    requested_tool_name=call.requested_tool_name,
                    tool_call_id=call.tool_call_id,
                    error=raw_dict,
                )
            )

    return calls, errors


async def assemble_tool_call_output(
    *,
    session_id: str,
    execution: Any,
    node_id: str,
    node_exec: Any,
    messages: list[dict[str, Any]],
    response: Any,
    tool_trace: list[dict[str, Any]],
    base_adapter: Any,
    tool_model: str | None,
    prompt: str | None,
    user_content: str | None,
    max_tool_calls: int,
    skills: list[Any],
    final_chat_kwargs: dict[str, Any] | None,
    prompt_cache_key: str | None,
    llm_cache: dict[str, Any] | None,
) -> AssembleResult:
    """Assemble the final LLM output payload after tool execution.

    This is a lightweight, side-effect-minimized helper used by unit tests.
    """

    normalized_calls, normalized_errors = _normalize_tool_trace(tool_trace)
    tool_call_rounds = len([message for message in messages if message.get("tool_calls")])
    messages_for_log: str | list[dict[str, Any]] = messages

    if not normalized_calls:
        payload = ToolCallOutputPayload(
            content=getattr(response, "content", ""),
            tool_calls=[],
            tool_errors=[],
            metadata={},
        )
        return AssembleResult(
            content=payload.content,
            output_payload=payload,
            messages_for_log=messages_for_log,
            tool_call_rounds=tool_call_rounds,
            normalized_tool_trace=[],
            tool_errors=[],
        )

    metadata: dict[str, Any] = {}
    execution_id = getattr(execution, "execution_id", None)
    if isinstance(execution_id, str) and execution_id:
        metadata["trace_id"] = execution_id

    cached_response: Any | None = None
    cache_key: str | None = None

    if llm_cache is not None and final_chat_kwargs is not None:
        effective_kwargs = dict(final_chat_kwargs)
        effective_kwargs.pop("prompt_cache_key", None)
        cache_key = build_llm_cache_key(
            adapter=base_adapter,
            messages=messages,
            tools=None,
            chat_kwargs=effective_kwargs,
        )
        cached_response = llm_cache.get(cache_key)

    if cached_response is not None:
        metadata["llm_cache_hit"] = True
        metadata["llm_cache_key"] = cache_key
        final_response = (
            cached_response.model_copy(deep=True)
            if hasattr(cached_response, "model_copy")
            else cached_response
        )
    else:
        final_response = response
        if final_chat_kwargs is not None:
            final_response = await base_adapter.chat(messages, tools=None, **final_chat_kwargs)
            if llm_cache is not None:
                cache_key = cache_key or build_llm_cache_key(
                    adapter=base_adapter,
                    messages=messages,
                    tools=None,
                    chat_kwargs={
                        k: v for k, v in final_chat_kwargs.items() if k != "prompt_cache_key"
                    },
                )
                llm_cache[cache_key] = (
                    final_response.model_copy(deep=True)
                    if hasattr(final_response, "model_copy")
                    else final_response
                )

    content = getattr(final_response, "content", "")
    usage_payload = getattr(final_response, "usage", None)
    if usage_payload is None:
        response_metadata = getattr(final_response, "metadata", None)
        if isinstance(response_metadata, dict):
            usage_payload = response_metadata.get("usage")
    if isinstance(usage_payload, dict):
        metadata["usage"] = usage_payload
    payload = ToolCallOutputPayload(
        content=content,
        tool_calls=normalized_calls,
        tool_errors=normalized_errors,
        metadata=metadata,
    )
    return AssembleResult(
        content=content,
        output_payload=payload,
        messages_for_log=messages_for_log,
        tool_call_rounds=tool_call_rounds,
        normalized_tool_trace=normalized_calls,
        tool_errors=normalized_errors,
    )
