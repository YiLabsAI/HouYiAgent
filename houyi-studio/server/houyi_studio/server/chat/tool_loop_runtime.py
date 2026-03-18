from __future__ import annotations

from typing import Any

from houyi.application.tool_calling.persisted_messages import (
    collect_persisted_tool_message_payloads,
)

from .sse_adapter import SSEEvent
from .types import Message, MessageRole


def build_tool_trace_events(
    *,
    tool_trace: list[dict[str, Any]] | None,
    assistant_message_id: str,
    trace_id: str,
) -> list[str]:
    entries = [entry for entry in (tool_trace or []) if isinstance(entry, dict)]
    event_chunks: list[str] = []
    round_indexes = sorted(
        {
            round_index
            for round_index in (entry.get("round_index") for entry in entries)
            if isinstance(round_index, int)
        }
    )
    for round_index in round_indexes:
        event_chunks.append(
            SSEEvent(
                event="agent.iteration",
                data={
                    "message_id": assistant_message_id,
                    "trace_id": trace_id,
                    "round_index": round_index,
                },
            ).encode()
        )

    for entry in entries:
        tool_call_id = entry.get("tool_call_id")
        tool_name = entry.get("tool_name")
        requested_tool_name = entry.get("requested_tool_name")
        parallel_group_id = entry.get("parallel_group_id")
        round_value = entry.get("round_index")
        duration_ms = entry.get("duration_ms")
        args = entry.get("args")
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        raw_result = result.get("raw") if isinstance(result, dict) else None

        event_chunks.append(
            SSEEvent(
                event="tool_call.start",
                data={
                    "message_id": assistant_message_id,
                    "trace_id": trace_id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "requested_tool_name": requested_tool_name,
                    "parallel_group_id": parallel_group_id,
                    "round_index": round_value,
                    "duration_ms": duration_ms,
                    "arguments": args,
                },
            ).encode()
        )

        if isinstance(raw_result, dict) and raw_result.get("error"):
            event_chunks.append(
                SSEEvent(
                    event="tool_call.error",
                    data={
                        "message_id": assistant_message_id,
                        "trace_id": trace_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "requested_tool_name": requested_tool_name,
                        "parallel_group_id": parallel_group_id,
                        "round_index": round_value,
                        "duration_ms": duration_ms,
                        "error": raw_result,
                    },
                ).encode()
            )
        else:
            event_chunks.append(
                SSEEvent(
                    event="tool_call.result",
                    data={
                        "message_id": assistant_message_id,
                        "trace_id": trace_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "requested_tool_name": requested_tool_name,
                        "parallel_group_id": parallel_group_id,
                        "round_index": round_value,
                        "duration_ms": duration_ms,
                        "result": raw_result,
                    },
                ).encode()
            )
    return event_chunks


def collect_persisted_tool_messages(
    *,
    intermediate_messages: list[dict[str, Any]],
    model: str | None = None,
    tool_result_max_tokens: int | None = None,
    per_tool_quota: dict[str, int] | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
) -> list[Message]:
    payloads = collect_persisted_tool_message_payloads(
        intermediate_messages=intermediate_messages,
        model=model,
        tool_result_max_tokens=tool_result_max_tokens,
        per_tool_quota=per_tool_quota,
        tool_trace=tool_trace,
    )
    persisted_tool_messages: list[Message] = []
    for payload in payloads:
        role = str(payload.get("role") or "")
        if role == MessageRole.ASSISTANT.value:
            persisted_tool_messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=str(payload.get("content") or ""),
                    reasoning_content=(
                        str(payload.get("reasoning_content"))
                        if isinstance(payload.get("reasoning_content"), str)
                        else None
                    ),
                    tool_calls=payload.get("tool_calls"),
                )
            )
            continue
        if role == MessageRole.TOOL.value:
            persisted_tool_messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=str(payload.get("content") or ""),
                    tool_call_id=(
                        str(payload.get("tool_call_id")) if payload.get("tool_call_id") else None
                    ),
                    name=(str(payload.get("name")) if payload.get("name") else None),
                    metadata=payload.get("metadata")
                    if isinstance(payload.get("metadata"), dict)
                    else {},
                )
            )
    return persisted_tool_messages
