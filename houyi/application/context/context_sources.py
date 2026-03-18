from __future__ import annotations

from collections.abc import Callable
from typing import Any

from houyi.application.context.types import ContextBlockType, ContextCandidate, ContextSourceKind

_PINNED_CONTEXT_PRIORITY = 5


def extract_latest_compaction_summary(
    metadata: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any]]:
    history = metadata.get("compaction_history") if isinstance(metadata, dict) else None
    if not isinstance(history, list):
        return None, {}
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        if not summary:
            continue
        item_metadata = item.get("metadata")
        return summary, {
            "compaction_id": item.get("compaction_id"),
            "backup_id": item.get("backup_id"),
            "trigger": item.get("trigger"),
            "pressure_level": item.get("pressure_level"),
            "created_at": item.get("created_at"),
            "source_message_ids": item.get("source_message_ids"),
            "summarization_mode": item_metadata.get("summarization_mode")
            if isinstance(item_metadata, dict)
            else None,
            "summary_model": item_metadata.get("summary_model")
            if isinstance(item_metadata, dict)
            else None,
        }
    return None, {}


def assemble_context_candidates(
    *,
    messages: list[dict[str, Any]],
    system_instructions: str,
    memory_context: str | None,
    summary_context: str | None = None,
    summary_metadata: dict[str, Any] | None = None,
    boundary_id: str | None = None,
) -> list[ContextCandidate]:
    boundary_metadata = _boundary_metadata(boundary_id)
    candidates: list[ContextCandidate] = []
    if system_instructions:
        candidates.append(
            ContextCandidate(
                source=ContextSourceKind.SYSTEM,
                block_type=ContextBlockType.SYSTEM,
                content=system_instructions,
                pinned=True,
                priority=0,
            )
        )
    if memory_context:
        candidates.append(
            ContextCandidate(
                source=ContextSourceKind.MEMORY,
                block_type=ContextBlockType.MEMORY,
                content=memory_context,
                priority=150,
            )
        )
    if summary_context:
        candidates.append(
            ContextCandidate(
                source=ContextSourceKind.SUMMARY,
                block_type=ContextBlockType.SUMMARY,
                content=summary_context,
                priority=200,
                metadata=dict(summary_metadata or {}),
            )
        )
    if messages:
        latest = messages[-1:]
        earlier_recent = messages[:-1]
        candidates.append(
            ContextCandidate(
                source=ContextSourceKind.CURRENT_TURN,
                block_type=ContextBlockType.RECENT,
                content=latest,
                pinned=True,
                priority=10,
                metadata={
                    "message_count": len(latest),
                    **boundary_metadata,
                },
            )
        )
        if earlier_recent:
            candidates.extend(
                _build_recent_candidates(
                    earlier_recent,
                    boundary_metadata=boundary_metadata,
                )
            )
    return candidates


def build_pinned_context_candidates(
    metadata: dict[str, Any] | None,
) -> list[ContextCandidate]:
    raw = metadata.get("pinned_contexts") if isinstance(metadata, dict) else None
    if not isinstance(raw, list):
        return []
    candidates: list[ContextCandidate] = []
    for item in raw:
        candidate = _build_pinned_candidate(item)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def build_tool_summary_candidates(
    messages: list[Any],
    *,
    boundary_id: str | None = None,
) -> tuple[list[ContextCandidate], set[str]]:
    if not isinstance(messages, list):
        return [], set()
    boundary_metadata = _boundary_metadata(boundary_id)
    candidates: list[ContextCandidate] = []
    summarized_tool_ids: set[str] = set()
    for index, message in enumerate(messages):
        if _message_role_value(message) != "tool":
            continue
        metadata = _message_field(message, "metadata")
        profile = metadata.get("tool_result_profile") if isinstance(metadata, dict) else None
        if not isinstance(profile, dict) or profile.get("compressed") is not True:
            continue
        summary = str(profile.get("summary") or "").strip()
        if not summary:
            continue
        tool_name = str(_message_field(message, "name") or "tool").strip() or "tool"
        tool_call_id = _message_field(message, "tool_call_id")
        message_id = _message_field(message, "message_id")
        candidates.append(
            ContextCandidate(
                source=ContextSourceKind.TOOL_SUMMARY,
                block_type=ContextBlockType.TOOL_SUMMARY,
                content=f"{tool_name}: {summary}",
                priority=140,
                metadata={
                    "tool_call_id": tool_call_id,
                    "source_message_id": message_id,
                    "recent_start_index": index,
                    **boundary_metadata,
                },
            )
        )
        if isinstance(tool_call_id, str) and tool_call_id:
            summarized_tool_ids.add(tool_call_id)
        elif isinstance(message_id, str) and message_id:
            summarized_tool_ids.add(message_id)
    return candidates, summarized_tool_ids


def build_history_message_payloads(
    messages: list[Any],
    *,
    message_to_payload: Callable[[Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    _, summarized_tool_ids = build_tool_summary_candidates(messages)
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if _message_role_value(message) == "system":
            continue
        if is_summarized_tool_message(message, summarized_tool_ids):
            continue
        payloads.append(message_to_payload(message))
    return payloads


def is_summarized_tool_message(message: Any, summarized_tool_ids: set[str]) -> bool:
    if _message_role_value(message) != "tool":
        return False
    tool_call_id = _message_field(message, "tool_call_id")
    message_id = _message_field(message, "message_id")
    return (isinstance(tool_call_id, str) and tool_call_id in summarized_tool_ids) or (
        isinstance(message_id, str) and message_id in summarized_tool_ids
    )


def _split_recent_chunks(
    messages: list[dict[str, Any]],
) -> list[tuple[int, list[dict[str, Any]], int]]:
    if not messages:
        return []
    chunks: list[tuple[int, list[dict[str, Any]], int]] = []
    current_chunk: list[dict[str, Any]] = []
    current_priority: int | None = None
    chunk_start = 0
    for index, message in enumerate(messages):
        priority = 100 if str(message.get("role") or "") == "user" else 160
        if current_priority is None:
            current_priority = priority
            chunk_start = index
        if current_chunk and priority != current_priority:
            chunks.append((chunk_start, current_chunk, current_priority))
            current_chunk = []
            current_priority = priority
            chunk_start = index
        current_chunk.append(message)
    if current_chunk and current_priority is not None:
        chunks.append((chunk_start, current_chunk, current_priority))
    return chunks


def _build_recent_candidates(
    messages: list[dict[str, Any]],
    *,
    boundary_metadata: dict[str, Any],
) -> list[ContextCandidate]:
    candidates: list[ContextCandidate] = []
    for start_index, chunk, priority in _split_recent_chunks(messages):
        candidates.append(
            ContextCandidate(
                source=ContextSourceKind.RECENT,
                block_type=ContextBlockType.RECENT,
                content=chunk,
                priority=priority,
                metadata={
                    "message_count": len(chunk),
                    "recent_start_index": start_index,
                    **boundary_metadata,
                },
            )
        )
    return candidates


def _build_pinned_candidate(item: Any) -> ContextCandidate | None:
    if not isinstance(item, dict):
        return None
    if str(item.get("status") or "").strip().lower() != "active":
        return None
    content = str(item.get("content") or "")
    if not content:
        return None
    pin_metadata = item.get("metadata")
    return ContextCandidate(
        source=ContextSourceKind.PINNED,
        block_type=ContextBlockType.PINNED,
        content=content,
        pinned=True,
        priority=_PINNED_CONTEXT_PRIORITY,
        token_count=_to_non_negative_int(item.get("token_count")),
        metadata={
            "pin_id": item.get("pin_id"),
            "source_message_id": item.get("source_message_id"),
            "pin_priority": _to_non_negative_int(item.get("priority"), default=25),
            "status": "active",
            **(dict(pin_metadata) if isinstance(pin_metadata, dict) else {}),
        },
    )


def _boundary_metadata(boundary_id: str | None) -> dict[str, Any]:
    if not boundary_id:
        return {}
    return {"boundary_id": boundary_id}


def _to_non_negative_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return default


def _message_field(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _message_role_value(message: Any) -> str:
    role = _message_field(message, "role")
    return str(getattr(role, "value", role) or "").lower()
