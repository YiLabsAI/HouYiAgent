from __future__ import annotations

from typing import Any


def detect_incomplete_turn_gate(messages: list[Any]) -> str | None:
    pending_tool_call_ids: set[str] = set()
    for message in messages:
        role = _message_role_value(message)
        if role == "assistant":
            pending_tool_call_ids, gate = _consume_assistant_message(
                message,
                pending_tool_call_ids=pending_tool_call_ids,
            )
            if gate is not None:
                return gate
            continue
        if role == "tool":
            gate = _consume_tool_message(message, pending_tool_call_ids=pending_tool_call_ids)
            if gate is not None:
                return gate
            continue
        if _is_pending_non_tool_message(pending_tool_call_ids):
            return "active_tool_loop"
    if pending_tool_call_ids:
        return "active_tool_loop"
    return None


def evaluate_safety_gate(
    *,
    messages: list[Any],
    active_streaming_message_id: str | None,
    last_compacted_message_count: int | None,
    last_compacted_at: float | None,
    now: float,
    recent_window: int,
    cooldown_messages: int,
    cooldown_seconds: float,
) -> str | None:
    if len(messages) <= recent_window:
        return "insufficient_history"
    if isinstance(active_streaming_message_id, str) and active_streaming_message_id:
        return "active_streaming"
    incomplete_turn_gate = detect_incomplete_turn_gate(messages)
    if incomplete_turn_gate is not None:
        return incomplete_turn_gate
    if (
        cooldown_messages > 0
        and isinstance(last_compacted_message_count, int)
        and last_compacted_message_count >= 0
        and (len(messages) - last_compacted_message_count) < cooldown_messages
    ):
        return "cooldown_active"
    if last_compacted_at is not None and (now - float(last_compacted_at)) < cooldown_seconds:
        return "cooldown_active"
    return None


def partition_messages_for_compaction(
    messages: list[Any],
    *,
    protected_message_ids: set[str],
    recent_window: int,
) -> tuple[list[Any], list[Any]]:
    recent_ids = {
        message_id
        for message_id in [_message_id_value(message) for message in messages[-recent_window:]]
        if message_id
    }
    kept: list[Any] = []
    dropped: list[Any] = []
    for message in messages:
        message_id = _message_id_value(message)
        if message_id and (message_id in protected_message_ids or message_id in recent_ids):
            kept.append(message)
        else:
            dropped.append(message)
    return kept, dropped


def build_prune_only_summary(messages: list[Any]) -> str:
    if not messages:
        return ""
    first_id = _message_id_value(messages[0]) or "unknown"
    last_id = _message_id_value(messages[-1]) or first_id
    return (
        f"Pruned {len(messages)} earlier messages to recover context budget. "
        f"Range: {first_id}..{last_id}."
    )


def _message_field(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _message_id_value(message: Any) -> str:
    message_id = _message_field(message, "message_id")
    return str(message_id) if isinstance(message_id, str) and message_id else ""


def _message_role_value(message: Any) -> str:
    role = _message_field(message, "role")
    return str(getattr(role, "value", role) or "").lower()


def _consume_assistant_message(
    message: Any,
    *,
    pending_tool_call_ids: set[str],
) -> tuple[set[str], str | None]:
    if pending_tool_call_ids:
        return pending_tool_call_ids, "split_incomplete_turn"
    extracted_ids = _extract_tool_call_ids(_message_field(message, "tool_calls"))
    return extracted_ids or set(), None


def _consume_tool_message(
    message: Any,
    *,
    pending_tool_call_ids: set[str],
) -> str | None:
    resolved_tool_call_id = _resolve_tool_call_id(message)
    if not pending_tool_call_ids or not resolved_tool_call_id:
        return "split_incomplete_turn"
    if resolved_tool_call_id not in pending_tool_call_ids:
        return "split_incomplete_turn"
    pending_tool_call_ids.discard(resolved_tool_call_id)
    return None


def _extract_tool_call_ids(tool_calls: Any) -> set[str]:
    if not isinstance(tool_calls, list):
        return set()
    return {
        str(call.get("id"))
        for call in tool_calls
        if isinstance(call, dict) and isinstance(call.get("id"), str) and call.get("id")
    }


def _resolve_tool_call_id(message: Any) -> str:
    tool_call_id = _message_field(message, "tool_call_id")
    return str(tool_call_id) if isinstance(tool_call_id, str) else ""


def _is_pending_non_tool_message(pending_tool_call_ids: set[str]) -> bool:
    return bool(pending_tool_call_ids)
