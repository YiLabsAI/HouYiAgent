from __future__ import annotations

import json
from typing import Any


def summarize_tool_calls(tool_calls: list[dict[str, Any]] | None) -> str:
    if not tool_calls:
        return "[tool loop]"
    tool_names: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_payload = tool_call.get("function")
        if not isinstance(function_payload, dict):
            continue
        name = function_payload.get("name")
        if isinstance(name, str) and name.strip():
            tool_names.append(name.strip())
    if not tool_names:
        return f"[tool loop: {len(tool_calls)} call(s)]"
    unique_names = list(dict.fromkeys(tool_names))
    preview = ", ".join(unique_names[:2])
    if len(unique_names) > 2:
        preview = f"{preview}, +{len(unique_names) - 2} more"
    return f"[tool loop: {preview}]"


def summarize_tool_payload(message: Any, max_chars: int = 120) -> str:
    tool_name = _message_name(message) or "tool"
    content = _message_content(message).strip()
    if not content:
        return f"{tool_name} returned empty result"
    try:
        payload = json.loads(content)
    except Exception:
        return _render_compact_value(tool_name, content, max_chars=max_chars)
    if isinstance(payload, dict):
        return _summarize_structured_payload(tool_name, payload, max_chars=max_chars)
    if isinstance(payload, list):
        return f"{tool_name} returned {len(payload)} item(s)"
    return _render_compact_value(tool_name, payload, max_chars=max_chars)


def summarize_compaction_message(message: Any, max_chars: int = 120) -> str:
    role = _message_role(message)
    content = _message_content(message).strip()
    if role == "tool":
        return f"{role}: {summarize_tool_payload(message, max_chars=max_chars)}"
    tool_calls = _message_tool_calls(message)
    if tool_calls and not content:
        return f"{role}: {summarize_tool_calls(tool_calls)}"
    if not content:
        content = "[empty]"
    return f"{role}: {content[:max_chars]}"


def build_compaction_summary(messages: list[Any]) -> str:
    return "\n".join(summarize_compaction_message(message) for message in messages[:3])


def _message_field(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _message_content(message: Any) -> str:
    return str(_message_field(message, "content") or "")


def _message_name(message: Any) -> str:
    name = _message_field(message, "name")
    return name.strip() if isinstance(name, str) and name.strip() else ""


def _message_role(message: Any) -> str:
    role = _message_field(message, "role")
    return str(getattr(role, "value", role) or "")


def _message_tool_calls(message: Any) -> list[dict[str, Any]] | None:
    tool_calls = _message_field(message, "tool_calls")
    return tool_calls if isinstance(tool_calls, list) else None


def _summarize_structured_payload(
    tool_name: str,
    payload: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    data_summary = _summarize_payload_data(tool_name, payload.get("data"))
    if data_summary is not None:
        return data_summary
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        compact_error = " ".join(error.split())
        return f"{tool_name} error: {compact_error[:max_chars]}"
    useful_keys = _useful_payload_keys(payload)
    if useful_keys:
        preview = ", ".join(useful_keys[:4])
        return f"{tool_name} result keys: {preview}"
    return f"{tool_name} returned structured result"


def _summarize_payload_data(tool_name: str, data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    pattern = data.get("pattern")
    matches = data.get("matches")
    if isinstance(pattern, str) and isinstance(matches, list):
        return f"{tool_name} search '{pattern}' returned {len(matches)} match(es)"
    if isinstance(data.get("count"), int):
        return f"{tool_name} returned {int(data['count'])} item(s)"
    useful_keys = [
        key
        for key in data
        if key not in {"truncated", "root_path", "matches", "content", "stdout", "stderr"}
    ]
    if useful_keys:
        preview = ", ".join(useful_keys[:4])
        return f"{tool_name} returned fields: {preview}"
    return None


def _useful_payload_keys(payload: dict[str, Any]) -> list[str]:
    return [
        key
        for key in payload
        if key
        not in {
            "data",
            "metadata",
            "truncated",
            "root_path",
            "matches",
            "content",
            "stdout",
            "stderr",
        }
    ]


def _render_compact_value(tool_name: str, value: Any, *, max_chars: int) -> str:
    compact = " ".join(str(value).split())
    return f"{tool_name}: {compact[:max_chars]}"
