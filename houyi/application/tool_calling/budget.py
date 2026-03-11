"""Message budget management for tool-calling loops."""

from __future__ import annotations

import json
import logging
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.adapters.llm.models import (
    CHARS_PER_TOKEN_BLENDED,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_OUTPUT_RESERVE,
    MODEL_CONTEXT_WINDOWS,
)

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_LOOP_MAX_MESSAGE_CHARS = 12_000
_MIN_TOOL_LOOP_MAX_MESSAGE_CHARS = 1_000
_MIN_TOOL_LOOP_MAX_TOTAL_CHARS = 8_000
_AUTO_TOOL_LOOP_INPUT_BUDGET_RATIO = 0.7
_AUTO_TOOL_LOOP_MESSAGE_RATIO = 0.1
_DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS = 4_000
_DEFAULT_TOOL_RESULT_SUMMARY_MAX_ITEMS = 50


class MessageBudget:
    """Manage per-message and total payload budgets for tool loops."""

    def __init__(
        self,
        adapter: Any,
        requested_model: str | None,
        env_max_message_chars: int | None,
        env_max_total_chars: int | None,
    ) -> None:
        self.max_message_chars, self.max_total_chars = self._resolve_budgets(
            adapter, env_max_message_chars, env_max_total_chars, requested_model
        )
        logger.debug(
            "MessageBudget initialized: max_message=%d max_total=%d",
            self.max_message_chars,
            self.max_total_chars,
        )

    def prepare_messages(self, messages: list[Any]) -> list[Any]:
        normalized_messages = LLMAdapter._sanitize_messages(
            [m for m in messages if isinstance(m, dict)]
        )
        normalized_messages = [
            self._truncate_message(msg, self.max_message_chars) for msg in normalized_messages
        ]
        return self._cap_total_payload(normalized_messages, self.max_total_chars)

    @staticmethod
    def summarize_tool_result(
        content: str,
        max_chars: int = _DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS,
        max_items: int = _DEFAULT_TOOL_RESULT_SUMMARY_MAX_ITEMS,
    ) -> tuple[str, bool]:
        if len(content) <= max_chars:
            return content, False

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and len(parsed) > max_items:
                truncated = parsed[:max_items]
                truncated.append(
                    {
                        "_truncated": True,
                        "_truncated_message": f"...[{len(parsed) - max_items} items truncated]...",
                        "_original_count": len(parsed),
                        "_showing": max_items,
                    }
                )
                summarized = json.dumps(truncated, ensure_ascii=False)
                if len(summarized) <= max_chars:
                    return summarized, True
            if isinstance(parsed, dict):
                summarized_dict: dict[str, Any] = {}
                char_count = 0
                for key, value in parsed.items():
                    value_str = json.dumps(value, ensure_ascii=False)
                    if char_count + len(value_str) > max_chars:
                        summarized_dict["_truncated"] = True
                        summarized_dict["_truncated_message"] = "...[truncated]..."
                        break
                    summarized_dict[key] = value
                    char_count += len(value_str)
                summarized = json.dumps(summarized_dict, ensure_ascii=False)
                if len(summarized) <= max_chars:
                    return summarized, True
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        truncated_content = (
            content[:max_chars] + f"\n... [truncated, original length: {len(content)}]"
        )
        return truncated_content, True

    @staticmethod
    def _resolve_budgets(
        adapter: Any,
        env_max_message_chars: int | None,
        env_max_total_chars: int | None,
        requested_model: str | None,
    ) -> tuple[int, int]:
        if env_max_message_chars and env_max_total_chars:
            return (
                max(env_max_message_chars, _MIN_TOOL_LOOP_MAX_MESSAGE_CHARS),
                max(env_max_total_chars, _MIN_TOOL_LOOP_MAX_TOTAL_CHARS),
            )

        model_name = requested_model or getattr(adapter, "model", None)
        context_window = DEFAULT_CONTEXT_WINDOW
        output_reserve = DEFAULT_OUTPUT_RESERVE

        if model_name:
            for known_model, window in MODEL_CONTEXT_WINDOWS.items():
                if model_name == known_model or model_name.startswith(known_model + "-"):
                    context_window = window
                    break

        input_budget_tokens = int(
            (context_window - output_reserve) * _AUTO_TOOL_LOOP_INPUT_BUDGET_RATIO
        )
        input_budget_chars = input_budget_tokens * CHARS_PER_TOKEN_BLENDED

        max_message_chars = int(input_budget_chars * _AUTO_TOOL_LOOP_MESSAGE_RATIO)
        max_total_chars = int(input_budget_chars)

        if env_max_message_chars:
            max_message_chars = env_max_message_chars
        if env_max_total_chars:
            max_total_chars = env_max_total_chars

        max_message_chars = max(max_message_chars, _MIN_TOOL_LOOP_MAX_MESSAGE_CHARS)
        max_total_chars = max(max_total_chars, _MIN_TOOL_LOOP_MAX_TOTAL_CHARS)

        return max_message_chars, max_total_chars

    @staticmethod
    def _truncate_message(msg: dict[str, Any], max_chars: int) -> dict[str, Any]:
        if not isinstance(msg, dict):
            return msg

        content = msg.get("content")
        if not isinstance(content, str):
            return msg

        if len(content) <= max_chars:
            return msg

        truncated_msg = dict(msg)
        truncated_msg["content"] = content[:max_chars] + "... [truncated]"
        return truncated_msg

    @staticmethod
    def _cap_total_payload(
        messages: list[dict[str, Any]],
        max_total_chars: int,
    ) -> list[dict[str, Any]]:
        if not messages:
            return messages

        total_chars = sum(_message_payload_chars(msg) for msg in messages)
        if total_chars <= max_total_chars:
            return messages

        system_msg = None
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            messages = messages[1:]

        last_user_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                break

        trimmed: list[dict[str, Any]] = []
        current_chars = 0

        if system_msg:
            current_chars += _message_payload_chars(system_msg)
            trimmed.append(system_msg)

        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            msg_chars = _message_payload_chars(msg)
            if idx == last_user_idx:
                trimmed.insert(1 if system_msg else 0, msg)
                current_chars += msg_chars
                continue
            if current_chars + msg_chars <= max_total_chars:
                trimmed.insert(1 if system_msg else 0, msg)
                current_chars += msg_chars

        logger.debug(
            "Message payload capped: %d chars → %d chars, %d messages → %d messages",
            total_chars,
            current_chars,
            len(messages) + (1 if system_msg else 0),
            len(trimmed),
        )

        return trimmed


def prepare_tool_loop_messages(
    messages: list[Any],
    max_message_chars: int,
    max_total_chars: int,
) -> list[dict[str, Any]]:
    """Prepare loop messages using explicit char limits.

    This helper keeps ToolCallRunner free of local payload-budget helpers and
    centralizes budget shaping logic in the application layer.
    """
    normalized_messages = LLMAdapter._sanitize_messages(
        [m for m in messages if isinstance(m, dict)]
    )
    normalized_messages = [
        _truncate_message_for_budget(msg, max_message_chars) for msg in normalized_messages
    ]
    capped_messages = _cap_total_payload_for_budget(normalized_messages, max_total_chars)
    return _sanitize_tool_message_structure(capped_messages)


def _truncate_middle_for_budget(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def _truncate_message_for_budget(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    normalized = dict(message)
    normalized["content"] = _truncate_middle_for_budget(
        LLMAdapter._coerce_message_content_to_text(normalized.get("content")),
        max_chars,
    )
    tool_calls = normalized.get("tool_calls")
    if not isinstance(tool_calls, list):
        return normalized

    fixed_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        fixed = LLMAdapter._sanitize_tool_call(call)
        if fixed is None:
            continue
        fn = fixed.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = _truncate_middle_for_budget(args, max_chars)
        fixed_calls.append(fixed)
    normalized["tool_calls"] = fixed_calls
    return normalized


def _cap_total_payload_for_budget(
    messages: list[dict[str, Any]],
    max_total_chars: int,
) -> list[dict[str, Any]]:
    total_chars = sum(_message_payload_chars(msg) for msg in messages)
    if total_chars <= max_total_chars:
        return messages

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    system_chars = sum(_message_payload_chars(msg) for msg in system_messages)
    budget_for_non_system = max(0, max_total_chars - system_chars)

    kept_non_system: list[dict[str, Any]] = []
    used = 0
    for msg in reversed(non_system):
        payload_chars = _message_payload_chars(msg)
        if kept_non_system and used + payload_chars > budget_for_non_system:
            continue
        kept_non_system.append(msg)
        used += payload_chars

    trimmed = system_messages + list(reversed(kept_non_system))
    logger.warning(
        "ToolCallRunner message budget applied: total_payload=%d -> %d, messages=%d -> %d",
        total_chars,
        sum(_message_payload_chars(msg) for msg in trimmed),
        len(messages),
        len(trimmed),
    )
    return trimmed


def _has_assistant_tool_turn(message: dict[str, Any]) -> bool:
    return (
        str(message.get("role") or "") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        and bool(message.get("tool_calls"))
    )


def _expected_tool_call_ids(message: dict[str, Any]) -> list[str]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return []
    return [
        str(call.get("id") or "")
        for call in tool_calls
        if isinstance(call, dict) and str(call.get("id") or "")
    ]


def _is_pending_tool_message(
    message: dict[str, Any],
    pending_assistant: dict[str, Any] | None,
    pending_expected_ids: list[str],
    pending_tool_ids: list[str],
) -> tuple[bool, str]:
    tool_call_id = str(message.get("tool_call_id") or "")
    should_keep = (
        pending_assistant is not None
        and bool(tool_call_id)
        and tool_call_id in pending_expected_ids
        and tool_call_id not in pending_tool_ids
    )
    return should_keep, tool_call_id


def _flush_pending_tool_group(
    sanitized: list[dict[str, Any]],
    pending_assistant: dict[str, Any] | None,
    pending_expected_ids: list[str],
    pending_tool_messages: list[dict[str, Any]],
    pending_tool_ids: list[str],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]], list[str]]:
    if pending_assistant is None:
        return None, [], [], []
    if not pending_tool_messages:
        sanitized.append(pending_assistant)
    elif pending_tool_ids == pending_expected_ids:
        sanitized.append(pending_assistant)
        sanitized.extend(pending_tool_messages)
    return None, [], [], []


def _sanitize_tool_message_structure(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_assistant_tool_turn = any(_has_assistant_tool_turn(message) for message in messages)
    if not has_assistant_tool_turn:
        return messages

    sanitized: list[dict[str, Any]] = []
    pending_assistant: dict[str, Any] | None = None
    pending_expected_ids: list[str] = []
    pending_tool_messages: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []

    for message in messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            pending_assistant, pending_expected_ids, pending_tool_messages, pending_tool_ids = (
                _flush_pending_tool_group(
                    sanitized,
                    pending_assistant,
                    pending_expected_ids,
                    pending_tool_messages,
                    pending_tool_ids,
                )
            )
            expected_ids = _expected_tool_call_ids(message)
            if expected_ids:
                pending_assistant = message
                pending_expected_ids = expected_ids
                pending_tool_messages = []
                pending_tool_ids = []
                continue
            sanitized.append(message)
            continue
        if role == "tool":
            should_keep, tool_call_id = _is_pending_tool_message(
                message,
                pending_assistant,
                pending_expected_ids,
                pending_tool_ids,
            )
            if should_keep:
                pending_tool_messages.append(message)
                pending_tool_ids.append(tool_call_id)
            continue
        pending_assistant, pending_expected_ids, pending_tool_messages, pending_tool_ids = (
            _flush_pending_tool_group(
                sanitized,
                pending_assistant,
                pending_expected_ids,
                pending_tool_messages,
                pending_tool_ids,
            )
        )
        sanitized.append(message)

    _flush_pending_tool_group(
        sanitized,
        pending_assistant,
        pending_expected_ids,
        pending_tool_messages,
        pending_tool_ids,
    )
    return sanitized


def _message_payload_chars(msg: dict[str, Any]) -> int:
    chars = 0
    content = msg.get("content")
    if isinstance(content, str):
        chars += len(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chars += len(text)

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                if isinstance(func, dict):
                    name = func.get("name")
                    arguments = func.get("arguments")
                    chars += len(name) if isinstance(name, str) else len(str(name or ""))
                    chars += (
                        len(arguments) if isinstance(arguments, str) else len(str(arguments or ""))
                    )

    return chars


def resolve_tool_loop_budget_chars(
    adapter: Any,
    message_chars_override: int | None,
    total_chars_override: int | None,
    model_name_override: str | None = None,
) -> tuple[int, int]:
    """Resolve tool-loop message and total char budgets.

    This mirrors historical ToolCallRunner behavior (including fuzzy model name
    matching) while centralizing budget logic in application layer.
    """
    model_name = str(model_name_override or getattr(adapter, "model", "") or "")
    context_window = MODEL_CONTEXT_WINDOWS.get(model_name)
    if context_window is None and model_name:
        model_name_lower = model_name.lower()
        for known_model, known_window in MODEL_CONTEXT_WINDOWS.items():
            known_lower = known_model.lower()
            if known_lower in model_name_lower or model_name_lower in known_lower:
                context_window = known_window
                break
    if context_window is None:
        context_window = DEFAULT_CONTEXT_WINDOW

    input_budget_tokens = max(
        1,
        int(max(0, context_window - DEFAULT_OUTPUT_RESERVE) * _AUTO_TOOL_LOOP_INPUT_BUDGET_RATIO),
    )
    auto_total_chars = max(
        _MIN_TOOL_LOOP_MAX_TOTAL_CHARS,
        int(input_budget_tokens * CHARS_PER_TOKEN_BLENDED),
    )

    total_chars = total_chars_override or auto_total_chars
    if message_chars_override is not None:
        message_chars = message_chars_override
        if total_chars_override is None:
            total_chars = max(total_chars, message_chars * 4)
    else:
        message_chars = max(
            _MIN_TOOL_LOOP_MAX_MESSAGE_CHARS,
            int(total_chars * _AUTO_TOOL_LOOP_MESSAGE_RATIO),
        )
        message_chars = min(message_chars, _DEFAULT_TOOL_LOOP_MAX_MESSAGE_CHARS)

    message_chars = min(message_chars, total_chars)
    return message_chars, total_chars


__all__ = [
    "MessageBudget",
    "prepare_tool_loop_messages",
    "resolve_tool_loop_budget_chars",
]
