"""Message budget management for tool-calling loops."""

from __future__ import annotations

import json
import logging
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.adapters.llm.models import (
    CHARS_PER_TOKEN_BLENDED,
    DEEPSEEK_R1,
    DEEPSEEK_V3_2,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_OUTPUT_RESERVE,
    MODEL_CONTEXT_WINDOWS,
    normalize_model_id,
)

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_LOOP_MAX_MESSAGE_CHARS = 12_000
_MIN_TOOL_LOOP_MAX_MESSAGE_CHARS = 1_000
_MIN_TOOL_LOOP_MAX_TOTAL_CHARS = 8_000
_DEFAULT_TOOL_LOOP_MAX_MESSAGES = 48
_DEFAULT_TOOL_LOOP_CONTEXT_GROUPS_WITH_TOOLS = 8
_AUTO_TOOL_LOOP_INPUT_BUDGET_RATIO = 0.7
_AUTO_TOOL_LOOP_MESSAGE_RATIO = 0.1
_DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS = 4_000
_DEFAULT_TOOL_RESULT_SUMMARY_MAX_ITEMS = 50
_DEEPSEEK_TOOL_MAX_MESSAGE_CHARS = 6_000
_DEEPSEEK_TOOL_MAX_TOTAL_CHARS = 48_000


def _is_deepseek_tool_model(model_name: str) -> bool:
    normalized = normalize_model_id(model_name or "")
    return normalized in {
        normalize_model_id(DEEPSEEK_R1),
        normalize_model_id(DEEPSEEK_V3_2),
    }


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
    structured_messages = _sanitize_tool_message_structure(normalized_messages)
    capped_messages = _cap_message_count_for_budget(
        structured_messages,
        max_messages=_DEFAULT_TOOL_LOOP_MAX_MESSAGES,
    )
    return _cap_total_payload_for_budget(capped_messages, max_total_chars)


def _truncate_middle_for_budget(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def _truncate_message_for_budget(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    normalized = dict(message)
    if "content" in normalized:
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
    if budget_for_non_system <= 0:
        logger.warning(
            "ToolCallRunner message budget applied: total_payload=%d -> %d, messages=%d -> %d",
            total_chars,
            system_chars,
            len(messages),
            len(system_messages),
        )
        return system_messages

    groups = _group_messages_for_budget(non_system)
    if not groups:
        return system_messages

    kept_group_indexes: set[int] = set()
    used = 0
    recent_tail_groups = min(2, len(groups))
    for group_index in range(len(groups) - recent_tail_groups, len(groups)):
        if group_index < 0:
            continue
        group_chars = _group_payload_chars(groups[group_index])
        if used + group_chars > budget_for_non_system and kept_group_indexes:
            continue
        kept_group_indexes.add(group_index)
        used += group_chars

    candidate_indexes = [
        group_index for group_index in range(len(groups)) if group_index not in kept_group_indexes
    ]
    candidate_indexes.sort(
        key=lambda group_index: (
            _score_group_for_budget(groups[group_index]),
            group_index,
        ),
        reverse=True,
    )
    for group_index in candidate_indexes:
        group_chars = _group_payload_chars(groups[group_index])
        if used + group_chars > budget_for_non_system:
            continue
        kept_group_indexes.add(group_index)
        used += group_chars

    if not kept_group_indexes:
        latest_group_index = len(groups) - 1
        kept_group_indexes.add(latest_group_index)
        used = _group_payload_chars(groups[latest_group_index])

    kept_non_system = [
        message
        for group_index, group in enumerate(groups)
        if group_index in kept_group_indexes
        for message in group
    ]
    trimmed = system_messages + kept_non_system
    logger.warning(
        "ToolCallRunner message budget applied: total_payload=%d -> %d, messages=%d -> %d",
        total_chars,
        sum(_message_payload_chars(msg) for msg in trimmed),
        len(messages),
        len(trimmed),
    )
    return trimmed


def _cap_message_count_for_budget(
    messages: list[dict[str, Any]],
    *,
    max_messages: int,
) -> list[dict[str, Any]]:
    if max_messages <= 0:
        return []

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    if len(non_system) <= max_messages:
        return messages

    groups = _group_messages_for_budget(non_system)
    groups = _prioritize_recent_tool_groups(groups)

    kept_groups: list[list[dict[str, Any]]] = []
    kept_count = 0
    for group in reversed(groups):
        group_size = len(group)
        if kept_groups and kept_count + group_size > max_messages:
            continue
        kept_groups.append(group)
        kept_count += group_size

    trimmed = system_messages + [item for group in reversed(kept_groups) for item in group]
    logger.warning(
        "ToolCallRunner message count cap applied: messages=%d -> %d (non_system=%d -> %d)",
        len(messages),
        len(trimmed),
        len(non_system),
        len(trimmed) - len(system_messages),
    )
    return trimmed


def _prioritize_recent_tool_groups(
    groups: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    latest_tool_group_index = _find_latest_tool_group_index(groups)
    if latest_tool_group_index < 0:
        return groups

    trailing_groups = groups[latest_tool_group_index:]
    leading_groups = groups[:latest_tool_group_index]
    if not leading_groups:
        return trailing_groups

    return leading_groups[-_DEFAULT_TOOL_LOOP_CONTEXT_GROUPS_WITH_TOOLS:] + trailing_groups


def _find_latest_tool_group_index(groups: list[list[dict[str, Any]]]) -> int:
    for index in range(len(groups) - 1, -1, -1):
        if _group_contains_tool_context(groups[index]):
            return index
    return -1


def _group_messages_for_budget(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = str(message.get("role") or "")
        expected_ids = _expected_tool_call_ids(message)
        if not expected_ids:
            if role == "user":
                group = [message]
                next_index = index + 1
                if next_index < len(messages):
                    candidate = messages[next_index]
                    candidate_role = str(candidate.get("role") or "")
                    if candidate_role == "assistant" and not _expected_tool_call_ids(candidate):
                        group.append(candidate)
                        next_index += 1
                groups.append(group)
                index = next_index
                continue
            groups.append([message])
            index += 1
            continue
        group = [message]
        pending_ids = set(expected_ids)
        next_index = index + 1
        while next_index < len(messages):
            candidate = messages[next_index]
            if str(candidate.get("role") or "") != "tool":
                break
            tool_call_id = str(candidate.get("tool_call_id") or "")
            if tool_call_id not in pending_ids:
                break
            group.append(candidate)
            pending_ids.remove(tool_call_id)
            next_index += 1
            if not pending_ids:
                break
        groups.append(group)
        index = next_index
    return groups


def _group_payload_chars(group: list[dict[str, Any]]) -> int:
    return sum(_message_payload_chars(message) for message in group)


def _score_group_for_budget(group: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    tool_context_score = 0
    structured_content_score = 0
    assistant_or_tool_score = 0
    for message in group:
        role = str(message.get("role") or "")
        if role in {"assistant", "tool"}:
            assistant_or_tool_score += 1
        if _expected_tool_call_ids(message) or role == "tool":
            tool_context_score += 3
        structured_content_score += _score_message_structure(message)
    return (
        tool_context_score,
        structured_content_score,
        assistant_or_tool_score,
        -_group_payload_chars(group),
    )


def _score_message_structure(message: dict[str, Any]) -> int:
    score = 0
    content = LLMAdapter._coerce_message_content_to_text(message.get("content"))
    text = content.strip()
    if not text:
        return score
    if text.startswith("{") or text.startswith("["):
        score += 3
    if "\n" in text:
        score += 1
    if any(token in text for token in ('{"', '"}:', '"],', '":', "\t", "- ", "1. ")):
        score += 1
    return score


def _group_contains_tool_context(group: list[dict[str, Any]]) -> bool:
    return any(
        _expected_tool_call_ids(message) or str(message.get("role") or "") == "tool"
        for message in group
    )


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


def _build_single_tool_assistant_message(
    message: dict[str, Any],
    *,
    tool_call_id: str,
) -> dict[str, Any] | None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    matched_call = next(
        (
            call
            for call in tool_calls
            if isinstance(call, dict) and str(call.get("id") or "") == tool_call_id
        ),
        None,
    )
    if matched_call is None:
        return None
    single_message: dict[str, Any] = {"role": "assistant", "tool_calls": [matched_call]}
    if "content" in message:
        single_message["content"] = message.get("content")
    if "reasoning_content" in message:
        single_message["reasoning_content"] = message.get("reasoning_content")
    return single_message


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

    deepseek_tool_model = _is_deepseek_tool_model(model_name)

    input_budget_tokens = max(
        1,
        int(max(0, context_window - DEFAULT_OUTPUT_RESERVE) * _AUTO_TOOL_LOOP_INPUT_BUDGET_RATIO),
    )
    auto_total_chars = max(
        _MIN_TOOL_LOOP_MAX_TOTAL_CHARS,
        int(input_budget_tokens * CHARS_PER_TOKEN_BLENDED),
    )
    if deepseek_tool_model:
        auto_total_chars = min(auto_total_chars, _DEEPSEEK_TOOL_MAX_TOTAL_CHARS)

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
        if deepseek_tool_model:
            message_chars = min(message_chars, _DEEPSEEK_TOOL_MAX_MESSAGE_CHARS)

    message_chars = min(message_chars, total_chars)
    return message_chars, total_chars


__all__ = [
    "MessageBudget",
    "prepare_tool_loop_messages",
    "resolve_tool_loop_budget_chars",
]
