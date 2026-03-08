"""Tool-call response message builders and fast-path prompt helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def apply_fast_path_tool_choice(*, fast_path_enabled: bool, tool_choice: Any) -> Any:
    if not fast_path_enabled:
        return tool_choice
    if tool_choice is None or tool_choice == "auto":
        return "required"
    return tool_choice


def build_fast_path_prompt(*, tool_names: list[str], skills: list[Any]) -> str:
    names = [name for name in (tool_names or []) if name]
    if not names:
        names = [
            getattr(skill, "name", "") for skill in (skills or []) if getattr(skill, "name", "")
        ]
    joined = ", ".join(names)
    return (
        "You are in tool-call fast-path mode. "
        f"Call each tool exactly once in a deterministic order: {joined}. "
        "Do not call any other tools. "
        "After tools return, provide the final answer."
    )


def build_assistant_tool_message(response: Any) -> dict[str, Any]:
    assistant_tool_message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": response.tool_calls,
    }
    response_metadata = getattr(response, "metadata", None)
    if isinstance(response_metadata, dict):
        reasoning_content = response_metadata.get("reasoning_content")
        if isinstance(reasoning_content, str):
            assistant_tool_message["reasoning_content"] = reasoning_content
    return assistant_tool_message


@dataclass(frozen=True)
class ToolCallMessagesResult:
    messages: list[dict[str, Any]]
    user_content: str
    tool_choice: Any


def build_tool_call_messages(
    *,
    system_prompt: str | None,
    user_prompt: str | None,
    prompt: str,
    fast_path_enabled: bool,
    fast_path_prompt: str | None,
    tool_choice: Any,
) -> ToolCallMessagesResult:
    if fast_path_enabled and fast_path_prompt:
        fast_messages = [
            {"role": "system", "content": fast_path_prompt},
            {"role": "user", "content": user_prompt or prompt},
        ]
        return ToolCallMessagesResult(
            messages=fast_messages,
            user_content=user_prompt or prompt,
            tool_choice=apply_fast_path_tool_choice(
                fast_path_enabled=True,
                tool_choice=tool_choice,
            ),
        )

    regular_messages: list[dict[str, Any]] = []
    if system_prompt:
        regular_messages.append({"role": "system", "content": system_prompt})
    regular_messages.append({"role": "user", "content": prompt})

    return ToolCallMessagesResult(
        messages=regular_messages,
        user_content=prompt,
        tool_choice=tool_choice,
    )


__all__ = [
    "ToolCallMessagesResult",
    "apply_fast_path_tool_choice",
    "build_assistant_tool_message",
    "build_fast_path_prompt",
    "build_tool_call_messages",
]
