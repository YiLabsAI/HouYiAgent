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
        messages = [
            {"role": "system", "content": fast_path_prompt},
            {"role": "user", "content": user_prompt or prompt},
        ]
        return ToolCallMessagesResult(
            messages=messages,
            user_content=user_prompt or prompt,
            tool_choice=apply_fast_path_tool_choice(
                fast_path_enabled=True,
                tool_choice=tool_choice,
            ),
        )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return ToolCallMessagesResult(
        messages=messages,
        user_content=prompt,
        tool_choice=tool_choice,
    )
