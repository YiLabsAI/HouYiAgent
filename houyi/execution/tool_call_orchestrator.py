from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def build_chat_kwargs(
    *,
    max_tokens: int | None,
    temperature: float | None,
    parallel_tool_calls: bool | None,
    prompt_cache_key: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = parallel_tool_calls
    if prompt_cache_key is not None:
        kwargs["prompt_cache_key"] = prompt_cache_key
    return kwargs


def choose_tool_cache(
    *, execution: Any, tool_cache: dict[str, Any] | None, allow_fresh_tool_cache: bool
) -> dict[str, Any] | None:
    if tool_cache is None:
        return None
    replay_mode = None
    metadata = getattr(execution, "metadata", None)
    if isinstance(metadata, dict):
        replay_mode = metadata.get("replay_mode")
    if replay_mode == "fresh" and not allow_fresh_tool_cache:
        return None
    return tool_cache


class _ChatAdapter(Protocol):
    async def chat(
        self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any: ...


@dataclass
class _ToolChoiceWrapper:
    adapter: _ChatAdapter
    tool_choice: Any

    async def chat(
        self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any:
        if self.tool_choice is not None and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = self.tool_choice
        return await self.adapter.chat(messages, tools=tools, **kwargs)


def wrap_tool_choice(*, adapter: _ChatAdapter, tool_choice: Any) -> _ChatAdapter:
    if tool_choice is None:
        return adapter
    return _ToolChoiceWrapper(adapter=adapter, tool_choice=tool_choice)
