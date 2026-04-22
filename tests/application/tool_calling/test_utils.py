from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from houyi.application.tool_calling.runtime_options import (
    build_chat_kwargs,
    choose_tool_cache,
    wrap_tool_choice,
)


@dataclass
class _Execution:
    metadata: dict[str, Any]


class _InnerAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return {"ok": True}


class TestToolCallOrchestrator:
    def test_build_chat_kwargs_empty(self) -> None:
        assert (
            build_chat_kwargs(
                max_tokens=None,
                temperature=None,
                parallel_tool_calls=None,
                max_parallel_calls=None,
                prompt_cache_key=None,
            )
            == {}
        )

    def test_build_kwargs_includes_values(self) -> None:
        assert build_chat_kwargs(
            max_tokens=123,
            temperature=0.4,
            parallel_tool_calls=True,
            max_parallel_calls=3,
            prompt_cache_key="cache_key",
        ) == {
            "max_tokens": 123,
            "temperature": 0.4,
            "parallel_tool_calls": True,
            "max_parallel_calls": 3,
            "prompt_cache_key": "cache_key",
        }

    def test_fresh_disables_when_denied(self) -> None:
        execution = _Execution(metadata={"replay_mode": "fresh"})
        tool_cache = {"k": {"raw": 1}}
        assert (
            choose_tool_cache(
                execution=execution,  # type: ignore[arg-type]
                tool_cache=tool_cache,
                allow_fresh_tool_cache=False,
            )
            is None
        )

    def test_fresh_keeps_when_allowed(self) -> None:
        execution = _Execution(metadata={"replay_mode": "fresh"})
        tool_cache = {"k": {"raw": 1}}
        assert (
            choose_tool_cache(
                execution=execution,  # type: ignore[arg-type]
                tool_cache=tool_cache,
                allow_fresh_tool_cache=True,
            )
            == tool_cache
        )

    def test_non_fresh_keeps(self) -> None:
        execution = _Execution(metadata={"replay_mode": "deterministic"})
        tool_cache = {"k": {"raw": 1}}
        assert (
            choose_tool_cache(
                execution=execution,  # type: ignore[arg-type]
                tool_cache=tool_cache,
                allow_fresh_tool_cache=False,
            )
            == tool_cache
        )

    async def test_wrap_none_returns_same(self) -> None:
        adapter = _InnerAdapter()
        assert wrap_tool_choice(adapter=adapter, tool_choice=None) is adapter

    async def test_wrap_sets_choice_chat(self) -> None:
        adapter = _InnerAdapter()
        wrapped = wrap_tool_choice(adapter=adapter, tool_choice="required")
        await wrapped.chat([{"role": "user", "content": "hi"}], tools=[{"name": "t"}])
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["kwargs"]["tool_choice"] == "required"
