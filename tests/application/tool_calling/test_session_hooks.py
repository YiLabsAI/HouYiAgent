"""Tests for SessionStart / Stop hook triggers in ToolCallRunner.

Verifies that ToolCallRunner.run() fires SessionStart at entry and Stop
before *every* return path (no-tool-calls, fast-path exit, max-rounds).
Also covers edge cases: hooks disabled (None manager), hook exceptions
(non-fatal), and argument correctness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from houyi.application.tool_calling.runner import ToolCallRunner
from houyi.domain.skill.hooks import (
    HookEvent,
    HookType,
    SkillHook,
    SkillHooksManager,
)
from houyi.domain.skill.spec import SkillSpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class EmptyInput(BaseModel):
    pass


class SimpleOutput(BaseModel):
    result: str = "ok"


@dataclass
class FakeResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_copy(self, deep: bool = False) -> FakeResponse:
        return FakeResponse(
            content=self.content,
            tool_calls=json.loads(json.dumps(self.tool_calls)) if deep else list(self.tool_calls),
            metadata=json.loads(json.dumps(self.metadata)) if deep else dict(self.metadata),
        )


class FakeAdapter:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.model = "fake-model"
        self.base_url = "http://fake.local"

    async def chat(self, _messages: Any, **_kw: Any) -> FakeResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return FakeResponse(content="done")


class FakeExecutor:
    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.results = results or {}
        self.max_retries = 1
        self.timeout = 10.0

    async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
        return self.results.get(skill.name, {"result": "ok"})


def _make_tool_call(
    tool_name: str = "test-skill", args: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "id": f"call_{tool_name}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(args or {}),
        },
    }


# ---------------------------------------------------------------------------
# Hook registration helper
# ---------------------------------------------------------------------------


def _make_skill_with_hooks(
    name: str,
    hook_events: list[HookEvent],
    handler: Any,
) -> SkillSpec:
    """Create a SkillSpec with handler-type hooks for the given events."""
    hooks = [
        SkillHook(event=evt, hook_type=HookType.HANDLER, handler=handler) for evt in hook_events
    ]
    return SkillSpec(
        name=name,
        description="Test skill with hooks",
        input_schema=EmptyInput,
        output_schema=SimpleOutput,
        hooks=hooks,
    )


def _register_hooks(
    manager: SkillHooksManager,
    events: list[HookEvent],
    handler: Any,
    skill_name: str = "__hook_test__",
) -> None:
    """Register hooks for given events via a temporary SkillSpec."""
    skill = _make_skill_with_hooks(skill_name, events, handler)
    manager.register_hooks(skill)


def _make_skill(name: str = "test-skill") -> SkillSpec:
    return SkillSpec(
        name=name,
        description="Test skill",
        input_schema=EmptyInput,
        output_schema=SimpleOutput,
    )


# ---------------------------------------------------------------------------
# Tests: SessionStart hook
# ---------------------------------------------------------------------------


class TestSessionStartHook:
    """Tests for the SessionStart hook at the beginning of run()."""

    @pytest.mark.asyncio
    async def test_session_start_fires_on_entry(self) -> None:
        """SessionStart should fire even when the LLM returns no tool calls."""
        fired: list[str] = []

        async def on_session_start(ctx: Any) -> dict[str, Any]:
            fired.append("session_start")
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.SESSION_START], on_session_start)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="no tools")])
        skill = _make_skill()

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert "session_start" in fired

    @pytest.mark.asyncio
    async def test_session_start_receives_correct_args(self) -> None:
        """SessionStart context should include max_rounds, tool_count, skill_count."""
        received_args: dict[str, Any] = {}

        async def on_session_start(ctx: Any) -> dict[str, Any]:
            received_args.update(ctx.tool_args)
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.SESSION_START], on_session_start)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="done")])
        skills = [_make_skill("a"), _make_skill("b")]

        tool_defs = [
            {"type": "function", "function": {"name": "a", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "parameters": {}}},
            {"type": "function", "function": {"name": "c", "parameters": {}}},
        ]

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=tool_defs,
            skills=skills,
            executor=FakeExecutor(),
            max_rounds=5,
        )

        assert received_args["max_rounds"] == 5
        assert received_args["tool_count"] == 3
        assert received_args["skill_count"] == 2

    @pytest.mark.asyncio
    async def test_session_start_exception_is_non_fatal(self) -> None:
        """SessionStart hook exception should not abort the run."""

        async def broken_hook(ctx: Any) -> dict[str, Any]:
            raise RuntimeError("SessionStart hook exploded")

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.SESSION_START], broken_hook)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="still works")])

        response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert response.content == "still works"

    @pytest.mark.asyncio
    async def test_no_session_start_when_hooks_manager_is_none(self) -> None:
        """No error when skill_hooks_manager is None."""
        runner = ToolCallRunner(skill_hooks_manager=None)
        adapter = FakeAdapter([FakeResponse(content="ok")])

        response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert response.content == "ok"


# ---------------------------------------------------------------------------
# Tests: Stop hook
# ---------------------------------------------------------------------------


class TestStopHook:
    """Tests for the Stop hook before every return in run()."""

    @pytest.mark.asyncio
    async def test_stop_fires_on_no_tool_calls(self) -> None:
        """Stop hook fires when LLM returns no tool calls (early return)."""
        stop_count = 0

        async def on_stop(ctx: Any) -> dict[str, Any]:
            nonlocal stop_count
            stop_count += 1
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.STOP], on_stop)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="no tools")])

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert stop_count == 1

    @pytest.mark.asyncio
    async def test_stop_fires_on_max_rounds(self) -> None:
        """Stop hook fires when max_rounds is exhausted."""
        stop_count = 0

        async def on_stop(ctx: Any) -> dict[str, Any]:
            nonlocal stop_count
            stop_count += 1
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.STOP], on_stop)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter(
            [
                FakeResponse(tool_calls=[_make_tool_call()]),
                FakeResponse(tool_calls=[_make_tool_call()]),
                FakeResponse(tool_calls=[_make_tool_call()]),
            ]
        )

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "test-skill", "parameters": {}}}],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=2,
        )

        assert stop_count == 1

    @pytest.mark.asyncio
    async def test_stop_receives_tool_trace_length(self) -> None:
        """Stop hook context should include tool_trace_length."""
        received_args: dict[str, Any] = {}

        async def on_stop(ctx: Any) -> dict[str, Any]:
            received_args.update(ctx.tool_args)
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.STOP], on_stop)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter(
            [
                FakeResponse(tool_calls=[_make_tool_call()]),
                FakeResponse(content="done"),
            ]
        )

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "test-skill", "parameters": {}}}],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=5,
        )

        assert received_args["tool_trace_length"] >= 1

    @pytest.mark.asyncio
    async def test_stop_exception_is_non_fatal(self) -> None:
        """Stop hook exception should not affect the returned result."""

        async def broken_stop(ctx: Any) -> dict[str, Any]:
            raise RuntimeError("Stop hook exploded")

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.STOP], broken_stop)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="works")])

        response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert response.content == "works"

    @pytest.mark.asyncio
    async def test_stop_trigger_hook_exception_is_non_fatal(self) -> None:
        """Even if trigger_hook itself raises (not handler), result is returned."""
        from unittest.mock import AsyncMock, MagicMock

        hooks = MagicMock(spec=SkillHooksManager)
        # First call (SessionStart) succeeds, second call (Stop) explodes
        hooks.trigger_hook = AsyncMock(side_effect=[None, RuntimeError("trigger_hook blew up")])

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="survived")])

        response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert response.content == "survived"

    @pytest.mark.asyncio
    async def test_no_stop_when_hooks_manager_is_none(self) -> None:
        """_trigger_stop_hook early-returns when hooks manager is None."""
        runner = ToolCallRunner(skill_hooks_manager=None)
        # Should not raise
        await runner._trigger_stop_hook([{"tool": "test"}])

    @pytest.mark.asyncio
    async def test_both_session_start_and_stop_fire(self) -> None:
        """SessionStart fires first, Stop fires last."""
        events: list[str] = []

        async def on_start(ctx: Any) -> dict[str, Any]:
            events.append("start")
            return {"success": True, "output": "ok"}

        async def on_stop(ctx: Any) -> dict[str, Any]:
            events.append("stop")
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.SESSION_START], on_start, skill_name="start_skill")
        _register_hooks(hooks, [HookEvent.STOP], on_stop, skill_name="stop_skill")

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="done")])

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert events == ["start", "stop"]

    @pytest.mark.asyncio
    async def test_stop_fires_exactly_once_with_tool_calls(self) -> None:
        """Regardless of how many rounds, Stop fires exactly once."""
        stop_count = 0

        async def on_stop(ctx: Any) -> dict[str, Any]:
            nonlocal stop_count
            stop_count += 1
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.STOP], on_stop)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter(
            [
                FakeResponse(tool_calls=[_make_tool_call()]),
                FakeResponse(tool_calls=[_make_tool_call()]),
                FakeResponse(content="done"),
            ]
        )

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "test-skill", "parameters": {}}}],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=5,
        )

        assert stop_count == 1

    @pytest.mark.asyncio
    async def test_stop_fires_on_first_round_no_tools(self) -> None:
        """Stop hook fires when LLM immediately returns without tool calls."""
        stop_args: dict[str, Any] = {}

        async def on_stop(ctx: Any) -> dict[str, Any]:
            stop_args.update(ctx.tool_args)
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.STOP], on_stop)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="immediate")])

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert stop_args["tool_trace_length"] == 0

    @pytest.mark.asyncio
    async def test_session_start_trigger_hook_exception_is_non_fatal(self) -> None:
        """Even if trigger_hook itself raises (not the handler), run proceeds."""
        from unittest.mock import AsyncMock, MagicMock

        hooks = MagicMock(spec=SkillHooksManager)
        hooks.trigger_hook = AsyncMock(side_effect=RuntimeError("trigger_hook blew up"))

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = FakeAdapter([FakeResponse(content="survived")])

        response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert response.content == "survived"

    @pytest.mark.asyncio
    async def test_session_start_fires_before_any_llm_call(self) -> None:
        """SessionStart fires before the first adapter.chat() call."""
        order: list[str] = []

        async def on_start(ctx: Any) -> dict[str, Any]:
            order.append("hook")
            return {"success": True, "output": "ok"}

        hooks = SkillHooksManager()
        _register_hooks(hooks, [HookEvent.SESSION_START], on_start)

        class TrackingAdapter(FakeAdapter):
            async def chat(self, *args: Any, **kwargs: Any) -> FakeResponse:
                order.append("chat")
                return await super().chat(*args, **kwargs)

        runner = ToolCallRunner(skill_hooks_manager=hooks)
        adapter = TrackingAdapter([FakeResponse(content="done")])

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            skills=[_make_skill()],
            executor=FakeExecutor(),
            max_rounds=3,
        )

        assert order == ["hook", "chat"]
