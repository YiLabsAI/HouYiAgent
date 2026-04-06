"""Tests for AgentRunner: tool-loop execution and streaming."""

from __future__ import annotations

import pytest

from houyi.application.context.context_strategy import ContextStrategy
from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.application.runtime.runner import AgentResult, AgentRunner
from houyi.domain.agent.spec import AgentSpec


def _spec(role: str = "tester") -> AgentSpec:
    return AgentSpec(role=role)


class TestAgentRunner:
    @pytest.mark.asyncio
    async def test_run_mock(self):
        runner = AgentRunner(_spec(), max_turns=5)
        result = await runner.run("What is 2+2?")
        assert isinstance(result, AgentResult)
        assert result.success
        assert result.turns_used >= 1
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_emits_events(self):
        collected: list[AgentEvent] = []

        async def handler(ev: AgentEvent) -> None:
            collected.append(ev)

        emitter = EventEmitter()
        emitter.on_any(handler)

        runner = AgentRunner(_spec(), event_emitter=emitter)
        await runner.run("test task")

        types = {e.event_type for e in collected}
        assert AgentEventType.AGENT_STARTED in types
        assert AgentEventType.AGENT_COMPLETED in types

    @pytest.mark.asyncio
    async def test_run_stream(self):
        runner = AgentRunner(_spec())
        events: list[AgentEvent] = []
        async for ev in runner.run_stream("streaming task"):
            events.append(ev)
        assert any(e.event_type == AgentEventType.AGENT_STARTED for e in events)
        assert any(e.event_type == AgentEventType.AGENT_COMPLETED for e in events)

    @pytest.mark.asyncio
    async def test_context_truncation(self):
        strategy = ContextStrategy(keep_tool_result=2)
        runner = AgentRunner(_spec(), context_strategy=strategy)
        result = await runner.run("long task")
        assert result.success

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        runner = AgentRunner(_spec(), max_turns=1)
        result = await runner.run("task")
        assert result.turns_used <= 1

    @pytest.mark.asyncio
    async def test_callable_llm(self):
        async def mock_llm(messages):
            return "custom answer"

        runner = AgentRunner(_spec(), llm_adapter=mock_llm)
        result = await runner.run("question")
        assert result.success
        assert "custom answer" in str(result.output)

    @pytest.mark.asyncio
    async def test_run_fail_event(self):
        collected: list[AgentEvent] = []

        async def handler(ev: AgentEvent) -> None:
            collected.append(ev)

        emitter = EventEmitter()
        emitter.on_any(handler)
        runner = AgentRunner(_spec(), event_emitter=emitter)

        async def boom(_task: str, _state: dict[str, object]) -> tuple[object, int]:
            raise RuntimeError("boom")

        runner._tool_loop = boom  # type: ignore[method-assign]

        result = await runner.run("broken task")

        assert result.success is False
        assert result.error == "boom"
        assert any(event.event_type == AgentEventType.AGENT_FAILED for event in collected)

    @pytest.mark.asyncio
    async def test_tool_exec_json(self):
        def add(*, x: int, y: int) -> int:
            return x + y

        add.name = "add"  # type: ignore[attr-defined]
        runner = AgentRunner(_spec(), tools=[add])

        result = await runner._execute_tool(
            {
                "function": {
                    "name": "add",
                    "arguments": '{"x": 1, "y": 2}',
                }
            }
        )

        assert result == 3

    @pytest.mark.asyncio
    async def test_tool_exec_missing(self):
        runner = AgentRunner(_spec(), tools=[])

        result = await runner._execute_tool(
            {
                "function": {
                    "name": "missing",
                    "arguments": "not-json",
                }
            }
        )

        assert result == "Tool 'missing' not found"
