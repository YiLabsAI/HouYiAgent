"""Tests for EventEmitter and AgentEvent."""

from __future__ import annotations

import pytest

from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter


class TestAgentEvent:
    def test_default_timestamp(self):
        ev = AgentEvent(event_type=AgentEventType.AGENT_STARTED)
        assert ev.timestamp > 0

    def test_custom_fields(self):
        ev = AgentEvent(
            event_type=AgentEventType.TOOL_COMPLETED,
            agent_id="a1",
            agent_name="Researcher",
            data={"tool": "search"},
        )
        assert ev.agent_id == "a1"
        assert ev.data["tool"] == "search"


class TestEventEmitter:
    @pytest.mark.asyncio
    async def test_emit_typed(self):
        collected: list[AgentEvent] = []

        async def handler(ev: AgentEvent) -> None:
            collected.append(ev)

        emitter = EventEmitter()
        emitter.on(AgentEventType.AGENT_STARTED, handler)
        await emitter.emit(AgentEvent(event_type=AgentEventType.AGENT_STARTED))
        assert len(collected) == 1

    @pytest.mark.asyncio
    async def test_wildcard_handler(self):
        collected: list[AgentEvent] = []

        async def handler(ev: AgentEvent) -> None:
            collected.append(ev)

        emitter = EventEmitter()
        emitter.on_any(handler)
        await emitter.emit(AgentEvent(event_type=AgentEventType.AGENT_STARTED))
        await emitter.emit(AgentEvent(event_type=AgentEventType.TOOL_COMPLETED))
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_off_removes(self):
        collected: list[AgentEvent] = []

        async def handler(ev: AgentEvent) -> None:
            collected.append(ev)

        emitter = EventEmitter()
        emitter.on(AgentEventType.AGENT_STARTED, handler)
        emitter.off(AgentEventType.AGENT_STARTED, handler)
        await emitter.emit(AgentEvent(event_type=AgentEventType.AGENT_STARTED))
        assert len(collected) == 0

    @pytest.mark.asyncio
    async def test_error_suppressed(self):
        async def bad_handler(ev: AgentEvent) -> None:
            raise RuntimeError("boom")

        emitter = EventEmitter()
        emitter.on(AgentEventType.AGENT_STARTED, bad_handler)
        await emitter.emit(AgentEvent(event_type=AgentEventType.AGENT_STARTED))

    @pytest.mark.asyncio
    async def test_no_listeners_noop(self):
        emitter = EventEmitter()
        await emitter.emit(AgentEvent(event_type=AgentEventType.AGENT_STARTED))

    @pytest.mark.asyncio
    async def test_multi_listeners(self):
        results: list[str] = []

        async def h1(ev: AgentEvent) -> None:
            results.append("h1")

        async def h2(ev: AgentEvent) -> None:
            results.append("h2")

        emitter = EventEmitter()
        emitter.on(AgentEventType.PROGRESS, h1)
        emitter.on(AgentEventType.PROGRESS, h2)
        await emitter.emit(AgentEvent(event_type=AgentEventType.PROGRESS))
        assert set(results) == {"h1", "h2"}

    @pytest.mark.asyncio
    async def test_emit_sync_ordered(self):
        results: list[int] = []

        async def h1(ev: AgentEvent) -> None:
            results.append(1)

        async def h2(ev: AgentEvent) -> None:
            results.append(2)

        emitter = EventEmitter()
        emitter.on(AgentEventType.PROGRESS, h1)
        emitter.on(AgentEventType.PROGRESS, h2)
        await emitter.emit_sync(AgentEvent(event_type=AgentEventType.PROGRESS))
        assert results == [1, 2]
