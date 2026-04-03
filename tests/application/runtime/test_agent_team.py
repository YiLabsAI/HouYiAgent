"""Tests for AgentTeamManager: spawn, join, parallel, terminate."""

from __future__ import annotations

import pytest

from houyi.application.runtime.agent_team import AgentTeamManager, TeamAgentHandle, TeamAgentStatus
from houyi.application.runtime.events import EventEmitter
from houyi.domain.agent.spec import AgentSpec, AgentTeamConfig


def _spec(role: str = "worker") -> AgentSpec:
    return AgentSpec(role=role)


class TestAgentTeamManager:
    @pytest.mark.asyncio
    async def test_spawn_join(self):
        mgr = AgentTeamManager()
        handle = await mgr.spawn(_spec(), "do something")
        assert handle.status == TeamAgentStatus.RUNNING
        result = await mgr.join(handle)
        assert result.success
        assert handle.status == TeamAgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_spawn_config(self):
        cfg = AgentTeamConfig(role="analyzer", max_turns=5)
        mgr = AgentTeamManager()
        handle = await mgr.spawn(cfg, "analyze data")
        result = await mgr.join(handle)
        assert result.success

    @pytest.mark.asyncio
    async def test_spawn_parallel(self):
        mgr = AgentTeamManager()
        agents = [
            (_spec("w1"), "task 1"),
            (_spec("w2"), "task 2"),
            (_spec("w3"), "task 3"),
        ]
        handles = await mgr.spawn_parallel(agents)
        assert len(handles) == 3
        results = await mgr.join_all(handles)
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_join_missing(self):
        mgr = AgentTeamManager()
        fake = TeamAgentHandle(handle_id="missing", agent_id="x")
        result = await mgr.join(fake)
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_terminate(self):
        mgr = AgentTeamManager()
        handle = await mgr.spawn(_spec(), "long task")
        await mgr.terminate(handle)
        assert handle.status == TeamAgentStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_events_emitted(self):
        collected = []

        async def h(ev):
            collected.append(ev)

        emitter = EventEmitter()
        emitter.on_any(h)
        mgr = AgentTeamManager(event_emitter=emitter)
        handle = await mgr.spawn(_spec(), "task")
        await mgr.join(handle)
        types = {e.event_type.value for e in collected}
        assert "team_agent.spawned" in types
        assert "team_agent.completed" in types
