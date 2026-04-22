"""
Tests routing logic (tool-loop vs orchestrated vs DAG) and constructor
branches without making real LLM or tool calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.application.runtime.agent import Agent
from houyi.application.runtime.task import Task
from houyi.domain.agent import AgentTeamConfig

# ── Constructor ──────────────────────────────────────────────────────


class TestAgentInit:
    def test_minimal_init(self) -> None:
        agent = Agent(role="helper")
        assert agent.role == "helper"
        assert agent.skills == []
        assert agent.mode is None
        assert agent._tools == []

    def test_with_tools_and_llm(self) -> None:
        llm = MagicMock()
        agent = Agent(role="r", llm=llm, tools=[MagicMock()])
        assert agent._llm_adapter is llm
        assert len(agent._tools) == 1

    def test_team_agent_config(self) -> None:
        cfg = AgentTeamConfig(role="worker", skills=[], system_prompt="sp")
        agent = Agent(role="supervisor", team_agents=[cfg])
        assert len(agent.spec.team_agents) == 1
        assert agent.spec.team_agents[0].role == "worker"

    def test_team_agent_instance(self) -> None:
        worker = Agent(role="worker")
        agent = Agent(role="supervisor", team_agents=[worker])
        assert len(agent.spec.team_agents) == 1
        assert "worker" in agent._team_instances

    def test_observability_defaults(self) -> None:
        agent = Agent(role="r")
        assert agent.observability_config == {"enabled": True}

    def test_custom_observability(self) -> None:
        agent = Agent(role="r", observability={"enabled": False})
        assert agent.observability_config["enabled"] is False


# ── Properties ───────────────────────────────────────────────────────


class TestAgentProperties:
    def test_role_property(self) -> None:
        assert Agent(role="analyst").role == "analyst"

    def test_skills_property(self) -> None:
        assert Agent(role="r").skills == []

    def test_get_tool_schemas(self) -> None:
        assert Agent(role="r").get_tool_schemas() == []


# ── Routing: arun ────────────────────────────────────────────────────


class TestAgentArun:
    @pytest.mark.asyncio
    async def test_arun_routes_tool_loop(self) -> None:
        llm = AsyncMock()
        agent = Agent(role="r", llm=llm, tools=[MagicMock()])
        with patch.object(agent, "_arun_tool_loop", new_callable=AsyncMock) as mock:
            mock.return_value = "result"
            out = await agent.arun("task")
        assert out == "result"
        mock.assert_awaited_once_with("task")

    @pytest.mark.asyncio
    async def test_arun_routes_orchestrated(self) -> None:
        cfg = AgentTeamConfig(role="w", skills=[])
        agent = Agent(role="s", team_agents=[cfg])
        with patch.object(agent, "_arun_orchestrated", new_callable=AsyncMock) as mock:
            mock.return_value = "orch"
            out = await agent.arun("task")
        assert out == "orch"

    @pytest.mark.asyncio
    async def test_arun_routes_dag(self) -> None:
        agent = Agent(role="r")
        with patch.object(agent, "_run_dag") as mock:
            mock.return_value = "dag"
            out = await agent.arun("task")
        assert out == "dag"

    @pytest.mark.asyncio
    async def test_arun_accepts_task_obj(self) -> None:
        agent = Agent(role="r", llm=AsyncMock(), tools=[MagicMock()])
        with patch.object(agent, "_arun_tool_loop", new_callable=AsyncMock) as mock:
            mock.return_value = "r"
            task = Task(description="d", expected_output="e")
            await agent.arun(task)
        mock.assert_awaited_once_with("d")

    @pytest.mark.asyncio
    async def test_arun_llm_only_loop(self) -> None:
        agent = Agent(role="r", llm=MagicMock())
        with patch.object(agent, "_arun_tool_loop", new_callable=AsyncMock) as mock:
            mock.return_value = "ok"
            out = await agent.arun("q")
        assert out == "ok"


# ── Routing: run (sync) ──────────────────────────────────────────────


class TestAgentRun:
    def test_run_routes_orchestrated(self) -> None:
        cfg = AgentTeamConfig(role="w", skills=[])
        agent = Agent(role="s", team_agents=[cfg])
        with patch.object(agent, "_run_orchestrated") as mock:
            mock.return_value = "orch"
            out = agent.run("task")
        assert out == "orch"

    def test_run_routes_delegate_mode(self) -> None:
        agent = Agent(role="s", mode="delegate")
        with patch.object(agent, "_run_orchestrated") as mock:
            mock.return_value = "del"
            out = agent.run("t")
        assert out == "del"

    def test_run_routes_tool_loop(self) -> None:
        agent = Agent(role="r", tools=[MagicMock()])

        def _consume_coro(coro):
            # asyncio.run normally drives the coroutine; the mock must close it or the
            # tool-loop coroutine is left un-awaited (RuntimeWarning under xdist GC).
            if asyncio.iscoroutine(coro):
                coro.close()
            return "tl"

        with patch("asyncio.run", side_effect=_consume_coro):
            out = agent.run("task")
        assert out == "tl"

    def test_run_routes_dag(self) -> None:
        agent = Agent(role="r")
        with patch.object(agent, "_run_dag") as mock:
            mock.return_value = "dag"
            out = agent.run("t")
        assert out == "dag"

    def test_run_accepts_task_obj(self) -> None:
        agent = Agent(role="r")
        task = Task(description="d", expected_output="e")
        with patch.object(agent, "_run_dag") as mock:
            mock.return_value = "d"
            agent.run(task)
        mock.assert_called_once_with(task)


# ── Internal: _build_system_prompt ───────────────────────────────────


class TestBuildSystemPrompt:
    def test_returns_string(self) -> None:
        agent = Agent(role="r", system_prompt="You are helpful.")
        prompt = agent._build_system_prompt()
        assert isinstance(prompt, str)
