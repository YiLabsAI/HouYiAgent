"""Tests for AgentOrchestrator: delegate, autonomous, sequential, parallel."""

from __future__ import annotations

import pytest

from houyi.application.runtime.error_policy import ErrorPolicy, FallbackStrategy
from houyi.application.runtime.events import EventEmitter
from houyi.application.runtime.orchestrator import (
    AgentOrchestrator,
    MergeStrategy,
    OrchestratorStage,
)
from houyi.application.runtime.runner import AgentRunner
from houyi.application.runtime.sub_agent import SubAgentManager
from houyi.domain.agent.spec import AgentSpec, SubAgentConfig


def _spec(role: str = "test") -> AgentSpec:
    return AgentSpec(role=role)


def _cfg(role: str = "sub") -> SubAgentConfig:
    return SubAgentConfig(role=role, max_turns=3)


def _orch(**kwargs) -> AgentOrchestrator:
    mgr = SubAgentManager()
    return AgentOrchestrator(mgr, **kwargs)


class TestRunDelegate:
    @pytest.mark.asyncio
    async def test_basic_delegate(self):
        orch = _orch()
        runner = AgentRunner(_spec("main"), max_turns=3)
        subs = {"analyst": _cfg("analyst"), "writer": _cfg("writer")}
        result = await orch.run_delegate(runner, subs, "research AI trends")
        assert result.success
        assert len(result.agent_results) >= 1
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_delegate_empty_subs(self):
        orch = _orch()
        runner = AgentRunner(_spec("main"), max_turns=2)
        result = await orch.run_delegate(runner, {}, "no sub-agents")
        assert result.success


class TestRunAutonomous:
    @pytest.mark.asyncio
    async def test_basic_autonomous(self):
        orch = _orch()
        agents = [_cfg("explorer"), _cfg("verifier")]
        result = await orch.run_autonomous(agents, "find facts about Python", max_rounds=2)
        assert result.success
        assert len(result.agent_results) >= 2
        assert result.metadata.get("rounds") is not None

    @pytest.mark.asyncio
    async def test_autonomous_state(self):
        orch = _orch()
        agents = [_cfg("a1")]
        result = await orch.run_autonomous(agents, "task", max_rounds=2)
        sid = result.metadata.get("state_id")
        assert sid is not None
        state = await orch.state_backend.read(sid)
        assert state.status == "completed"


class TestRunSequential:
    @pytest.mark.asyncio
    async def test_two_stages(self):
        orch = _orch()
        stages = [
            OrchestratorStage(spec=_cfg("gatherer"), task_template="Gather data on AI"),
            OrchestratorStage(
                spec=_cfg("summarizer"), task_template="Summarize: {previous_output}"
            ),
        ]
        result = await orch.run_sequential(stages)
        assert result.success
        assert len(result.agent_results) == 2

    @pytest.mark.asyncio
    async def test_abort_on_failure(self):
        policy = ErrorPolicy(fallback_strategy=FallbackStrategy.ABORT)
        orch = _orch(error_policy=policy)
        stages = [OrchestratorStage(spec=_cfg("s1"), task_template="stage 1")]
        result = await orch.run_sequential(stages)
        assert result.success


class TestRunParallel:
    @pytest.mark.asyncio
    async def test_concat_merge(self):
        orch = _orch()
        tasks = [
            (_cfg("a1"), "search topic A"),
            (_cfg("a2"), "search topic B"),
        ]
        result = await orch.run_parallel(tasks, merge_strategy=MergeStrategy.CONCAT)
        assert result.success
        assert isinstance(result.output, list)
        assert len(result.output) == 2

    @pytest.mark.asyncio
    async def test_first_success(self):
        orch = _orch()
        tasks = [(_cfg("a1"), "task A"), (_cfg("a2"), "task B")]
        result = await orch.run_parallel(tasks, merge_strategy=MergeStrategy.FIRST_SUCCESS)
        assert result.success
        assert result.output is not None


class TestConflictDetection:
    @pytest.mark.asyncio
    async def test_delegate_detects_conflicts(self):
        orch = _orch()
        runner = AgentRunner(_spec("main"), max_turns=2)
        subs = {"a1": _cfg("a1"), "a2": _cfg("a2")}
        result = await orch.run_delegate(runner, subs, "controversial topic")
        assert isinstance(result.conflicts, list)


class TestEventEmission:
    @pytest.mark.asyncio
    async def test_delegate_emits(self):
        collected = []

        async def h(ev):
            collected.append(ev)

        emitter = EventEmitter()
        emitter.on_any(h)
        orch = _orch(event_emitter=emitter)
        runner = AgentRunner(_spec("m"), max_turns=2)
        await orch.run_delegate(runner, {"s": _cfg("s")}, "task")
        assert len(collected) >= 2
