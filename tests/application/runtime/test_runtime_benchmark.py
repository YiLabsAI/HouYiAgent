"""Performance benchmarks for Agent Runtime v2 components."""

from __future__ import annotations

import time

import pytest

from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.application.runtime.runner import AgentRunner
from houyi.application.runtime.sub_agent import SubAgentManager
from houyi.domain.agent.spec import AgentSpec


def _spec(role: str = "bench") -> AgentSpec:
    return AgentSpec(role=role)


class TestRuntimeBenchmarks:
    @pytest.mark.asyncio
    async def test_event_emit_latency(self):
        """EventEmitter emit p95 < 20ms."""
        emitter = EventEmitter()
        received: list[float] = []

        async def handler(ev: AgentEvent) -> None:
            received.append(time.monotonic())

        emitter.on(AgentEventType.PROGRESS, handler)

        latencies: list[float] = []
        for _ in range(100):
            start = time.monotonic()
            await emitter.emit(AgentEvent(event_type=AgentEventType.PROGRESS))
            latencies.append((time.monotonic() - start) * 1000)

        latencies.sort()
        p95 = latencies[94]
        assert p95 < 20, f"EventEmitter p95 = {p95:.1f}ms > 20ms"

    @pytest.mark.asyncio
    async def test_spawn_latency(self):
        """SubAgentManager spawn p95 < 200ms."""
        mgr = SubAgentManager()
        latencies: list[float] = []
        handles = []

        for _ in range(20):
            start = time.monotonic()
            h = await mgr.spawn(_spec(), "benchmark task")
            latencies.append((time.monotonic() - start) * 1000)
            handles.append(h)

        for h in handles:
            await mgr.join(h)

        latencies.sort()
        p95 = latencies[18]
        assert p95 < 200, f"spawn p95 = {p95:.1f}ms > 200ms"

    @pytest.mark.asyncio
    async def test_runner_latency(self):
        """AgentRunner run (mock LLM) p95 < 100ms."""
        runner = AgentRunner(_spec(), max_turns=2)
        latencies: list[float] = []

        for _ in range(20):
            start = time.monotonic()
            await runner.run("bench task")
            latencies.append((time.monotonic() - start) * 1000)

        latencies.sort()
        p95 = latencies[18]
        assert p95 < 100, f"Runner p95 = {p95:.1f}ms > 100ms"

    @pytest.mark.asyncio
    async def test_five_agent_stability(self):
        """5 concurrent agents complete without error."""
        mgr = SubAgentManager()
        agents = [(_spec(f"agent_{i}"), f"task {i}") for i in range(5)]
        handles = await mgr.spawn_parallel(agents)
        results = await mgr.join_all(handles)
        assert all(r.success for r in results)
        assert len(results) == 5
