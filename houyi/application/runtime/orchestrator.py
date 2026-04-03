"""AgentOrchestrator: dual-mode orchestration engine (Delegate + Autonomous).

Conceptual hierarchy:
  - **Orchestration patterns** (primary API): ``run_delegate``, ``run_autonomous``
  - **Execution primitives** (building blocks): ``run_sequential``, ``run_parallel``

Delegate and Autonomous are the two main modes of multi-agent collaboration.
Sequential and Parallel are execution strategies used within them (or standalone
for simple scenarios).
"""

from __future__ import annotations

import logging
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from houyi.application.runtime.agent_team import AgentTeamManager
from houyi.application.runtime.error_policy import (
    AgentTaskResult,
    ConflictResolver,
    ErrorPolicy,
    FallbackStrategy,
)
from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.application.runtime.message_bus import AgentMessageBus
from houyi.application.runtime.runner import AgentRunner
from houyi.application.runtime.shared_state import InMemoryStateBackend, SharedStateBackend
from houyi.domain.agent.spec import AgentSpec, AgentTeamConfig

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """How to merge results from parallel agents."""

    CONCAT = "concat"
    VOTE = "vote"
    FIRST_SUCCESS = "first_success"


class OrchestratorStage(BaseModel):
    """A single stage in a sequential pipeline."""

    spec: AgentSpec | AgentTeamConfig
    task_template: str = ""
    tools: list[Any] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class OrchestratorResult(BaseModel):
    """Aggregated result from an orchestration run."""

    orchestration_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    success: bool = True
    output: Any = None
    agent_results: list[AgentTaskResult] = Field(default_factory=list)
    conflicts: list[Any] = Field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentOrchestrator:
    """Dual-mode orchestration engine for multi-agent collaboration.

    **Orchestration patterns** (primary):
      - ``run_delegate``: One main agent delegates tasks to team agents.
      - ``run_autonomous``: Peer agents collaborate via shared state
        and message bus in iterative rounds.

    **Execution primitives** (building blocks):
      - ``run_sequential``: Execute stages one after another, each receiving
        the previous stage's output.
      - ``run_parallel``: Execute agents concurrently and merge results.

    The orchestrator integrates ``SharedStateBackend``, ``AgentMessageBus``,
    ``ErrorPolicy``, and ``ConflictResolver`` for production-grade execution.
    """

    def __init__(
        self,
        team_manager: AgentTeamManager,
        *,
        message_bus: AgentMessageBus | None = None,
        state_backend: SharedStateBackend | None = None,
        event_emitter: EventEmitter | None = None,
        error_policy: ErrorPolicy | None = None,
        conflict_resolver: ConflictResolver | None = None,
        default_tools: list[Any] | None = None,
    ) -> None:
        self.team_manager = team_manager
        self.message_bus = message_bus or AgentMessageBus()
        self.state_backend = state_backend or InMemoryStateBackend()
        self.event_emitter = event_emitter
        self.error_policy = error_policy or ErrorPolicy()
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.default_tools = default_tools or []

    # ── Orchestration Patterns (primary) ───────────────────────

    async def run_delegate(
        self,
        main_runner: AgentRunner,
        agents: dict[str, AgentTeamConfig | Any],
        task: str,
    ) -> OrchestratorResult:
        """Delegate mode: main agent decides sub-tasks, delegates, synthesises.

        *agents* values may be ``AgentTeamConfig`` or ``Agent`` instances.

        1. Main runner produces a plan / task decomposition.
        2. Team agents execute their assigned tasks in parallel.
        3. Main runner synthesises the results into a final answer.
        """
        start = time.perf_counter()
        orch_id = uuid.uuid4().hex[:10]
        await self._emit(AgentEventType.AGENT_STARTED, {"mode": "delegate", "task": task})

        main_result = await main_runner.run(task)
        sub_tasks = self._decompose(main_result.output, agents)

        handles = []
        for agent_name, sub_task in sub_tasks.items():
            spec = agents.get(agent_name)
            if spec is None:
                continue
            h = await self.team_manager.spawn(spec, sub_task, tools=self.default_tools)
            handles.append(h)

        agent_results_raw = await self.team_manager.join_all(handles)
        agent_results = await self._apply_error_policy(agent_results_raw)

        conflicts = await self.conflict_resolver.detect(
            [AgentTaskResult(**r.model_dump()) for r in agent_results]
        )
        for c in conflicts:
            c.resolution = await self.conflict_resolver.resolve(c)

        synthesis_task = self._build_synthesis_prompt(task, agent_results)
        final = await main_runner.run(synthesis_task)

        elapsed = (time.perf_counter() - start) * 1000
        await self._emit(
            AgentEventType.AGENT_COMPLETED, {"mode": "delegate", "duration_ms": elapsed}
        )

        return OrchestratorResult(
            orchestration_id=orch_id,
            success=final.success,
            output=final.output,
            agent_results=[AgentTaskResult(**r.model_dump()) for r in agent_results],
            conflicts=conflicts,
            duration_ms=elapsed,
        )

    async def run_autonomous(
        self,
        agents: list[AgentTeamConfig | Any],
        task: str,
        *,
        state_id: str | None = None,
        max_rounds: int = 10,
    ) -> OrchestratorResult:
        """Autonomous mode: peer agents collaborate via shared state + message bus.

        Each round:
          1. All agents read shared state and process bus messages.
          2. Agents produce findings that update shared state.
          3. Orchestrator checks convergence / budget.
        """
        start = time.perf_counter()
        orch_id = uuid.uuid4().hex[:10]
        sid = state_id or f"state_{orch_id}"
        await self._emit(AgentEventType.AGENT_STARTED, {"mode": "autonomous", "task": task})

        await self.state_backend.write(sid, {"task": task, "status": "running"})

        for agent_cfg in agents:
            self.message_bus.register_agent(f"agent_{agent_cfg.role}")

        all_results: list[AgentTaskResult] = []

        for round_num in range(1, max_rounds + 1):
            state = await self.state_backend.read(sid)
            await self._emit(AgentEventType.PROGRESS, {"round": round_num, "state": state.status})

            handles = []
            for cfg in agents:
                round_task = f"[Round {round_num}] {task}\nCurrent findings: {len(state.findings)}"
                h = await self.team_manager.spawn(cfg, round_task, tools=self.default_tools)
                handles.append(h)

            round_results = await self.team_manager.join_all(handles)
            all_results.extend([AgentTaskResult(**r.model_dump()) for r in round_results])

            findings = [
                {"agent": r.agent_id, "round": round_num, "output": str(r.output)[:500]}
                for r in round_results
                if r.success
            ]
            await self.state_backend.write(sid, {"findings": findings})

            if self._check_convergence(round_results, round_num, max_rounds):
                break

        await self.state_backend.write(sid, {"status": "completed"})
        final_state = await self.state_backend.read(sid)

        conflicts = await self.conflict_resolver.detect(all_results)
        for c in conflicts:
            c.resolution = await self.conflict_resolver.resolve(c)

        elapsed = (time.perf_counter() - start) * 1000
        await self._emit(
            AgentEventType.AGENT_COMPLETED, {"mode": "autonomous", "duration_ms": elapsed}
        )

        return OrchestratorResult(
            orchestration_id=orch_id,
            success=True,
            output=final_state.findings,
            agent_results=all_results,
            conflicts=conflicts,
            duration_ms=elapsed,
            metadata={"rounds": round_num, "state_id": sid},
        )

    # ── Execution Primitives (building blocks) ─────────────────

    async def run_sequential(
        self,
        stages: list[OrchestratorStage],
        *,
        initial_context: dict[str, Any] | None = None,
    ) -> OrchestratorResult:
        """Execute stages sequentially, piping each output to the next."""
        start = time.perf_counter()
        ctx = initial_context or {}
        results: list[AgentTaskResult] = []

        for i, stage in enumerate(stages):
            task = stage.task_template.format(**ctx) if ctx else stage.task_template
            h = await self.team_manager.spawn(stage.spec, task, tools=stage.tools)
            r = await self.team_manager.join(h)
            results.append(AgentTaskResult(**r.model_dump()))
            ctx["previous_output"] = r.output
            ctx[f"stage_{i}_output"] = r.output

            if not r.success and self.error_policy.fallback_strategy == FallbackStrategy.ABORT:
                break

        elapsed = (time.perf_counter() - start) * 1000
        return OrchestratorResult(
            success=all(r.success for r in results),
            output=results[-1].output if results else None,
            agent_results=results,
            duration_ms=elapsed,
        )

    async def run_parallel(
        self,
        tasks: list[tuple[AgentTeamConfig, str]],
        *,
        merge_strategy: MergeStrategy = MergeStrategy.CONCAT,
        max_concurrent: int = 5,
    ) -> OrchestratorResult:
        """Execute agents concurrently and merge results."""
        start = time.perf_counter()
        handles = await self.team_manager.spawn_parallel(
            [(cfg, t) for cfg, t in tasks],
            max_concurrent=max_concurrent,
        )
        raw_results = await self.team_manager.join_all(handles)
        results = [AgentTaskResult(**r.model_dump()) for r in raw_results]

        merged = self._merge_results(results, merge_strategy)

        elapsed = (time.perf_counter() - start) * 1000
        return OrchestratorResult(
            success=any(r.success for r in results),
            output=merged,
            agent_results=results,
            duration_ms=elapsed,
        )

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _decompose(main_output: Any, agents: dict[str, Any]) -> dict[str, str]:
        """Simple decomposition: assign the main output as task to each agent."""
        text = str(main_output) if main_output else ""
        return {name: f"Task for {name}: {text}" for name in agents}

    @staticmethod
    def _build_synthesis_prompt(original_task: str, results: list[Any]) -> str:
        parts = [f"Original task: {original_task}\n\nAgent results:"]
        for r in results:
            parts.append(f"- Agent {r.agent_id}: {str(r.output)[:300]}")
        parts.append("\nSynthesise these into a final answer.")
        return "\n".join(parts)

    async def _apply_error_policy(self, results: list[Any]) -> list[Any]:
        """Apply error policy: retry or fallback for failed results."""
        final: list[Any] = []
        for r in results:
            if r.success:
                final.append(r)
            elif self.error_policy.fallback_strategy == FallbackStrategy.SKIP:
                logger.warning("Skipping failed agent %s: %s", r.agent_id, r.error)
            elif self.error_policy.fallback_strategy == FallbackStrategy.ABORT:
                raise RuntimeError(f"Agent {r.agent_id} failed: {r.error}")
            else:
                final.append(r)
        return final

    @staticmethod
    def _check_convergence(results: list[Any], round_num: int, max_rounds: int) -> bool:
        """Simple convergence: all succeeded or max rounds reached."""
        if round_num >= max_rounds:
            return True
        return all(r.success for r in results) and round_num >= 2

    @staticmethod
    def _merge_results(results: list[AgentTaskResult], strategy: MergeStrategy) -> Any:
        if strategy == MergeStrategy.CONCAT:
            return [r.output for r in results if r.success]
        if strategy == MergeStrategy.FIRST_SUCCESS:
            for r in results:
                if r.success:
                    return r.output
            return None
        return [r.output for r in results if r.success]

    async def _emit(self, event_type: AgentEventType, data: dict[str, Any]) -> None:
        if self.event_emitter is None:
            return
        await self.event_emitter.emit(AgentEvent(event_type=event_type, data=data))
