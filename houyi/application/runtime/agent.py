"""Runtime agent canonical module.

``Agent`` is the **user-facing entry point** for all HouYi agent execution —
single-agent tool loops, sub-agent delegation, and autonomous collaboration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from houyi.application.runtime.events import EventEmitter
from houyi.application.workflow.orchestration.state import SessionState
from houyi.domain.agent import AgentSpec, SubAgentConfig
from houyi.domain.skill.spec import SkillSpec

if TYPE_CHECKING:
    from houyi.application.runtime.task import Task


class Agent:
    """Agent runtime instance — the canonical SDK entry point.

    Wraps ``AgentSpec`` with execution capabilities.  Supports three
    execution paths, selected automatically by ``run()`` / ``arun()``:

    1. **Tool-loop** — agent has ``tools`` (no sub_agents): iterative
       LLM → tool-call → result loop via ``AgentRunner``.
    2. **Orchestrated** — agent has ``sub_agents`` or ``mode`` is set:
       ``AgentOrchestrator`` handles delegate / autonomous collaboration.
    3. **DAG** — fallback graph-based execution via planner + executor.

    Usage::

        # Tool-loop (single agent with tools)
        agent = Agent(role="Searcher", llm=llm, tools=[web_search_tool])
        result = await agent.arun("Find AI papers")

        # Delegate orchestration (supervisor + sub-agents)
        supervisor = Agent(
            role="Supervisor", llm=llm,
            sub_agents=[SubAgentConfig(role="Worker", ...)],
            mode="delegate",
            tools=[web_search_tool],
        )
        result = await supervisor.arun("Research topic")
    """

    def __init__(
        self,
        role: str,
        skills: list[SkillSpec] | None = None,
        llm: Any = None,
        memory: bool = False,
        system_prompt: str | None = None,
        observability: dict | None = None,
        *,
        sub_agents: list[SubAgentConfig] | None = None,
        max_turns: int = 50,
        mode: str | None = None,
        tools: list[Any] | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        self.spec = AgentSpec(
            role=role,
            skills=skills or [],
            system_prompt=system_prompt,
            policies={"llm": llm, "memory": memory},
            sub_agents=sub_agents or [],
            max_turns=max_turns,
        )

        self.mode = mode
        self._llm_adapter = llm
        self._tools = tools or []
        self._event_emitter = event_emitter
        self.observability_config = observability or {"enabled": True}
        self._init_trace_manager()

        self.state = SessionState(
            session_id=f"session_{id(self)}",
            agent_id=f"agent_{role}_{id(self)}",
        )

    @property
    def role(self) -> str:
        return self.spec.role

    @property
    def skills(self) -> list[SkillSpec]:
        return self.spec.skills

    def _init_trace_manager(self) -> None:
        from houyi.infrastructure.observability.trace_manager import TraceManager

        enabled = self.observability_config.get("enabled", True)
        exporters = self.observability_config.get("exporters", None)
        self.trace_manager = TraceManager(enabled=enabled, exporters=exporters)

    def _build_system_prompt(self) -> str:
        return self.spec.to_system_prompt()

    # ── Sync entry point ──────────────────────────────────────

    def run(self, input: str | Task) -> Any:
        """Synchronous entry point (wraps ``asyncio.run``).

        Use ``arun()`` when already inside an event loop (servers, notebooks).
        """
        from houyi.application.runtime.task import Task

        if isinstance(input, Task):
            description = input.description
        else:
            description = input

        if self.spec.sub_agents or self.mode in ("delegate", "autonomous"):
            return self._run_orchestrated(description)

        if self._tools:
            import asyncio

            return asyncio.run(self._arun_tool_loop(description))

        return self._run_dag(input)

    # ── Async entry point ─────────────────────────────────────

    async def arun(self, input: str | Task) -> Any:
        """Async entry point for use within existing event loops.

        Routing logic matches ``run()`` but avoids ``asyncio.run()``.
        """
        from houyi.application.runtime.task import Task

        description = input.description if isinstance(input, Task) else input

        if self.spec.sub_agents or self.mode in ("delegate", "autonomous"):
            return await self._arun_orchestrated(description)

        if self._tools or self._llm_adapter is not None:
            return await self._arun_tool_loop(description)

        return self._run_dag(input)

    # ── Execution paths ───────────────────────────────────────

    def _run_dag(self, input: str | Task) -> Any:
        """Classic DAG-based execution via planner + executor."""
        import asyncio

        from houyi.application.runtime.task import Task
        from houyi.application.workflow.executor import LocalExecutor
        from houyi.application.workflow.orchestration.planner import DAGPlanner

        if isinstance(input, Task):
            description = input.description
            expected_output = input.expected_output
        else:
            description = input
            expected_output = None

        with self.trace_manager.start_span(
            "agent.run",
            attributes={
                "agent.role": self.role,
                "agent.input": description,
                "agent.expected_output": expected_output,
            },
        ) as span:
            planner = DAGPlanner()
            plan = planner.plan(description, self.spec, self.state)
            span.set_attribute("plan.nodes", len(plan.nodes))

            executor = LocalExecutor(trace_manager=self.trace_manager)
            result = asyncio.run(executor.execute(plan, self.state))
            span.set_attribute("result.success", result.success)
            return result.output

    def _run_orchestrated(self, task: str) -> Any:
        """Orchestrated execution (sync wrapper)."""
        import asyncio

        return asyncio.run(self._arun_orchestrated(task))

    async def _arun_orchestrated(self, task: str) -> Any:
        """Orchestrated execution: delegate or autonomous via AgentOrchestrator."""
        from houyi.application.runtime.orchestrator import AgentOrchestrator
        from houyi.application.runtime.runner import AgentRunner
        from houyi.application.runtime.sub_agent import SubAgentManager

        runner = AgentRunner(
            self.spec,
            llm_adapter=self._llm_adapter,
            tools=self._tools,
            max_turns=self.spec.max_turns,
            event_emitter=self._event_emitter,
        )
        mgr = SubAgentManager(
            llm_adapter=self._llm_adapter,
            event_emitter=self._event_emitter,
        )
        orch = AgentOrchestrator(
            mgr,
            event_emitter=self._event_emitter,
            default_tools=self._tools,
        )

        sub_agents_map = {cfg.role: cfg for cfg in self.spec.sub_agents}

        if self.mode == "autonomous":
            result = await orch.run_autonomous(self.spec.sub_agents, task)
        else:
            result = await orch.run_delegate(runner, sub_agents_map, task)
        return result.output

    async def _arun_tool_loop(self, task: str) -> Any:
        """Execute the agent's tool-loop directly via AgentRunner."""
        from houyi.application.runtime.runner import AgentRunner

        runner = AgentRunner(
            self.spec,
            llm_adapter=self._llm_adapter,
            tools=self._tools,
            max_turns=self.spec.max_turns,
            event_emitter=self._event_emitter,
        )
        result = await runner.run(task)
        return result.output

    def get_tool_schemas(self) -> list[dict]:
        return [skill.to_tool_schema() for skill in self.skills]
