"""Runtime agent canonical module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from houyi.application.workflow.orchestration.state import SessionState
from houyi.domain.agent import AgentSpec
from houyi.domain.skill.spec import SkillSpec

if TYPE_CHECKING:
    from houyi.application.runtime.task import Task


class Agent:
    """Agent runtime instance.

    Wraps AgentSpec with execution capabilities.
    Manages session state and coordinates with executor.
    """

    def __init__(
        self,
        role: str,
        skills: list[SkillSpec] | None = None,
        llm: Any = None,
        memory: bool = False,
        system_prompt: str | None = None,
        observability: dict | None = None,
    ):
        self.spec = AgentSpec(
            role=role,
            skills=skills or [],
            system_prompt=system_prompt,
            policies={"llm": llm, "memory": memory},
        )

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

    def run(self, input: str | Task) -> Any:
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

    def get_tool_schemas(self) -> list[dict]:
        return [skill.to_tool_schema() for skill in self.skills]
