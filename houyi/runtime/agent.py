"""Agent runtime implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from houyi.core.agent import AgentSpec
from houyi.core.skill import SkillSpec
from houyi.orchestration.state import SessionState

if TYPE_CHECKING:
    from houyi.runtime.task import Task


class Agent:
    """Agent runtime instance.

    Wraps AgentSpec with execution capabilities.
    Manages session state and coordinates with executor.
    """

    def __init__(
        self,
        role: str,
        skills: list[SkillSpec] | None = None,
        llm: str = "gpt-4",
        memory: bool = False,
        system_prompt: str | None = None,
        observability: dict | None = None,
    ):
        # Create AgentSpec
        self.spec = AgentSpec(
            role=role,
            skills=skills or [],
            system_prompt=system_prompt,
            policies={"llm": llm, "memory": memory},
        )

        # Initialize observability
        self.observability_config = observability or {"enabled": True}
        self._init_trace_manager()

        # Runtime state
        self.state = SessionState(
            session_id=f"session_{id(self)}",
            agent_id=f"agent_{role}_{id(self)}",
        )

    @property
    def role(self) -> str:
        """Get agent role."""
        return self.spec.role

    @property
    def skills(self) -> list[SkillSpec]:
        """Get agent skills."""
        return self.spec.skills

    def _init_trace_manager(self) -> None:
        """Initialize trace manager."""
        from houyi.observability.trace_manager import TraceManager

        enabled = self.observability_config.get("enabled", True)
        exporters = self.observability_config.get("exporters", None)

        self.trace_manager = TraceManager(enabled=enabled, exporters=exporters)

    def _build_system_prompt(self) -> str:
        """Build system prompt from AgentSpec."""
        return self.spec.to_system_prompt()

    def run(self, input: str | Task) -> Any:
        """Execute task.

        Args:
            input: Task description (str) or Task object

        Returns:
            Execution result
        """
        import asyncio

        from houyi.execution.local_executor import LocalExecutor
        from houyi.orchestration.planner import DAGPlanner
        from houyi.runtime.task import Task

        # Parse input
        if isinstance(input, Task):
            description = input.description
            expected_output = input.expected_output
        else:
            description = input
            expected_output = None

        # Start trace
        with self.trace_manager.start_span(
            "agent.run",
            attributes={
                "agent.role": self.role,
                "agent.input": description,
                "agent.expected_output": expected_output,
            },
        ) as span:
            # Generate execution plan
            planner = DAGPlanner()
            plan = planner.plan(description, self.spec, self.state)

            span.set_attribute("plan.nodes", len(plan.nodes))

            # Execute plan
            executor = LocalExecutor(trace_manager=self.trace_manager)
            result = asyncio.run(executor.execute(plan, self.state))

            span.set_attribute("result.success", result.success)

            return result.output

    def get_tool_schemas(self) -> list[dict]:
        """Get OpenAI function calling schemas for all skills."""
        return [skill.to_tool_schema() for skill in self.skills]
