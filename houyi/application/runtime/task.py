"""Runtime task canonical module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from houyi.application.workflow.orchestration.state import TaskState, TaskStatus
from houyi.domain.task import TaskSpec

if TYPE_CHECKING:
    from houyi.application.runtime.agent import Agent


class Task:
    """Task definition (not executable independently)."""

    def __init__(
        self,
        description: str,
        agent: Agent | None = None,
        expected_output: str | None = None,
        context: list[int] | None = None,
    ):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.context = context

        self.spec = TaskSpec(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=context,
        )

        self.state = TaskState(
            task_id=f"task_{id(self)}",
            status=TaskStatus.PENDING,
            input_data={"description": description},
            output_data=None,
        )

    def __repr__(self) -> str:
        agent_info = f", agent={self.agent.role}" if self.agent else ""
        return f"Task(description='{self.description}'{agent_info})"
