"""Runtime team canonical module."""

from typing import Any

from houyi.application.runtime.agent import Agent
from houyi.application.runtime.task import Task


class Team:
    """Team runtime instance (multi-agent orchestration)."""

    def __init__(
        self,
        agents: list[Agent],
        tasks: list[Task],
        observability: dict | None = None,
    ):
        self.agents = agents
        self.tasks = tasks
        self.observability = observability or {"enabled": True}
        self._validate_tasks()

    def _validate_tasks(self) -> None:
        for task in self.tasks:
            if task.agent and task.agent not in self.agents:
                raise ValueError(
                    f"Task '{task.description}' references agent '{task.agent}' "
                    f"which is not in the team's agent list"
                )

    def _get_ready_tasks(
        self,
        *,
        task_map: dict[int, Task],
        completed_tasks: set[int],
    ) -> list[tuple[int, Task]]:
        ready_tasks: list[tuple[int, Task]] = []
        for task_id, task in task_map.items():
            if task_id in completed_tasks:
                continue

            deps_completed = all(dep in completed_tasks for dep in (task.context or []))
            if deps_completed:
                ready_tasks.append((task_id, task))
        return ready_tasks

    def _execute_task(
        self,
        *,
        task_id: int,
        task: Task,
        results: dict[int, dict[str, Any]],
    ) -> None:
        agent = task.agent
        if not agent:
            agent = self.agents[0] if self.agents else None

        if not agent:
            raise ValueError(f"No agent available for task: {task.description}")

        context_data = {
            dep_id: results[dep_id]["result"]
            for dep_id in (task.context or [])
            if dep_id in results
        }

        if context_data:
            pass

        result = agent.run(task)
        results[task_id] = {
            "task": task.description,
            "agent": agent.role if hasattr(agent, "role") else str(agent),
            "result": result,
            "dependencies": task.context or [],
        }

    def run(self) -> Any:
        results = {}
        completed_tasks = set()
        task_map = dict(enumerate(self.tasks))

        while len(completed_tasks) < len(self.tasks):
            ready_tasks = self._get_ready_tasks(task_map=task_map, completed_tasks=completed_tasks)

            if not ready_tasks:
                if len(completed_tasks) < len(self.tasks):
                    raise RuntimeError(
                        f"Circular dependency detected. Completed {len(completed_tasks)}/{len(self.tasks)} tasks"
                    )
                break

            for task_id, task in ready_tasks:
                self._execute_task(task_id=task_id, task=task, results=results)
                completed_tasks.add(task_id)

        return {"tasks_completed": len(results), "results": list(results.values())}

    def execute(self) -> Any:
        return self.run()


Workflow = Team
