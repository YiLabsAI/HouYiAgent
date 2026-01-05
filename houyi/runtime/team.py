"""Team runtime for multi-agent orchestration."""

from typing import Any, List, Optional

from houyi.runtime.agent import Agent
from houyi.runtime.task import Task


class Team:
    """Team runtime instance (multi-agent orchestration).
    
    Coordinates multiple agents working on multiple tasks.
    Supports task dependencies through context parameter.
    
    Alias: Workflow
    """

    def __init__(
        self,
        agents: List[Agent],
        tasks: List[Task],
        observability: Optional[dict] = None,
    ):
        self.agents = agents
        self.tasks = tasks
        self.observability = observability or {"enabled": True}
        
        # Validate tasks
        self._validate_tasks()

    def _validate_tasks(self) -> None:
        """Validate task configuration."""
        for task in self.tasks:
            if task.agent and task.agent not in self.agents:
                raise ValueError(
                    f"Task '{task.description}' references agent '{task.agent}' "
                    f"which is not in the team's agent list"
                )
    
    def run(self) -> Any:
        """Execute all tasks in the team with DAG-based dependency management.
        
        Returns:
            Execution results
        """
        results = {}
        completed_tasks = set()
        
        # Build dependency graph
        task_map = {i: task for i, task in enumerate(self.tasks)}
        
        # Execute tasks respecting dependencies
        while len(completed_tasks) < len(self.tasks):
            # Find ready tasks (all dependencies completed)
            ready_tasks = []
            for i, task in task_map.items():
                if i in completed_tasks:
                    continue
                
                # Check if all dependencies are completed
                deps_completed = all(dep in completed_tasks for dep in task.context)
                if deps_completed:
                    ready_tasks.append((i, task))
            
            if not ready_tasks:
                # Check for circular dependency
                if len(completed_tasks) < len(self.tasks):
                    raise RuntimeError(
                        f"Circular dependency detected. Completed {len(completed_tasks)}/{len(self.tasks)} tasks"
                    )
                break
            
            # Execute ready tasks
            for task_id, task in ready_tasks:
                # Get agent for this task
                agent = task.agent
                if not agent:
                    agent = self.agents[0] if self.agents else None
                
                if not agent:
                    raise ValueError(f"No agent available for task: {task.description}")
                
                # Build context from dependencies
                context_data = {
                    dep_id: results[dep_id]["result"] 
                    for dep_id in task.context 
                    if dep_id in results
                }
                
                # Execute task with context
                task_input = task.description
                if context_data:
                    task_input = f"{task.description}\n\nContext: {context_data}"
                
                result = agent.run(task)
                
                results[task_id] = {
                    "task": task.description,
                    "agent": agent.role if hasattr(agent, 'role') else str(agent),
                    "result": result,
                    "dependencies": task.context
                }
                
                completed_tasks.add(task_id)
        
        return {
            "tasks_completed": len(results),
            "results": list(results.values())
        }

    def execute(self) -> Any:
        """Alias for run() (for technical users)."""
        return self.run()


# Workflow is an alias for Team
Workflow = Team
