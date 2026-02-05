"""Planning with Files skill.

This skill helps manage complex tasks through structured plan files.
It tracks task breakdown, progress, and completion status.

Usage:
    from houyi.skills.planning import PlanningSkill

    skill = PlanningSkill()
    registry.register(skill.to_spec())
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from houyi.core.skill import SkillSpec
from houyi.skills.planning.hooks import (
    PLAN_FILENAME,
    create_plan,
    find_plan_file,
    get_progress,
    is_plan_complete,
    parse_plan,
    update_plan_subtask,
)

__all__ = [
    "PlanningSkill",
    "PLAN_FILENAME",
    "create_plan",
    "find_plan_file",
    "get_progress",
    "is_plan_complete",
    "parse_plan",
    "update_plan_subtask",
]


class PlanningSkill:
    """Planning with Files skill.

    Provides task planning and tracking through markdown files.
    """

    def __init__(self, workspace: Path | None = None) -> None:
        """Initialize planning skill.

        Args:
            workspace: Workspace directory for plan files
        """
        self.workspace = workspace or Path.cwd()
        self.skill_dir = Path(__file__).parent

    def to_spec(self) -> SkillSpec:
        """Convert to SkillSpec for registration.

        Returns:
            SkillSpec instance
        """
        skill_md_path = self.skill_dir / "SKILL.md"
        return SkillSpec.from_file(str(skill_md_path), skill_dir=str(self.skill_dir))

    async def execute(self, action: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a planning action.

        Args:
            action: Action to perform (create, update, complete, status)
            **kwargs: Action-specific arguments

        Returns:
            Result dictionary
        """
        if action == "create":
            return await self._create_plan(**kwargs)
        elif action == "update":
            return await self._update_subtask(**kwargs)
        elif action == "complete":
            return await self._complete_plan(**kwargs)
        elif action == "status":
            return await self._get_status(**kwargs)
        else:
            return {
                "success": False,
                "message": f"Unknown action: {action}",
            }

    async def _create_plan(
        self,
        task: str,
        subtasks: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Create a new plan file.

        Args:
            task: Task description
            subtasks: Optional list of subtasks

        Returns:
            Result with plan_path and status
        """
        plan_dir = self.workspace / ".plan"
        plan_dir.mkdir(exist_ok=True)
        plan_path = plan_dir / PLAN_FILENAME

        content = create_plan(task, subtasks)
        plan_path.write_text(content)

        return {
            "success": True,
            "plan_path": str(plan_path),
            "status": "in_progress",
            "message": f"Created plan for: {task}",
        }

    async def _update_subtask(
        self,
        subtask_index: int,
        completed: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        """Update a subtask's status.

        Args:
            subtask_index: 0-based index of subtask
            completed: Whether subtask is complete

        Returns:
            Result with updated status
        """
        plan_path = find_plan_file(self.workspace)
        if not plan_path:
            return {
                "success": False,
                "message": "No plan file found",
            }

        content = plan_path.read_text()
        updated_content = update_plan_subtask(content, subtask_index, completed)
        plan_path.write_text(updated_content)

        plan = parse_plan(updated_content)
        progress = get_progress(plan)

        return {
            "success": True,
            "plan_path": str(plan_path),
            "progress": progress,
            "message": f"Updated subtask {subtask_index + 1}",
        }

    async def _complete_plan(self, **_: Any) -> dict[str, Any]:
        """Mark plan as complete.

        Returns:
            Result with final status
        """
        plan_path = find_plan_file(self.workspace)
        if not plan_path:
            return {
                "success": False,
                "message": "No plan file found",
            }

        content = plan_path.read_text()
        plan = parse_plan(content)
        progress = get_progress(plan)

        if not is_plan_complete(plan):
            return {
                "success": False,
                "progress": progress,
                "message": f"Cannot complete: {progress['remaining']} subtasks remaining",
            }

        # Update status to completed
        updated_content = content.replace(
            "## Status: in_progress",
            "## Status: completed"
        )
        plan_path.write_text(updated_content)

        return {
            "success": True,
            "plan_path": str(plan_path),
            "status": "completed",
            "progress": progress,
            "message": "Plan completed successfully!",
        }

    async def _get_status(self, **_: Any) -> dict[str, Any]:
        """Get current plan status.

        Returns:
            Result with status and progress
        """
        plan_path = find_plan_file(self.workspace)
        if not plan_path:
            return {
                "success": False,
                "message": "No plan file found",
            }

        content = plan_path.read_text()
        plan = parse_plan(content)
        progress = get_progress(plan)

        return {
            "success": True,
            "plan_path": str(plan_path),
            "task": plan["task"],
            "status": plan["status"],
            "progress": progress,
            "subtasks": plan["subtasks"],
        }
