"""Planning with Files skill hooks.

This module implements the hooks for the planning-with-files skill:
- pre_write_hook: Called before Write/Edit operations
- post_tool_hook: Called after any tool use
- stop_hook: Called before stopping to verify completion

Reference: Based on planning-with-files Claude skill pattern.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.core.skill.hooks import HookContext, HookResult

logger = logging.getLogger(__name__)

# Default plan file name
PLAN_FILENAME = "PLAN.md"


def find_plan_file(cwd: Path | None = None) -> Path | None:
    """Find the plan file in the current directory or parent directories.
    
    Args:
        cwd: Current working directory (defaults to Path.cwd())
        
    Returns:
        Path to plan file if found, None otherwise
    """
    if cwd is None:
        cwd = Path.cwd()
    
    # Check current directory
    plan_path = cwd / PLAN_FILENAME
    if plan_path.exists():
        return plan_path
    
    # Check .plan directory
    plan_dir = cwd / ".plan"
    if plan_dir.exists():
        plan_path = plan_dir / PLAN_FILENAME
        if plan_path.exists():
            return plan_path
    
    # Check parent directories (up to 3 levels)
    for _ in range(3):
        cwd = cwd.parent
        if cwd == cwd.parent:  # Reached root
            break
        plan_path = cwd / PLAN_FILENAME
        if plan_path.exists():
            return plan_path
    
    return None


def parse_plan(content: str) -> dict[str, Any]:
    """Parse plan file content.
    
    Args:
        content: Plan file content
        
    Returns:
        Parsed plan with task, status, subtasks, and notes
    """
    result: dict[str, Any] = {
        "task": "",
        "status": "in_progress",
        "subtasks": [],
        "notes": "",
    }
    
    # Extract task title
    task_match = re.search(r"^#\s*Task:\s*(.+)$", content, re.MULTILINE)
    if task_match:
        result["task"] = task_match.group(1).strip()
    
    # Extract status
    status_match = re.search(r"^##\s*Status:\s*(\w+)", content, re.MULTILINE)
    if status_match:
        result["status"] = status_match.group(1).strip().lower()
    
    # Extract subtasks
    subtask_pattern = r"^-\s*\[([ xX])\]\s*(.+)$"
    for match in re.finditer(subtask_pattern, content, re.MULTILINE):
        completed = match.group(1).lower() == "x"
        description = match.group(2).strip()
        result["subtasks"].append({
            "description": description,
            "completed": completed,
        })
    
    # Extract notes section
    notes_match = re.search(r"^##\s*Notes\s*\n(.+?)(?=^##|\Z)", content, re.MULTILINE | re.DOTALL)
    if notes_match:
        result["notes"] = notes_match.group(1).strip()
    
    return result


def get_progress(plan: dict[str, Any]) -> dict[str, Any]:
    """Calculate progress from parsed plan.
    
    Args:
        plan: Parsed plan dictionary
        
    Returns:
        Progress dictionary with total, completed, percentage
    """
    subtasks = plan.get("subtasks", [])
    total = len(subtasks)
    completed = sum(1 for s in subtasks if s.get("completed", False))
    percentage = (completed / total * 100) if total > 0 else 0
    
    return {
        "total": total,
        "completed": completed,
        "remaining": total - completed,
        "percentage": round(percentage, 1),
    }


def is_plan_complete(plan: dict[str, Any]) -> bool:
    """Check if all subtasks are complete.
    
    Args:
        plan: Parsed plan dictionary
        
    Returns:
        True if all subtasks are complete
    """
    progress = get_progress(plan)
    return progress["remaining"] == 0


async def pre_write_hook(context: HookContext) -> HookResult:
    """Hook called before Write/Edit operations.
    
    Checks if the write operation aligns with the current plan.
    Can add context or warnings to the output.
    
    Args:
        context: Hook context with tool_name, tool_args, etc.
        
    Returns:
        HookResult with optional output message
    """
    from houyi.core.skill.hooks import HookResult
    
    cwd = context.cwd or Path.cwd()
    plan_path = find_plan_file(cwd)
    
    if not plan_path:
        # No plan file, proceed normally
        return HookResult(success=True)
    
    try:
        plan_content = plan_path.read_text()
        plan = parse_plan(plan_content)
        progress = get_progress(plan)
        
        # Add plan context to output
        output = (
            f"[Planning] Current task: {plan['task']}\n"
            f"[Planning] Progress: {progress['completed']}/{progress['total']} "
            f"({progress['percentage']}%)\n"
        )
        
        if progress["remaining"] > 0:
            # List remaining subtasks
            remaining = [s for s in plan["subtasks"] if not s.get("completed")]
            if remaining:
                output += f"[Planning] Remaining subtasks:\n"
                for i, subtask in enumerate(remaining[:3], 1):
                    output += f"  {i}. {subtask['description']}\n"
                if len(remaining) > 3:
                    output += f"  ... and {len(remaining) - 3} more\n"
        
        return HookResult(success=True, output=output)
        
    except Exception as e:
        logger.warning(f"Error in pre_write_hook: {e}")
        return HookResult(success=True)


async def post_tool_hook(context: HookContext) -> HookResult:
    """Hook called after any tool use.
    
    Updates plan progress if relevant changes were made.
    
    Args:
        context: Hook context with tool_name, tool_args, tool_result, etc.
        
    Returns:
        HookResult with optional output message
    """
    from houyi.core.skill.hooks import HookResult
    
    cwd = context.cwd or Path.cwd()
    plan_path = find_plan_file(cwd)
    
    if not plan_path:
        return HookResult(success=True)
    
    try:
        plan_content = plan_path.read_text()
        plan = parse_plan(plan_content)
        progress = get_progress(plan)
        
        # Check if we just completed a subtask
        tool_result = context.tool_result
        if isinstance(tool_result, dict) and tool_result.get("success"):
            # Could analyze the result to auto-update plan
            # For now, just report progress
            pass
        
        return HookResult(
            success=True,
            output=f"[Planning] Progress: {progress['completed']}/{progress['total']}"
        )
        
    except Exception as e:
        logger.warning(f"Error in post_tool_hook: {e}")
        return HookResult(success=True)


async def stop_hook(context: HookContext) -> HookResult:
    """Hook called before stopping.
    
    Verifies all planned subtasks are complete.
    If incomplete, returns a message prompting to continue.
    
    Args:
        context: Hook context
        
    Returns:
        HookResult with success=False if incomplete tasks remain
    """
    from houyi.core.skill.hooks import HookResult
    
    cwd = context.cwd or Path.cwd()
    plan_path = find_plan_file(cwd)
    
    if not plan_path:
        return HookResult(success=True)
    
    try:
        plan_content = plan_path.read_text()
        plan = parse_plan(plan_content)
        progress = get_progress(plan)
        
        if is_plan_complete(plan):
            return HookResult(
                success=True,
                output=f"[Planning] All {progress['total']} subtasks complete. Task finished!"
            )
        
        # Incomplete - return failure to potentially block stop
        remaining = [s for s in plan["subtasks"] if not s.get("completed")]
        remaining_list = "\n".join(f"  - {s['description']}" for s in remaining[:5])
        if len(remaining) > 5:
            remaining_list += f"\n  ... and {len(remaining) - 5} more"
        
        output = (
            f"[Planning] WARNING: Task incomplete!\n"
            f"[Planning] Progress: {progress['completed']}/{progress['total']} "
            f"({progress['percentage']}%)\n"
            f"[Planning] Remaining subtasks:\n{remaining_list}\n"
            f"[Planning] Consider completing remaining subtasks or explicitly marking task as done."
        )
        
        # Return success=True but with warning output
        # The actual blocking behavior is controlled by the host
        return HookResult(
            success=True,  # Don't hard-block, just warn
            output=output,
            metadata={"incomplete_tasks": len(remaining)},
        )
        
    except Exception as e:
        logger.warning(f"Error in stop_hook: {e}")
        return HookResult(success=True)


def create_plan(task: str, subtasks: list[str] | None = None) -> str:
    """Create a new plan file content.
    
    Args:
        task: Main task description
        subtasks: Optional list of subtask descriptions
        
    Returns:
        Plan file content as markdown
    """
    if subtasks is None:
        subtasks = []
    
    subtask_list = "\n".join(f"- [ ] {s}" for s in subtasks)
    
    return f"""# Task: {task}

## Status: in_progress

## Subtasks

{subtask_list if subtask_list else "- [ ] Define subtasks"}

## Notes

Plan created. Update subtasks as needed and mark them complete as you progress.
"""


def update_plan_subtask(
    content: str,
    subtask_index: int,
    completed: bool,
) -> str:
    """Update a subtask's completion status in plan content.
    
    Args:
        content: Current plan content
        subtask_index: 0-based index of subtask to update
        completed: New completion status
        
    Returns:
        Updated plan content
    """
    lines = content.split("\n")
    subtask_count = 0
    
    for i, line in enumerate(lines):
        if re.match(r"^-\s*\[([ xX])\]", line):
            if subtask_count == subtask_index:
                mark = "x" if completed else " "
                lines[i] = re.sub(r"\[([ xX])\]", f"[{mark}]", line)
                break
            subtask_count += 1
    
    return "\n".join(lines)
