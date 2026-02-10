"""Knowledge base analyze skill hooks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.core.skill.hooks import HookContext, HookResult

logger = logging.getLogger(__name__)

# Analyze state tracking
_analyze_state: dict[str, Any] = {
    "files_scanned": 0,
    "indexes_checked": 0,
    "issues_found": [],
}


def reset_analyze_state() -> None:
    """Reset analyze state."""
    _analyze_state.clear()
    _analyze_state.update(
        {
            "files_scanned": 0,
            "indexes_checked": 0,
            "issues_found": [],
        }
    )


def get_analyze_state() -> dict[str, Any]:
    """Get current analyze state."""
    return _analyze_state.copy()


async def pre_analyze_hook(context: HookContext) -> HookResult:
    """Hook called before Read/Glob during analysis."""
    from houyi.core.skill.hooks import HookResult

    tool_name = context.tool_name or ""

    if tool_name.lower() == "glob":
        return HookResult(
            success=True,
            output="[kb-analyze] Scanning knowledge base structure...",
        )

    return HookResult(success=True)


async def post_analyze_hook(context: HookContext) -> HookResult:
    """Hook called after tool use during analysis."""
    from houyi.core.skill.hooks import HookResult

    tool_name = context.tool_name or ""
    tool_result = context.tool_result

    if tool_name.lower() == "glob" and tool_result:
        if isinstance(tool_result, list):
            _analyze_state["files_scanned"] += len(tool_result)
        elif isinstance(tool_result, dict):
            files = tool_result.get("files", [])
            _analyze_state["files_scanned"] += len(files)

    files = _analyze_state["files_scanned"]

    if files > 0:
        return HookResult(
            success=True,
            output=f"[kb-analyze] Scanned {files} files",
        )

    return HookResult(success=True)


async def stop_hook(context: HookContext) -> HookResult:
    """Hook called before stopping."""
    from houyi.core.skill.hooks import HookResult

    files = _analyze_state["files_scanned"]
    indexes = _analyze_state["indexes_checked"]
    issues = _analyze_state["issues_found"]

    output_parts = [
        f"[kb-analyze] Analysis complete: {files} files scanned, {indexes} indexes checked"
    ]

    if issues:
        output_parts.append(f"[kb-analyze] Found {len(issues)} issues")

    reset_analyze_state()

    return HookResult(
        success=True,
        output="\n".join(output_parts),
    )
