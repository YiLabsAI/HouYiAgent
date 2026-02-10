"""Knowledge base ingest skill hooks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.core.skill.hooks import HookContext, HookResult

logger = logging.getLogger(__name__)

# Ingest state tracking
_ingest_state: dict[str, Any] = {
    "files_processed": 0,
    "chunks_created": 0,
    "errors": [],
}


def reset_ingest_state() -> None:
    """Reset ingest state."""
    _ingest_state.clear()
    _ingest_state.update(
        {
            "files_processed": 0,
            "chunks_created": 0,
            "errors": [],
        }
    )


def get_ingest_state() -> dict[str, Any]:
    """Get current ingest state."""
    return _ingest_state.copy()


async def pre_ingest_hook(context: HookContext) -> HookResult:
    """Hook called before Write operations during ingest."""
    from houyi.core.skill.hooks import HookResult

    tool_args = context.tool_args or {}
    file_path = tool_args.get("file_path", "")

    if file_path:
        # Check if writing to index directory
        if ".rag_index" in file_path:
            return HookResult(
                success=True,
                output=f"[kb-ingest] Writing index: {Path(file_path).name}",
            )

    return HookResult(success=True)


async def post_ingest_hook(context: HookContext) -> HookResult:
    """Hook called after tool use during ingest."""
    from houyi.core.skill.hooks import HookResult

    tool_name = context.tool_name or ""
    tool_result = context.tool_result

    if tool_name.lower() == "read" and tool_result:
        _ingest_state["files_processed"] += 1

    files = _ingest_state["files_processed"]
    chunks = _ingest_state["chunks_created"]

    if files > 0:
        return HookResult(
            success=True,
            output=f"[kb-ingest] Progress: {files} files processed, {chunks} chunks created",
        )

    return HookResult(success=True)


async def stop_hook(context: HookContext) -> HookResult:
    """Hook called before stopping."""
    from houyi.core.skill.hooks import HookResult

    files = _ingest_state["files_processed"]
    chunks = _ingest_state["chunks_created"]
    errors = _ingest_state["errors"]

    output_parts = [
        f"[kb-ingest] Ingest complete: {files} files processed, {chunks} chunks created"
    ]

    if errors:
        output_parts.append(f"[kb-ingest] Warnings: {len(errors)} issues encountered")

    reset_ingest_state()

    return HookResult(
        success=True,
        output="\n".join(output_parts),
    )
