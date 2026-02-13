"""Knowledge base search skill hooks.

This module implements the hooks for the kb-search skill:
- pre_search_hook: Called before Read/Glob/Grep operations
- post_search_hook: Called after any tool use
- stop_hook: Called before stopping to verify search completion

Reference: Based on SimpleSkill v0.1 specification.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.core.skill.hooks import HookContext, HookResult

logger = logging.getLogger(__name__)

# Search state tracking (per session)
_search_state: dict[str, Any] = {
    "query": "",
    "knowledge_dir": "",
    "files_searched": 0,
    "matches_found": 0,
    "sources_collected": [],
}


def reset_search_state() -> None:
    """Reset search state for new search session."""
    _search_state.clear()
    _search_state.update(
        {
            "query": "",
            "knowledge_dir": "",
            "files_searched": 0,
            "matches_found": 0,
            "sources_collected": [],
        }
    )


def get_search_state() -> dict[str, Any]:
    """Get current search state."""
    return _search_state.copy()


async def pre_search_hook(context: HookContext) -> HookResult:
    """Hook called before Read/Glob/Grep operations.

    Injects knowledge base context information to help guide the search.

    Args:
        context: Hook context with tool_name, tool_args, etc.

    Returns:
        HookResult with optional context injection
    """
    from houyi.core.skill.hooks import HookResult

    tool_name = context.tool_name or ""
    tool_args = context.tool_args or {}

    # Get knowledge directory from environment or context
    knowledge_dir = os.environ.get("KNOWLEDGE_DIR", "")
    if not knowledge_dir and context.metadata:
        knowledge_dir = context.metadata.get("knowledge_dir", "")
    if not knowledge_dir:
        from houyi.config import env

        knowledge_dir = env.rag_knowledge_dir

    # Track search state
    _search_state["knowledge_dir"] = knowledge_dir

    # Build context injection
    output_parts = []

    # For Grep operations, provide search guidance
    if tool_name.lower() == "grep":
        pattern = tool_args.get("pattern", "")
        if pattern:
            _search_state["query"] = pattern
            output_parts.append(f"[kb-search] Searching knowledge base: {knowledge_dir}")
            output_parts.append(f"[kb-search] Pattern: {pattern}")

    # For Read operations, track files accessed
    elif tool_name.lower() == "read":
        file_path = tool_args.get("file_path", "")
        if file_path:
            _search_state["files_searched"] += 1
            # Check if this is within knowledge directory
            try:
                kb_path = Path(knowledge_dir).resolve()
                file_p = Path(file_path).resolve()
                if kb_path in file_p.parents or file_p == kb_path:
                    output_parts.append(f"[kb-search] Reading knowledge file: {file_path}")
            except (ValueError, OSError):
                pass

    # For Glob operations, track directory exploration
    elif tool_name.lower() == "glob":
        pattern = tool_args.get("pattern", "")
        if pattern:
            output_parts.append(f"[kb-search] Exploring: {pattern}")

    if output_parts:
        return HookResult(
            success=True,
            output="\n".join(output_parts),
            inject_to_prompt=True,
        )

    return HookResult(success=True)


async def post_search_hook(context: HookContext) -> HookResult:
    """Hook called after any tool use.

    Records search progress and result statistics.

    Args:
        context: Hook context with tool_name, tool_args, tool_result, etc.

    Returns:
        HookResult with search progress
    """
    from houyi.core.skill.hooks import HookResult

    tool_name = context.tool_name or ""
    tool_result = context.tool_result

    # Track matches from Grep results
    if tool_name.lower() == "grep" and tool_result:
        if isinstance(tool_result, dict):
            matches = tool_result.get("matches", [])
            if matches:
                _search_state["matches_found"] += len(matches)
                # Collect sources
                for match in matches[:5]:  # Limit to first 5
                    if isinstance(match, dict) and "file" in match:
                        source = {
                            "file_path": match["file"],
                            "location": f"line {match.get('line', '?')}",
                            "snippet": match.get("content", "")[:200],
                        }
                        _search_state["sources_collected"].append(source)

    # Track content from Read results
    elif tool_name.lower() == "read" and tool_result:
        if isinstance(tool_result, str) and len(tool_result) > 0:
            _search_state["files_searched"] += 1

    # Return progress summary
    files = _search_state["files_searched"]
    matches = _search_state["matches_found"]

    if files > 0 or matches > 0:
        return HookResult(
            success=True,
            output=f"[kb-search] Progress: {files} files searched, {matches} matches found",
            metadata={
                "files_searched": files,
                "matches_found": matches,
            },
        )

    return HookResult(success=True)


async def stop_hook(context: HookContext) -> HookResult:
    """Hook called before stopping.

    Verifies that a valid answer was found and provides summary.

    Args:
        context: Hook context

    Returns:
        HookResult with search summary
    """
    from houyi.core.skill.hooks import HookResult

    files = _search_state["files_searched"]
    matches = _search_state["matches_found"]
    sources = _search_state["sources_collected"]

    output_parts = []

    if files == 0 and matches == 0:
        output_parts.append(
            "[kb-search] WARNING: No search results found. "
            "Consider broadening search terms or checking knowledge directory."
        )
    else:
        output_parts.append(
            f"[kb-search] Search complete: {files} files searched, {matches} matches found"
        )

        if sources:
            output_parts.append("[kb-search] Sources collected:")
            for i, source in enumerate(sources[:3], 1):
                output_parts.append(f"  {i}. {source['file_path']}")
            if len(sources) > 3:
                output_parts.append(f"  ... and {len(sources) - 3} more")

    # Reset state for next search
    reset_search_state()

    return HookResult(
        success=True,
        output="\n".join(output_parts),
        metadata={
            "total_files": files,
            "total_matches": matches,
            "sources_count": len(sources),
        },
    )
