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
    from houyi.domain.skill.hooks import HookContext, HookResult

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


def _resolve_knowledge_dir(metadata: dict[str, Any] | None) -> str:
    knowledge_dir = os.environ.get("KNOWLEDGE_DIR", "")
    if not knowledge_dir and metadata:
        knowledge_dir = str(metadata.get("knowledge_dir", ""))
    if knowledge_dir:
        return knowledge_dir

    from houyi.infrastructure.config import env

    return env.rag_knowledge_dir


def _handle_grep(tool_args: dict[str, Any], knowledge_dir: str) -> list[str]:
    pattern = tool_args.get("pattern", "")
    if not pattern:
        return []
    _search_state["query"] = pattern
    return [
        f"[kb-search] Searching knowledge base: {knowledge_dir}",
        f"[kb-search] Pattern: {pattern}",
    ]


def _handle_read(tool_args: dict[str, Any], knowledge_dir: str) -> list[str]:
    file_path = tool_args.get("file_path", "")
    if not file_path:
        return []

    _search_state["files_searched"] += 1
    try:
        kb_path = Path(knowledge_dir).resolve()
        file_p = Path(file_path).resolve()
        if kb_path in file_p.parents or file_p == kb_path:
            return [f"[kb-search] Reading knowledge file: {file_path}"]
    except (ValueError, OSError):
        return []
    return []


def _handle_glob(tool_args: dict[str, Any]) -> list[str]:
    pattern = tool_args.get("pattern", "")
    if not pattern:
        return []
    return [f"[kb-search] Exploring: {pattern}"]


async def pre_search_hook(context: HookContext) -> HookResult:
    """Hook called before Read/Glob/Grep operations.

    Injects knowledge base context information to help guide the search.

    Args:
        context: Hook context with tool_name, tool_args, etc.

    Returns:
        HookResult with optional context injection
    """
    from houyi.domain.skill.hooks import HookResult

    tool_name = context.tool_name or ""
    tool_args = context.tool_args or {}

    knowledge_dir = _resolve_knowledge_dir(context.metadata)
    _search_state["knowledge_dir"] = knowledge_dir

    tool_handlers: dict[str, Any] = {
        "grep": lambda: _handle_grep(tool_args, knowledge_dir),
        "read": lambda: _handle_read(tool_args, knowledge_dir),
        "glob": lambda: _handle_glob(tool_args),
    }
    output_parts = tool_handlers.get(tool_name.lower(), lambda: [])()

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
    from houyi.domain.skill.hooks import HookResult

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
    elif (
        tool_name.lower() == "read"
        and tool_result
        and isinstance(tool_result, str)
        and len(tool_result) > 0
    ):
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
    from houyi.domain.skill.hooks import HookResult

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
