"""Knowledge graph query skill hooks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.core.skill.hooks import HookContext, HookResult

logger = logging.getLogger(__name__)

# Graph query state tracking
_graph_state: dict[str, Any] = {
    "entities_found": 0,
    "relations_found": 0,
    "hops_traversed": 0,
}


def reset_graph_state() -> None:
    """Reset graph state."""
    _graph_state.clear()
    _graph_state.update(
        {
            "entities_found": 0,
            "relations_found": 0,
            "hops_traversed": 0,
        }
    )


def get_graph_state() -> dict[str, Any]:
    """Get current graph state."""
    return _graph_state.copy()


async def pre_graph_hook(context: HookContext) -> HookResult:
    """Hook called before Read/Grep during graph traversal."""
    from houyi.core.skill.hooks import HookResult

    tool_name = context.tool_name or ""
    tool_args = context.tool_args or {}

    if tool_name.lower() == "grep":
        pattern = tool_args.get("pattern", "")
        if pattern:
            return HookResult(
                success=True,
                output=f"[kb-graph] Searching for entities matching: {pattern}",
            )

    return HookResult(success=True)


async def post_graph_hook(context: HookContext) -> HookResult:
    """Hook called after tool use during graph traversal."""
    from houyi.core.skill.hooks import HookResult

    tool_result = context.tool_result

    # Track entities/relations from results
    if isinstance(tool_result, dict):
        entities = tool_result.get("entities", [])
        relations = tool_result.get("relations", [])
        if entities:
            _graph_state["entities_found"] += len(entities)
        if relations:
            _graph_state["relations_found"] += len(relations)

    entities = _graph_state["entities_found"]
    relations = _graph_state["relations_found"]

    if entities > 0 or relations > 0:
        return HookResult(
            success=True,
            output=f"[kb-graph] Found: {entities} entities, {relations} relations",
        )

    return HookResult(success=True)


async def stop_hook(context: HookContext) -> HookResult:
    """Hook called before stopping."""
    from houyi.core.skill.hooks import HookResult

    entities = _graph_state["entities_found"]
    relations = _graph_state["relations_found"]
    hops = _graph_state["hops_traversed"]

    output = (
        f"[kb-graph] Query complete: {entities} entities, "
        f"{relations} relations, {hops} hops traversed"
    )

    reset_graph_state()

    return HookResult(success=True, output=output)
