"""Execution order helpers.

This module is intentionally free of any server-specific dependencies.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

from houyi.protocol.ir import PlanIR

logger = logging.getLogger(__name__)


def get_execution_order(plan: PlanIR) -> list[str]:
    """Return node execution order for the given plan.

    This is a stable topological-ish ordering:
    - Prefer the entry node when multiple nodes have zero in-degree.
    - If the graph has cycles or disconnected nodes, append remaining nodes deterministically.
    """

    node_ids = [node.node_id for node in plan.nodes if node.deleted_at is None]
    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in plan.edges:
        if edge.source_node_id in in_degree and edge.target_node_id in in_degree:
            adjacency[edge.source_node_id].append(edge.target_node_id)
            in_degree[edge.target_node_id] += 1

    zero_in_degree = [node_id for node_id in node_ids if in_degree[node_id] == 0]
    if plan.entry_node_id in zero_in_degree:
        zero_in_degree.remove(plan.entry_node_id)
        zero_in_degree.insert(0, plan.entry_node_id)

    queue = deque(zero_in_degree)

    ordered: list[str] = []
    seen: set[str] = set()
    while queue:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
        for neighbor in adjacency.get(node_id, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(node_ids):
        remaining = [node_id for node_id in node_ids if node_id not in seen]
        ordered.extend(remaining)

    logger.debug(
        "Execution order computed: entry=%s nodes=%s edges=%s order=%s",
        plan.entry_node_id,
        len(node_ids),
        len(plan.edges),
        ordered,
    )

    return ordered


class ExecutionOrderService:
    def get_execution_order(self, plan: PlanIR) -> list[str]:
        return get_execution_order(plan)


__all__ = [
    "ExecutionOrderService",
    "get_execution_order",
]
