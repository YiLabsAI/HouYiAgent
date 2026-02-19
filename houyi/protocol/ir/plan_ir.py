"""Plan IR: Frontend-backend shared representation of execution plans."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator

from houyi.orchestration.plan import NodeType


class NodeIR(BaseModel):
    """Node in the execution plan DAG.

    This is the isomorphic representation shared between frontend and backend.
    Frontend renders this as a visual node, backend uses it for execution.
    """

    node_id: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Type of execution node")

    # Visual properties (for frontend rendering)
    position: dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0},
        description="Node position in the visual canvas",
    )

    # Execution properties
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Node configuration (timeout, retry, etc.)",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Input mapping (key -> value or $variable reference)",
    )
    outputs: dict[str, str] = Field(
        default_factory=dict,
        description="Output variable names (key -> variable name)",
    )

    # Soft delete support (DECISION-004)
    deleted_at: datetime | None = Field(
        default=None,
        description="Soft delete timestamp (None if not deleted)",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional node metadata",
    )


class EdgeIR(BaseModel):
    """Edge connecting two nodes in the DAG."""

    edge_id: str = Field(..., description="Unique edge identifier")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")

    # Visual properties
    label: str | None = Field(
        default=None,
        description="Edge label (for conditional routing)",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional edge metadata",
    )


class PlanPositionIR(BaseModel):
    x: float = Field(default=0.0)
    y: float = Field(default=0.0)


class PlanLayoutIR(BaseModel):
    positions: dict[str, PlanPositionIR] = Field(
        default_factory=dict,
        description="Node positions keyed by node_id",
    )


class PlanIR(BaseModel):
    """Complete execution plan representation.

    This is the single source of truth for both frontend and backend.
    Frontend renders it visually, backend executes it.
    """

    plan_id: str = Field(..., description="Unique plan identifier")
    version: int = Field(
        default=1,
        description="Version number for optimistic locking (DECISION-002)",
    )

    # DAG structure
    nodes: list[NodeIR] = Field(..., description="List of nodes")
    edges: list[EdgeIR] = Field(..., description="List of edges")
    entry_node_id: str = Field(..., description="Entry node ID")

    layout: PlanLayoutIR | None = Field(
        default=None,
        description="Optional visual layout separated from the execution spec",
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Plan creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Plan metadata (task, agent_role, etc.)",
    )

    @model_validator(mode="after")
    def _migrate_layout(self) -> PlanIR:
        if self.layout is None:
            self.layout = PlanLayoutIR()

        # Migrate legacy NodeIR.position into layout when layout is missing positions.
        for node in self.nodes:
            if node.node_id not in self.layout.positions:
                pos = node.position if isinstance(node.position, dict) else {"x": 0.0, "y": 0.0}
                x = float(pos.get("x", 0.0))
                y = float(pos.get("y", 0.0))
                self.layout.positions[node.node_id] = PlanPositionIR(x=x, y=y)

        # Mirror layout back to NodeIR.position for backward compatibility with existing clients.
        for node in self.nodes:
            layout_pos = self.layout.positions.get(node.node_id)
            if layout_pos is None:
                continue
            node.position = {"x": float(layout_pos.x), "y": float(layout_pos.y)}
        return self

    def get_node_position(self, node_id: str) -> dict[str, float]:
        if self.layout is None:
            return {"x": 0.0, "y": 0.0}
        pos = self.layout.positions.get(node_id)
        if pos is None:
            return {"x": 0.0, "y": 0.0}
        return {"x": float(pos.x), "y": float(pos.y)}

    def set_node_position(self, node_id: str, position: dict[str, Any]) -> None:
        if self.layout is None:
            self.layout = PlanLayoutIR()
        x = float(position.get("x", 0.0))
        y = float(position.get("y", 0.0))
        self.layout.positions[node_id] = PlanPositionIR(x=x, y=y)
        node = self.get_node(node_id)
        if node is not None:
            node.position = {"x": x, "y": y}

    def get_node(self, node_id: str) -> NodeIR | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_dependencies(self, node_id: str) -> list[str]:
        """Get dependency node IDs for a given node."""
        return [edge.source_node_id for edge in self.edges if edge.target_node_id == node_id]

    def validate_dag(self) -> tuple[bool, str | None]:
        """Validate DAG structure (no cycles, entry reachable).

        Returns:
            (is_valid, error_message)
        """
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            # Get children
            children = [
                edge.target_node_id for edge in self.edges if edge.source_node_id == node_id
            ]

            for child in children:
                if child not in visited:
                    if has_cycle(child):
                        return True
                elif child in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        # Check from entry node
        if has_cycle(self.entry_node_id):
            return False, "Cycle detected in DAG"

        # Check all nodes are reachable from entry
        reachable = set()

        def dfs(node_id: str) -> None:
            reachable.add(node_id)
            children = [
                edge.target_node_id for edge in self.edges if edge.source_node_id == node_id
            ]
            for child in children:
                if child not in reachable:
                    dfs(child)

        dfs(self.entry_node_id)

        all_node_ids = {node.node_id for node in self.nodes if node.deleted_at is None}
        unreachable = all_node_ids - reachable

        if unreachable:
            return False, f"Unreachable nodes: {unreachable}"

        return True, None
