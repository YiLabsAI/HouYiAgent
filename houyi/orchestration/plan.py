"""Execution plan and IR (Intermediate Representation)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from houyi.core.skill import SkillSpec


class NodeType(str, Enum):
    """Types of execution nodes in the DAG."""

    LLM = "llm"  # LLM reasoning node
    TOOL = "tool"  # Skill execution node
    VERIFY = "verify"  # Assertion verification node
    LOGIC = "logic"  # Logic control node
    ROUTE = "route"  # Routing decision node


class IRNode(BaseModel):
    """Intermediate Representation node in the execution DAG."""

    node_id: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Type of execution node")
    skill_ref: SkillSpec | None = Field(
        default=None, description="Skill reference (for TOOL nodes)"
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Input mapping (key -> value or $variable reference)",
    )
    outputs: dict[str, str] = Field(
        default_factory=dict,
        description="Output variable names (key -> variable name)",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of node IDs this node depends on",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Node-specific metadata",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_ready(self, completed_nodes: set[str]) -> bool:
        """Check if node is ready to execute.

        Args:
            completed_nodes: Set of completed node IDs

        Returns:
            True if all dependencies are satisfied
        """
        return all(dep in completed_nodes for dep in self.dependencies)

    def get_input_values(self, context: dict[str, Any]) -> dict[str, Any]:
        """Resolve input values from context.

        Args:
            context: Execution context with variable values

        Returns:
            Resolved input values
        """
        resolved = {}
        for key, value in self.inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # Variable reference
                var_name = value[1:]
                resolved[key] = context.get(var_name, value)
            else:
                # Literal value
                resolved[key] = value
        return resolved


class ExecutionPlan(BaseModel):
    """Execution plan represented as a DAG of IR nodes."""

    plan_id: str = Field(..., description="Unique plan identifier")
    nodes: list[IRNode] = Field(..., description="List of execution nodes")
    entry_node: str = Field(..., description="ID of the entry node")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Plan metadata (cost budget, priority, etc.)",
    )

    def get_node(self, node_id: str) -> IRNode | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_ready_nodes(self, completed_nodes: set[str]) -> list[IRNode]:
        """Get nodes whose dependencies are all completed."""
        ready = []
        for node in self.nodes:
            if node.node_id in completed_nodes:
                continue
            if all(dep in completed_nodes for dep in node.dependencies):
                ready.append(node)
        return ready
