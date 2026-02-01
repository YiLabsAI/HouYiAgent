"""Execution IR: Runtime state representation for executing plans."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .tooling_ir import LLMToolCallOutputIR, ToolNodeOutputIR

NodeOutputsIR = LLMToolCallOutputIR | ToolNodeOutputIR | dict[str, Any]


class NodeStatus(str, Enum):
    """Execution status of a node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Soft-deleted nodes (DECISION-004)


class NodeExecutionIR(BaseModel):
    """Runtime execution state of a single node."""

    node_id: str = Field(..., description="Node identifier")
    status: NodeStatus = Field(
        default=NodeStatus.PENDING,
        description="Current execution status",
    )

    # Execution timing
    started_at: datetime | None = Field(
        default=None,
        description="Execution start time",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Execution completion time",
    )

    # Execution results
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved input values",
    )
    outputs: NodeOutputsIR = Field(
        default_factory=dict,
        description="Execution output values",
    )
    error: str | None = Field(
        default=None,
        description="Error message if failed",
    )

    # Streaming output (for LLM nodes)
    streaming_output: str = Field(
        default="",
        description="Accumulated streaming output",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (duration, retries, etc.)",
    )


class ExecutionStatus(str, Enum):
    """Overall execution status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class ExecutionIR(BaseModel):
    """Complete execution state representation.

    This tracks the runtime state of a plan execution.
    Frontend displays this state, backend updates it.
    """

    execution_id: str = Field(..., description="Unique execution identifier")
    plan_id: str = Field(..., description="Associated plan ID")

    # Overall status
    status: ExecutionStatus = Field(
        default=ExecutionStatus.IDLE,
        description="Overall execution status",
    )

    # Node execution states
    node_executions: dict[str, NodeExecutionIR] = Field(
        default_factory=dict,
        description="Map of node_id -> execution state",
    )

    # Execution context (shared variables)
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution context with variable values",
    )

    # Timing
    started_at: datetime | None = Field(
        default=None,
        description="Execution start time",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Execution completion time",
    )

    # Error tracking
    error: str | None = Field(
        default=None,
        description="Overall error message if failed",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata",
    )

    def get_node_execution(self, node_id: str) -> NodeExecutionIR | None:
        """Get execution state for a node."""
        return self.node_executions.get(node_id)

    def update_node_status(
        self,
        node_id: str,
        status: NodeStatus,
        **kwargs: Any,
    ) -> None:
        """Update node execution status.

        Args:
            node_id: Node identifier
            status: New status
            **kwargs: Additional fields to update (inputs, outputs, error, etc.)
        """
        if node_id not in self.node_executions:
            self.node_executions[node_id] = NodeExecutionIR(node_id=node_id)

        node_exec = self.node_executions[node_id]
        node_exec.status = status

        # Update timing
        if status == NodeStatus.RUNNING and node_exec.started_at is None:
            node_exec.started_at = datetime.now(timezone.utc)
        elif status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED):
            if node_exec.completed_at is None:
                node_exec.completed_at = datetime.now(timezone.utc)

        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(node_exec, key):
                setattr(node_exec, key, value)

    def get_completed_nodes(self) -> set[str]:
        """Get set of completed node IDs."""
        return {
            node_id
            for node_id, node_exec in self.node_executions.items()
            if node_exec.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
        }
