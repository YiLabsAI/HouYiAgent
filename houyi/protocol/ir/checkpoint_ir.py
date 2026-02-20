"""Checkpoint IR: State snapshot representation for time-travel debugging."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CheckpointTrigger(str, Enum):
    """Trigger that caused checkpoint creation."""

    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    USER_PAUSE = "user_pause"
    USER_CHECKPOINT = "user_checkpoint"
    PERIODIC = "periodic"


class ReplayMode(str, Enum):
    """Mode for replaying from a checkpoint."""

    DETERMINISTIC = "deterministic"  # Use recorded LLM outputs
    FRESH = "fresh"  # Re-call LLM with same inputs


class LLMCallLog(BaseModel):
    """Log of a single LLM call for deterministic replay (DECISION-001).

    Stores the complete LLM interaction to enable deterministic replay
    without relying on seed parameters.
    """

    call_id: str = Field(..., description="Unique call identifier")
    node_id: str = Field(..., description="Node that made this call")

    # Call details
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Call timestamp",
    )
    model: str = Field(..., description="Model identifier")

    # Input/Output
    prompt: str | list[dict[str, Any]] = Field(
        ...,
        description="Input prompt (string or messages)",
    )
    response: str = Field(..., description="Complete LLM response")

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tokens, latency, etc.)",
    )


class CheckpointIR(BaseModel):
    """Complete state snapshot for time-travel debugging.

    Captures the complete execution state at a point in time.
    Enables:
    - Time-travel: Restore to this point
    - Deterministic replay: Use recorded LLM outputs
    - Fresh replay: Re-run from this point with new LLM calls
    """

    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    execution_id: str = Field(..., description="Associated execution ID")
    plan_id: str = Field(..., description="Associated plan ID")

    # Checkpoint metadata
    sequence_number: int = Field(
        ...,
        description="Sequential checkpoint number (for ordering)",
    )
    trigger: CheckpointTrigger = Field(
        ...,
        description="What triggered this checkpoint",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Checkpoint creation time",
    )

    # Snapshot of execution state
    execution_snapshot: dict[str, Any] = Field(
        ...,
        description="Serialized ExecutionIR state",
    )

    # LLM call logs (for deterministic replay)
    llm_call_logs: list[LLMCallLog] = Field(
        default_factory=list,
        description="LLM calls made up to this checkpoint",
    )

    # Incremental storage (DECISION-001)
    parent_checkpoint_id: str | None = Field(
        default=None,
        description="Parent checkpoint ID (for delta storage)",
    )
    delta: dict[str, Any] | None = Field(
        default=None,
        description="Delta from parent checkpoint (if incremental)",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional checkpoint metadata",
    )

    def is_incremental(self) -> bool:
        """Check if this is an incremental checkpoint."""
        return self.parent_checkpoint_id is not None and self.delta is not None

    def get_llm_calls_for_node(self, node_id: str) -> list[LLMCallLog]:
        """Get all LLM calls made by a specific node."""
        return [log for log in self.llm_call_logs if log.node_id == node_id]
