"""State management with immutable snapshots."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status of a task execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionState(BaseModel):
    """Session-level state snapshot.

    Immutable snapshot of session context at a point in time.
    Forms a linked list with parent_state_id for time-travel debugging.
    """

    session_id: str = Field(..., description="Session identifier")
    agent_id: str = Field(..., description="Agent identifier")
    current_plan_id: str | None = Field(None, description="Current execution plan ID")
    memory_stack: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Short-term memory stack",
    )
    execution_pointer: str | None = Field(None, description="Current executing node ID")
    parent_state_id: str | None = Field(None, description="Parent state version ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VerificationResult(BaseModel):
    """Result of an assertion verification."""

    assertion_name: str
    passed: bool
    message: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class TaskState(BaseModel):
    """Task-level state snapshot."""

    task_id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    input_data: dict[str, Any] = Field(default_factory=dict, description="Task input")
    output_data: dict[str, Any] | None = Field(None, description="Task output")
    verification_results: list[VerificationResult] = Field(
        default_factory=list,
        description="Verification results",
    )
    retry_count: int = Field(default=0, description="Number of retries")
    error: str | None = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
