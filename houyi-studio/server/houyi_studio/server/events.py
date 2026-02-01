"""Server events sent to frontend clients."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from houyi.protocol.ir.checkpoint_ir import LLMCallLog
from houyi.protocol.ir.execution_ir import ExecutionStatus, NodeStatus
from houyi.protocol.ir.plan_ir import PlanIR


class EventType(str, Enum):
    """Types of server events."""

    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    NODE_STATUS = "node_status"
    STREAMING_OUTPUT = "streaming_output"
    CHECKPOINT_CREATED = "checkpoint_created"
    EXECUTION_STATUS = "execution_status"
    RETRY_STATUS = "retry_status"
    RETRY_SUCCESS = "retry_success"
    RETRY_FAILED = "retry_failed"
    RESTORE_CHECKPOINT_RESULT = "restore_checkpoint_result"
    WORKFLOW_LIST = "workflow_list"
    CONFLICT = "conflict"
    LOG_LEVEL = "log_level"


class ServerEvent(BaseModel):
    """Base class for server events."""

    event_type: EventType = Field(..., description="Type of event")
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp",
    )
    session_id: str = Field(..., description="Session identifier")


class PlanCreatedEvent(ServerEvent):
    """Event when a new plan is created."""

    event_type: EventType = Field(default=EventType.PLAN_CREATED)
    plan: PlanIR = Field(..., description="Created plan")


class PlanUpdatedEvent(ServerEvent):
    """Event when a plan is updated (runtime editing)."""

    event_type: EventType = Field(default=EventType.PLAN_UPDATED)
    plan: PlanIR = Field(..., description="Updated plan")
    changes: list[str] = Field(
        default_factory=list,
        description="Description of changes made",
    )


class NodeStatusEvent(ServerEvent):
    """Event when a node's execution status changes."""

    event_type: EventType = Field(default=EventType.NODE_STATUS)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    status: NodeStatus = Field(..., description="New status")

    # Optional fields depending on status
    inputs: dict[str, Any] | None = Field(
        default=None,
        description="Node inputs (when starting)",
    )
    outputs: dict[str, Any] | None = Field(
        default=None,
        description="Node outputs (when completed)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (when failed)",
    )
    execution_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Execution-level metadata snapshot",
    )

    # NOTE(core): Optional observability payload (OTel-aligned). Kept generic for incremental rollout.
    # Expected shape (minimum): trace_id/span_id/parent_span_id/span_type/start_time/end_time/status
    # Extended fields (phase 1): tokens_*, cost_*, cache_hit; reasoning artifacts are sent but collapsed by default in UI.
    observation: dict[str, Any] | None = Field(
        default=None,
        description="Optional observability payload (trace/span + AI cost/reasoning artifacts)",
    )


class StreamingOutputEvent(ServerEvent):
    """Event for streaming LLM output."""

    event_type: EventType = Field(default=EventType.STREAMING_OUTPUT)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    chunk: str = Field(..., description="Output chunk")
    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk",
    )


class CheckpointCreatedEvent(ServerEvent):
    """Event when a checkpoint is created."""

    event_type: EventType = Field(default=EventType.CHECKPOINT_CREATED)
    checkpoint_id: str = Field(..., description="Checkpoint identifier")
    execution_id: str = Field(..., description="Execution identifier")
    sequence_number: int = Field(..., description="Checkpoint sequence number")
    trigger: str = Field(..., description="What triggered the checkpoint")
    llm_call_logs: list[LLMCallLog] = Field(
        default_factory=list,
        description="LLM call logs captured at this checkpoint",
    )


class ExecutionStatusEvent(ServerEvent):
    """Event when overall execution status changes."""

    event_type: EventType = Field(default=EventType.EXECUTION_STATUS)
    execution_id: str = Field(..., description="Execution identifier")
    status: ExecutionStatus = Field(..., description="New execution status")
    message: str | None = Field(
        default=None,
        description="Optional status message",
    )


class RestoreCheckpointResultEvent(ServerEvent):
    """Event emitted after a restore_checkpoint command is processed."""

    event_type: EventType = Field(default=EventType.RESTORE_CHECKPOINT_RESULT)
    checkpoint_id: str = Field(..., description="Checkpoint identifier")
    execution_id: str | None = Field(
        default=None,
        description="Execution identifier (if known / restored)",
    )
    replay_mode: str | None = Field(
        default=None,
        description="Replay mode requested (deterministic/fresh)",
    )
    success: bool = Field(..., description="Whether restore succeeded")
    message: str | None = Field(default=None, description="Optional details")


class RetryStatusEvent(ServerEvent):
    """Event when a retry attempt is scheduled or executed."""

    event_type: EventType = Field(default=EventType.RETRY_STATUS)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    attempt: int = Field(..., description="Current retry attempt (1-indexed)")
    max_retries: int = Field(..., description="Maximum retries allowed")
    error: str | None = Field(
        default=None,
        description="Error that triggered the retry",
    )


class RetrySuccessEvent(ServerEvent):
    """Event when a retry eventually succeeds."""

    event_type: EventType = Field(default=EventType.RETRY_SUCCESS)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    attempt: int = Field(..., description="Attempt that succeeded (1-indexed)")


class RetryFailedEvent(ServerEvent):
    """Event when retries are exhausted without success."""

    event_type: EventType = Field(default=EventType.RETRY_FAILED)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node identifier")
    max_retries: int = Field(..., description="Maximum retries attempted")
    error: str | None = Field(
        default=None,
        description="Final error after retries",
    )


class WorkflowListEvent(ServerEvent):
    """Event when workflow list is requested."""

    event_type: EventType = Field(default=EventType.WORKFLOW_LIST)
    workflows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of saved workflows",
    )


class ConflictEvent(ServerEvent):
    """Event when a plan modification conflicts (DECISION-002)."""

    event_type: EventType = Field(default=EventType.CONFLICT)
    your_version: int = Field(..., description="Client's base version")
    server_version: int = Field(..., description="Current server version")
    server_plan: PlanIR = Field(..., description="Current server plan")
    your_changes: list[dict[str, Any]] = Field(
        ...,
        description="Client's attempted changes",
    )


class LogLevelEvent(ServerEvent):
    """Event when server log level is updated or synced."""

    event_type: EventType = Field(default=EventType.LOG_LEVEL)
    level: str = Field(..., description="Effective log level")
    requested_level: str | None = Field(
        default=None,
        description="Requested log level (if provided by client)",
    )
