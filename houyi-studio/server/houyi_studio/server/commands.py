"""Client commands sent to server."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from houyi.protocol.ir.checkpoint_ir import ReplayMode


class CommandType(str, Enum):
    """Types of client commands."""

    START_EXECUTION = "start_execution"
    PAUSE = "pause"
    RESUME = "resume"
    ABORT = "abort"
    RETRY_NODE = "retry_node"
    PATCH_PLAN = "patch_plan"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    SET_LOG_LEVEL = "set_log_level"


class ClientCommand(BaseModel):
    """Base class for client commands."""

    command_type: CommandType = Field(..., description="Type of command")
    command_id: str = Field(..., description="Unique command identifier")
    session_id: str = Field(..., description="Session identifier")


class StartExecutionCommand(ClientCommand):
    """Command to start executing a plan."""

    command_type: CommandType = Field(default=CommandType.START_EXECUTION)
    plan_id: str = Field(..., description="Plan to execute")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Initial execution inputs",
    )


class PauseCommand(ClientCommand):
    """Command to pause execution."""

    command_type: CommandType = Field(default=CommandType.PAUSE)
    execution_id: str = Field(..., description="Execution to pause")


class ResumeCommand(ClientCommand):
    """Command to resume execution."""

    command_type: CommandType = Field(default=CommandType.RESUME)
    execution_id: str = Field(..., description="Execution to resume")


class AbortCommand(ClientCommand):
    """Command to abort execution."""

    command_type: CommandType = Field(default=CommandType.ABORT)
    execution_id: str = Field(..., description="Execution to abort")


class RetryNodeCommand(ClientCommand):
    """Command to retry a failed node with modified inputs."""

    command_type: CommandType = Field(default=CommandType.RETRY_NODE)
    execution_id: str = Field(..., description="Execution identifier")
    node_id: str = Field(..., description="Node to retry")
    new_inputs: dict[str, Any] = Field(
        ...,
        description="Modified inputs for retry",
    )


class PlanPatch(BaseModel):
    """A single patch operation on a plan."""

    action: str = Field(
        ...,
        description="Patch action (add_node, delete_node, update_node, etc.)",
    )
    node_id: str | None = Field(default=None, description="Node ID for update/delete operations")
    node: dict[str, Any] | None = Field(
        default=None, description="Node data for add/update operations"
    )
    edge_id: str | None = Field(default=None, description="Edge ID for edge operations")
    edge: dict[str, Any] | None = Field(default=None, description="Edge data for add operations")


class PatchPlanCommand(ClientCommand):
    """Command to modify a plan at runtime."""

    command_type: CommandType = Field(default=CommandType.PATCH_PLAN)
    execution_id: str = Field(..., description="Execution identifier")
    base_version: int = Field(
        ...,
        description="Base version for optimistic locking",
    )
    patches: list[PlanPatch] = Field(
        ...,
        description="List of patch operations",
    )


class RestoreCheckpointCommand(ClientCommand):
    """Command to restore from a checkpoint."""

    command_type: CommandType = Field(default=CommandType.RESTORE_CHECKPOINT)
    execution_id: str | None = Field(
        default=None,
        description="Execution that owns the checkpoint (required to disambiguate cp_N across executions)",
    )
    checkpoint_id: str = Field(..., description="Checkpoint to restore")
    replay_mode: ReplayMode = Field(
        default=ReplayMode.DETERMINISTIC,
        description="How to replay execution",
    )


class SetLogLevelCommand(ClientCommand):
    """Command to change server log level at runtime."""

    command_type: CommandType = Field(default=CommandType.SET_LOG_LEVEL)
    level: str = Field(..., description="Requested log level")
