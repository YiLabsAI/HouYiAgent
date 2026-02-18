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

    # SimpleSkill Console integration commands
    LIST_SKILLS = "list_skills"
    GET_SKILL_DETAIL = "get_skill_detail"
    GET_SKILL_METRICS = "get_skill_metrics"
    LOAD_SKILL = "load_skill"
    UNLOAD_SKILL = "unload_skill"
    DRY_RUN_SKILL = "dry_run_skill"
    CONFIGURE_SKILL = "configure_skill"
    CONSENT_RESPONSE = "consent_response"


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


# =============================================================================
# SimpleSkill Console Integration Commands
# =============================================================================


class ListSkillsCommand(ClientCommand):
    """Command to list all registered skills."""

    command_type: CommandType = Field(default=CommandType.LIST_SKILLS)


class GetSkillDetailCommand(ClientCommand):
    """Command to get full detail of a specific skill."""

    command_type: CommandType = Field(default=CommandType.GET_SKILL_DETAIL)
    skill_name: str = Field(..., description="Skill name to get detail for")


class GetSkillMetricsCommand(ClientCommand):
    """Command to get metrics for a specific skill."""

    command_type: CommandType = Field(default=CommandType.GET_SKILL_METRICS)
    skill_name: str = Field(..., description="Skill name to get metrics for")


class LoadSkillCommand(ClientCommand):
    """Command to load a skill from a file path, URL, or directory.

    Accepts any of:
    - Local file path to SKILL.md or simpleskill.json
    - URL (http/https) pointing to a SKILL.md file
    - Directory path containing SKILL.md files
    """

    command_type: CommandType = Field(default=CommandType.LOAD_SKILL)
    path: str = Field(
        default="",
        description="(Deprecated) File path — use 'source' instead",
    )
    source: str = Field(
        default="",
        description="File path, URL, or directory path to load skill(s) from",
    )

    @property
    def resolved_source(self) -> str:
        """Return the effective source, preferring 'source' over legacy 'path'."""
        return self.source or self.path


class UnloadSkillCommand(ClientCommand):
    """Command to unload a skill."""

    command_type: CommandType = Field(default=CommandType.UNLOAD_SKILL)
    skill_name: str = Field(..., description="Skill name to unload")


class DryRunSkillCommand(ClientCommand):
    """Command to perform a dry-run validation of a skill invocation."""

    command_type: CommandType = Field(default=CommandType.DRY_RUN_SKILL)
    skill_name: str = Field(..., description="Skill name")
    tool_name: str = Field(..., description="Tool name within the skill")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Input to validate",
    )
    live: bool = Field(
        default=False,
        description="If True, also verify with a real LLM call to check "
        "the skill produces a valid tool invocation",
    )


class ConfigureSkillCommand(ClientCommand):
    """Command to update runtime configuration for a skill.

    Supports changing:
    - policy_action: 'allow' | 'allow_with_consent' | 'deny'
    - auto_invoke: whether LLM can auto-trigger (True/False)
    """

    command_type: CommandType = Field(default=CommandType.CONFIGURE_SKILL)
    skill_name: str = Field(..., description="Skill to configure")
    policy_action: str | None = Field(
        default=None,
        description="New policy action: 'allow', 'allow_with_consent', or 'deny'",
    )
    auto_invoke: bool | None = Field(
        default=None,
        description="Whether LLM can auto-invoke this skill",
    )


class ConsentResponseCommand(ClientCommand):
    """Command to respond to a consent request."""

    command_type: CommandType = Field(default=CommandType.CONSENT_RESPONSE)
    request_id: str = Field(..., description="Consent request ID to respond to")
    granted: bool = Field(..., description="Whether consent is granted")
    remember: bool = Field(
        default=False,
        description="Whether to remember this decision",
    )
