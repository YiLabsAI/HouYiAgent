"""Command parsing for WebSocket client payloads."""

from __future__ import annotations

import logging
from typing import Any

from .commands import (
    AbortCommand,
    ClientCommand,
    CommandType,
    ConfigureSkillCommand,
    ConsentResponseCommand,
    DryRunSkillCommand,
    GetSkillDetailCommand,
    GetSkillMetricsCommand,
    ListSkillsCommand,
    LoadSkillCommand,
    PatchPlanCommand,
    PauseCommand,
    RemoveSkillFromDiskCommand,
    RestoreCheckpointCommand,
    ResumeCommand,
    RetryNodeCommand,
    SetLogLevelCommand,
    StartExecutionCommand,
    UnloadSkillCommand,
)

TypedCommand = ClientCommand
ParsedCommand = TypedCommand | dict[str, Any]


class CommandParser:
    """Parse incoming command payloads into typed commands or passthrough dicts."""

    def __init__(
        self,
        *,
        typed_command_map: dict[str, type[ClientCommand]] | None = None,
        passthrough_command_types: set[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._typed_command_map = typed_command_map or build_default_typed_command_map()
        self._passthrough_command_types = (
            passthrough_command_types or build_default_passthrough_command_types()
        )
        self._logger = logger or logging.getLogger(__name__)

    def parse(self, data: dict[str, Any]) -> ParsedCommand | None:
        """Return a parsed command object or None when unsupported/invalid."""
        try:
            command_type = data.get("command_type")
            if not isinstance(command_type, str):
                self._logger.warning("Unknown command type: %s", command_type)
                return None

            self._logger.debug("Parsing command type: %s", command_type)

            command_cls = self._typed_command_map.get(command_type)
            if command_cls is not None:
                return command_cls(**data)

            if command_type in self._passthrough_command_types:
                return data

            self._logger.warning("Unknown command type: %s", command_type)
            return None
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._logger.error("Error parsing command: %s", exc)
            return None


def build_default_typed_command_map() -> dict[str, type[ClientCommand]]:
    """Map command_type to Pydantic command classes."""
    return {
        CommandType.START_EXECUTION.value: StartExecutionCommand,
        CommandType.PAUSE.value: PauseCommand,
        CommandType.RESUME.value: ResumeCommand,
        CommandType.ABORT.value: AbortCommand,
        CommandType.RETRY_NODE.value: RetryNodeCommand,
        CommandType.PATCH_PLAN.value: PatchPlanCommand,
        CommandType.RESTORE_CHECKPOINT.value: RestoreCheckpointCommand,
        CommandType.SET_LOG_LEVEL.value: SetLogLevelCommand,
        CommandType.LIST_SKILLS.value: ListSkillsCommand,
        CommandType.GET_SKILL_DETAIL.value: GetSkillDetailCommand,
        CommandType.GET_SKILL_METRICS.value: GetSkillMetricsCommand,
        CommandType.LOAD_SKILL.value: LoadSkillCommand,
        CommandType.UNLOAD_SKILL.value: UnloadSkillCommand,
        CommandType.REMOVE_SKILL_FROM_DISK.value: RemoveSkillFromDiskCommand,
        CommandType.DRY_RUN_SKILL.value: DryRunSkillCommand,
        CommandType.CONFIGURE_SKILL.value: ConfigureSkillCommand,
        CommandType.CONSENT_RESPONSE.value: ConsentResponseCommand,
    }


def build_default_passthrough_command_types() -> set[str]:
    """Dict passthrough commands handled by dedicated branches/services."""
    return {
        "save_workflow",
        "load_workflow",
        "list_workflows",
        "list_knowledge_libraries",
        "create_knowledge_library",
        "delete_knowledge_library",
        "search_knowledge",
        "update_knowledge_library",
        "ingest_knowledge_files",
        "rebuild_knowledge_index",
        "cancel_ingest",
        "list_documents",
        "get_document",
        "delete_document",
        "disable_document",
        "enable_document",
        "list_chunks",
        "preview_chunks",
        "frontend_log",
    }
