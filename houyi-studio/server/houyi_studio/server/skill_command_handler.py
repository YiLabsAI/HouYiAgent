"""Skill-lifecycle command handler (Load, Unload, Configure, DryRun, Metrics, Consent).

Part of the command-handler hierarchy extracted from ``app.py`` following the
**Single Responsibility Principle (SRP)** and **Open/Closed Principle (OCP)**:

Architecture
~~~~~~~~~~~~
::

    WebSocket message
         │
         ▼
    CommandParser          → parse raw JSON into typed/dict command
         │
         ▼
    CommandDispatcher      → route command_type to the correct handler  (OCP)
         │
         ├─► SkillCommandHandler  ◄── this module  – skill lifecycle
         │                                (load/unload/configure/dry-run/consent)
         ├─► CommandHandler               – resource CRUD (workflow/knowledge/document)
         └─► ExecutionCommandHandler      – execution lifecycle
                                              (start/pause/abort/patch/restore)

Design rationale
~~~~~~~~~~~~~~~~
*   **SRP**: This class owns *skill-lifecycle* commands — operations that
    discover, load/unload, configure, dry-run, and collect metrics for
    agent skills.  Skills are first-class extensibility points in Houyi,
    so their lifecycle is complex enough to warrant a dedicated handler.

*   **OCP**: New skill-related commands (e.g., a future "upgrade-skill" or
    "export-skill-config") can be added here without touching the dispatcher
    or the other two handlers.

*   **Dependency Inversion**: Collaborators (``send_event``, ``SkillService``)
    are constructor-injected, enabling isolated unit testing with fakes.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from .commands import (
    ConfigureSkillCommand,
    ConsentResponseCommand,
    DryRunSkillCommand,
    GetSkillDetailCommand,
    GetSkillMetricsCommand,
    ListSkillsCommand,
    LoadSkillCommand,
    UnloadSkillCommand,
)
from .events import (
    ConsentResultEvent,
    DryRunResult,
    DryRunResultEvent,
    SkillConfiguredEvent,
    SkillDetail,
    SkillDetailEvent,
    SkillErrorEvent,
    SkillListEvent,
    SkillLoadedEvent,
    SkillMetricsData,
    SkillMetricsEvent,
    SkillPermission,
    SkillSummary,
    SkillUnloadedEvent,
)
from .skill_service import get_skill_service

SkillCommand = (
    ListSkillsCommand
    | GetSkillDetailCommand
    | GetSkillMetricsCommand
    | LoadSkillCommand
    | UnloadSkillCommand
    | ConfigureSkillCommand
    | DryRunSkillCommand
    | ConsentResponseCommand
)


class SkillCommandHandler:
    """Command handler for agent-skill lifecycle management.

    Responsibilities
    ----------------
    - Discovery: list available skills with summary metadata (``list_skills``).
    - Inspection: retrieve full skill detail including parameters, permissions,
      and side-effect declarations (``get_skill_detail``).
    - Loading / unloading: activate or deactivate a skill in the current session
      (``load_skill`` / ``unload_skill``).
    - Configuration: update runtime parameters for a loaded skill
      (``configure_skill``).
    - Dry-run: execute a skill in sandbox mode and return the result preview
      without committing side effects (``dry_run_skill``).
    - Consent flow: relay user consent responses for skills that require explicit
      approval before performing privileged operations (``consent_response``).
    - Metrics: gather per-skill usage statistics (``get_skill_metrics``).

    Integration
    -----------
    Registered with ``CommandDispatcher`` under the ``command_type`` values of
    each typed ``SkillCommand``.  The dispatcher invokes ``handle()`` with
    the already-parsed Pydantic command and the session id.
    """

    def __init__(
        self,
        *,
        send_event: Callable[[str, object], Awaitable[None]],
        skill_service_getter: Callable[[], object] = get_skill_service,
        logger: logging.Logger | None = None,
    ) -> None:
        self._send_event = send_event
        self._skill_service_getter = skill_service_getter
        self._logger = logger or logging.getLogger(__name__)

    @staticmethod
    def can_handle(command: object) -> bool:
        return isinstance(
            command,
            (
                ListSkillsCommand,
                GetSkillDetailCommand,
                GetSkillMetricsCommand,
                LoadSkillCommand,
                UnloadSkillCommand,
                ConfigureSkillCommand,
                DryRunSkillCommand,
                ConsentResponseCommand,
            ),
        )

    async def handle(self, command: SkillCommand, session_id: str) -> None:
        if isinstance(command, ListSkillsCommand):
            await self._handle_list_skills(session_id)
        elif isinstance(command, GetSkillDetailCommand):
            await self._handle_get_skill_detail(command, session_id)
        elif isinstance(command, GetSkillMetricsCommand):
            await self._handle_get_skill_metrics(command, session_id)
        elif isinstance(command, LoadSkillCommand):
            await self._handle_load_skill(command, session_id)
        elif isinstance(command, UnloadSkillCommand):
            await self._handle_unload_skill(command, session_id)
        elif isinstance(command, ConfigureSkillCommand):
            await self._handle_configure_skill(command, session_id)
        elif isinstance(command, DryRunSkillCommand):
            await self._handle_dry_run_skill(command, session_id)
        elif isinstance(command, ConsentResponseCommand):
            await self._handle_consent_response(command, session_id)

    async def _handle_list_skills(self, session_id: str) -> None:
        skill_service = self._skill_service_getter()
        skills_data = skill_service.list_skills()
        skill_summaries = [
            SkillSummary(
                name=s["name"],
                display_name=s.get("display_name", s["name"]),
                description=s.get("description"),
                tools=s.get("tools", []),
                policy_action=s.get("policy_action", "allow"),
                side_effect=s.get("side_effect", "none"),
                certification=s.get("certification", "unverified"),
            )
            for s in skills_data
        ]
        event = SkillListEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            skills=skill_summaries,
        )
        await self._send_event(session_id, event)
        self._logger.info("Sent skill list with %d skills", len(skill_summaries))

    async def _handle_get_skill_detail(
        self, command: GetSkillDetailCommand, session_id: str
    ) -> None:
        skill_service = self._skill_service_getter()
        detail_data = skill_service.get_skill_detail(command.skill_name)
        if detail_data:
            skill_detail = SkillDetail(
                name=detail_data["name"],
                display_name=detail_data.get("display_name", detail_data["name"]),
                description=detail_data.get("description"),
                version=detail_data.get("version") or "0.0.0",
                author=detail_data.get("author"),
                tools=detail_data.get("tools", []),
                permissions=[
                    SkillPermission(
                        name=p["name"],
                        description=p.get("description"),
                        is_sensitive=p.get("is_sensitive", False),
                    )
                    for p in detail_data.get("permissions", [])
                ],
                policy=detail_data.get("policy", {}),
                hooks=detail_data.get("hooks", []),
                certification=detail_data.get("certification", "unverified"),
                side_effect=detail_data.get("side_effect", "none"),
            )
            event = SkillDetailEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                skill=skill_detail,
            )
            await self._send_event(session_id, event)
            return

        error_event = SkillErrorEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            skill_name=command.skill_name,
            error_code="skill_not_found",
            message=f"Skill '{command.skill_name}' not found",
            suggestions=["Check skill name", "List available skills"],
        )
        await self._send_event(session_id, error_event)

    async def _handle_get_skill_metrics(
        self, command: GetSkillMetricsCommand, session_id: str
    ) -> None:
        skill_service = self._skill_service_getter()
        metrics_data = skill_service.get_skill_metrics(command.skill_name)
        if not metrics_data:
            return
        metrics = SkillMetricsData(
            skill_name=metrics_data["skill_name"],
            total_calls=metrics_data.get("total_calls", 0),
            success_count=metrics_data.get("success_count", 0),
            failure_count=metrics_data.get("failure_count", 0),
            avg_latency_ms=metrics_data.get("avg_latency_ms", 0.0),
            p50_latency_ms=metrics_data.get("p50_latency_ms", 0.0),
            p99_latency_ms=metrics_data.get("p99_latency_ms", 0.0),
            success_rate=metrics_data.get("success_rate", 0.0),
            last_invoked=metrics_data.get("last_invoked"),
        )
        event = SkillMetricsEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            metrics=metrics,
        )
        await self._send_event(session_id, event)

    async def _handle_load_skill(self, command: LoadSkillCommand, session_id: str) -> None:
        skill_service = self._skill_service_getter()
        source = command.resolved_source
        if not source:
            error_event = SkillErrorEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                skill_name=None,
                error_code="missing_source",
                message="No source specified. Provide a file path, URL, or directory.",
                suggestions=[
                    "Use a local path: /path/to/SKILL.md",
                    "Use a URL: https://raw.githubusercontent.com/.../SKILL.md",
                    "Use a directory: /path/to/skills/",
                ],
            )
            await self._send_event(session_id, error_event)
            return

        success, name_or_code, error_msg = skill_service.load_skill(source)
        if success:
            event = SkillLoadedEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                skill_name=name_or_code,
                message=f"Skill '{name_or_code}' loaded successfully from {source}",
            )
            await self._send_event(session_id, event)
            self._logger.info("Loaded skill '%s' from: %s", name_or_code, source)
            return

        error_event = SkillErrorEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            skill_name=None,
            error_code=name_or_code,
            message=error_msg or "Failed to load skill",
            suggestions=[
                "Check file path or URL",
                "Verify SKILL.md format (YAML frontmatter required)",
                "For directories, ensure they contain SKILL.md files",
            ],
        )
        await self._send_event(session_id, error_event)

    async def _handle_unload_skill(self, command: UnloadSkillCommand, session_id: str) -> None:
        skill_service = self._skill_service_getter()
        success, error_msg = skill_service.unload_skill(command.skill_name)
        if success:
            event = SkillUnloadedEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                skill_name=command.skill_name,
            )
            await self._send_event(session_id, event)
            self._logger.info("Unloaded skill: %s", command.skill_name)
            return

        error_event = SkillErrorEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            skill_name=command.skill_name,
            error_code="unload_failed",
            message=error_msg or "Failed to unload skill",
            suggestions=["Check skill name"],
        )
        await self._send_event(session_id, error_event)

    async def _handle_configure_skill(
        self, command: ConfigureSkillCommand, session_id: str
    ) -> None:
        skill_service = self._skill_service_getter()
        success, error_msg = skill_service.configure_skill(
            command.skill_name,
            policy_action=command.policy_action,
            auto_invoke=command.auto_invoke,
        )
        if success:
            event = SkillConfiguredEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                skill_name=command.skill_name,
                policy_action=command.policy_action,
                auto_invoke=command.auto_invoke,
                message=f"Skill '{command.skill_name}' configuration updated",
            )
            await self._send_event(session_id, event)
            self._logger.info("Configured skill: %s", command.skill_name)
            return

        error_event = SkillErrorEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            skill_name=command.skill_name,
            error_code="configure_failed",
            message=error_msg or "Failed to configure skill",
            suggestions=["Check skill name and configuration values"],
        )
        await self._send_event(session_id, error_event)

    async def _handle_dry_run_skill(self, command: DryRunSkillCommand, session_id: str) -> None:
        skill_service = self._skill_service_getter()
        result_data = skill_service.dry_run(command.skill_name, command.tool_name, command.input)
        result = DryRunResult(
            valid=result_data["valid"],
            schema_errors=result_data.get("schema_errors", []),
            policy_result=result_data.get("policy_result", "allow"),
            capability_gaps=result_data.get("capability_gaps", []),
            estimated_side_effects=result_data.get("estimated_side_effects", []),
        )
        event = DryRunResultEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            skill_name=command.skill_name,
            result=result,
        )
        await self._send_event(session_id, event)
        self._logger.info(
            "Dry-run for %s.%s: valid=%s",
            command.skill_name,
            command.tool_name,
            result.valid,
        )

    async def _handle_consent_response(
        self, command: ConsentResponseCommand, session_id: str
    ) -> None:
        skill_service = self._skill_service_getter()
        found = skill_service.respond_to_consent(
            command.request_id, command.granted, command.remember
        )
        if found:
            event = ConsentResultEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                request_id=command.request_id,
                granted=command.granted,
                remembered=command.remember,
            )
            await self._send_event(session_id, event)
            self._logger.info(
                "Consent response: request_id=%s granted=%s",
                command.request_id,
                command.granted,
            )
            return

        self._logger.warning("Consent request not found: %s", command.request_id)
