"""Execution-lifecycle command handler (Start, Pause, Resume, Abort, Retry, Patch, Restore, LogLevel).

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
         ├─► SkillCommandHandler          – skill lifecycle (load/unload/configure/dry-run)
         ├─► CommandHandler               – resource CRUD (workflow/knowledge/document)
         └─► ExecutionCommandHandler  ◄── this module  – execution lifecycle
                                              (start/pause/abort/patch/restore)

Design rationale
~~~~~~~~~~~~~~~~
*   **SRP**: This class owns *execution-lifecycle* commands — operations that
    directly mutate or query the state machine of a running DAG execution
    (start, pause, resume, abort, retry a specific node, patch the live plan,
    restore from a checkpoint).  ``SetLogLevel`` is co-located here because it
    is a typed ``ClientCommand`` that affects the server runtime context rather
    than a persistent resource.

*   **OCP**: New execution-related commands (e.g., a future "skip-node" or
    "force-complete") can be added here without modifying the dispatcher or
    the resource / skill handlers.

*   **Dependency Inversion**: The class receives its collaborators — event
    sender, execution engine, plan-patch logic — as constructor-injected
    callables, keeping it fully testable in isolation (no server needed).

*   **Strategy pattern**: ``handle()`` dispatches to private ``_handle_*``
    methods keyed on command type, mirroring the same pattern used by
    ``SkillCommandHandler`` and ``CommandHandler`` so the codebase stays
    consistent and predictable.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from uuid import uuid4

from houyi.protocol.ir import ExecutionStatus

from ..gateway.commands import (
    AbortCommand,
    ClientCommand,
    PatchPlanCommand,
    PauseCommand,
    PlanPatch,
    RestoreCheckpointCommand,
    ResumeCommand,
    RetryNodeCommand,
    SetLogLevelCommand,
    StartExecutionCommand,
)
from ..gateway.events import ExecutionStatusEvent, LogLevelEvent
from ..logging_config import get_log_level, set_log_level

_HANDLED_TYPES = (
    StartExecutionCommand,
    PauseCommand,
    ResumeCommand,
    AbortCommand,
    RetryNodeCommand,
    PatchPlanCommand,
    RestoreCheckpointCommand,
    SetLogLevelCommand,
)


class ExecutionCommandHandler:
    """Command handler for DAG execution state-machine transitions.

    Responsibilities
    ----------------
    - Execution control: start a new execution from a ``PlanIR``, pause / resume /
      abort an in-flight execution, and retry a single failed node.
    - Live plan editing: apply ``PlanPatch`` deltas to the execution plan without
      interrupting the running DAG (hot-patching).
    - Checkpoint / restore: delegate ``RestoreCheckpointCommand`` to the engine,
      enabling deterministic or fresh replay from a prior state.
    - Log-level management: adjust the server-wide log level at runtime via
      ``SetLogLevelCommand`` and broadcast the change back to all connected clients.

    Integration
    -----------
    This handler is called directly from ``handle_command()`` in ``app.py`` after
    ``CommandDispatcher`` returns no match — because these are *typed* Pydantic
    commands (not dict-based), they use ``can_handle()`` for isinstance dispatch
    rather than string-based ``command_type`` routing.

    All heavy lifting (scheduling, state transitions, persistence) is delegated
    to the injected ``ExecutionEngine``, keeping this class a thin coordination
    layer that translates WebSocket commands into engine API calls.
    """

    def __init__(
        self,
        *,
        send_event: Callable[[str, object], Awaitable[None]],
        get_execution_engine: Callable[[], object],
        apply_plan_patches: Callable[[object, list[PlanPatch]], bool],
        logger: logging.Logger | None = None,
    ) -> None:
        self._send_event = send_event
        self._get_execution_engine = get_execution_engine
        self._apply_plan_patches = apply_plan_patches
        self._logger = logger or logging.getLogger(__name__)

    @staticmethod
    def can_handle(command: object) -> bool:
        return isinstance(command, _HANDLED_TYPES)

    async def handle(self, command: ClientCommand, session_id: str) -> None:
        if isinstance(command, StartExecutionCommand):
            await self._handle_start_execution(command, session_id)
        elif isinstance(command, PauseCommand):
            await self._get_execution_engine().pause_execution(command.execution_id)
        elif isinstance(command, ResumeCommand):
            await self._get_execution_engine().resume_execution(command.execution_id)
        elif isinstance(command, AbortCommand):
            await self._get_execution_engine().abort_execution(command.execution_id)
        elif isinstance(command, RetryNodeCommand):
            await self._handle_retry_node(command, session_id)
        elif isinstance(command, PatchPlanCommand):
            await self._handle_patch_plan(command, session_id)
        elif isinstance(command, RestoreCheckpointCommand):
            await self._handle_restore_checkpoint(command, session_id)
        elif isinstance(command, SetLogLevelCommand):
            await self._handle_set_log_level(command, session_id)

    @staticmethod
    def _event_id() -> str:
        return f"evt_{uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # StartExecution
    # ------------------------------------------------------------------
    async def _handle_start_execution(
        self, command: StartExecutionCommand, session_id: str
    ) -> None:
        from houyi.orchestration.plan import NodeType
        from houyi.protocol.ir.plan_ir import EdgeIR, NodeIR, PlanIR

        engine = self._get_execution_engine()

        plan_data = command.inputs.get("plan") if command.inputs else None
        run_settings = command.inputs.get("run_settings") if command.inputs else None

        if not plan_data:
            self._logger.error("No plan data provided in start_execution command")
            return

        nodes = []
        for node_data in plan_data.get("nodes", []):
            node = NodeIR(
                node_id=node_data.get("node_id", node_data.get("id", "")),
                node_type=NodeType(node_data.get("node_type", node_data.get("type", "llm"))),
                position=node_data.get("position", {"x": 0, "y": 0}),
                config=node_data.get("config", {}),
                inputs=node_data.get("inputs", {}),
                outputs=node_data.get("outputs", {}),
                metadata=node_data.get("metadata", {}),
            )
            nodes.append(node)

        edges = []
        for edge_data in plan_data.get("edges", []):
            edge = EdgeIR(
                edge_id=edge_data["id"],
                source_node_id=edge_data["source"],
                target_node_id=edge_data["target"],
                metadata=edge_data.get("metadata", {}),
            )
            edges.append(edge)

        plan = PlanIR(
            plan_id=command.plan_id,
            nodes=nodes,
            edges=edges,
            entry_node_id=nodes[0].node_id if nodes else None,
            metadata={"source": "console_ui"},
            version=0,
        )
        self._logger.info(
            "Converted frontend plan to PlanIR: %d nodes, %d edges", len(nodes), len(edges)
        )

        # Validate tool nodes
        validation_errors: list[str] = []
        for node in plan.nodes:
            if node.node_type != NodeType.TOOL:
                continue
            tool_name: str | None = None
            if isinstance(node.config, dict):
                tool_name = node.config.get("tool_name")
            if not tool_name and isinstance(node.metadata, dict):
                tool_name = node.metadata.get("tool_name") or node.metadata.get("skill_name")
            if not tool_name:
                validation_errors.append(f"tool node {node.node_id} missing tool_name")

        if validation_errors:
            self._logger.debug(
                "Start execution validation failed: %s", "; ".join(validation_errors)
            )
            event = ExecutionStatusEvent(
                event_id=self._event_id(),
                session_id=session_id,
                execution_id=f"exec_validation_{uuid4().hex[:8]}",
                status=ExecutionStatus.FAILED,
                message=f"Execution failed: {'; '.join(validation_errors)}",
            )
            await self._send_event(session_id, event)
            return

        self._logger.info("Starting execution with plan: %s", plan.plan_id)
        await engine.start_execution(session_id, plan, run_settings=run_settings)
        self._logger.info("Execution started successfully")

    # ------------------------------------------------------------------
    # RetryNode
    # ------------------------------------------------------------------
    async def _handle_retry_node(self, command: RetryNodeCommand, session_id: str) -> None:
        engine = self._get_execution_engine()
        self._logger.info(
            "Retry node: execution_id=%s, node_id=%s",
            command.execution_id,
            command.node_id,
        )
        await engine.retry_node(
            session_id=session_id,
            execution_id=command.execution_id,
            node_id=command.node_id,
            new_inputs=command.new_inputs,
        )

    # ------------------------------------------------------------------
    # PatchPlan
    # ------------------------------------------------------------------
    async def _handle_patch_plan(self, command: PatchPlanCommand, session_id: str) -> None:
        from houyi.protocol.ir.plan_ir import PlanIR

        engine = self._get_execution_engine()
        self._logger.info(
            "Patch plan: execution_id=%s, base_version=%s, patches=%d",
            command.execution_id,
            command.base_version,
            len(command.patches),
        )

        current_plan = engine.plan_service.get_current_plan(session_id)
        if not current_plan:
            current_plan = PlanIR(
                plan_id=f"plan_{int(datetime.now().timestamp() * 1000)}",
                version=0,
                nodes=[],
                edges=[],
                entry_node_id="",
                metadata={"source": "console_ui"},
            )
            engine.plan_service.set_current_plan(session_id, current_plan, persist=False)
            self._logger.debug("Created new empty plan for session: %s", session_id)

        self._logger.debug("Processing %d patches", len(command.patches))
        plan_modified = self._apply_plan_patches(current_plan, command.patches)

        if plan_modified:
            current_plan.version += 1
            self._logger.debug("Plan patched successfully")
            engine.plan_service.save_plan_to_file(session_id, current_plan)

            from ..gateway.events import PlanUpdatedEvent

            plan_event = PlanUpdatedEvent(
                event_id=self._event_id(),
                session_id=session_id,
                plan=current_plan,
            )
            await self._send_event(session_id, plan_event)
            self._logger.debug("Plan updated and sent to frontend")

    # ------------------------------------------------------------------
    # RestoreCheckpoint
    # ------------------------------------------------------------------
    async def _handle_restore_checkpoint(
        self, command: RestoreCheckpointCommand, session_id: str
    ) -> None:
        engine = self._get_execution_engine()
        self._logger.info(
            "Restore checkpoint: execution_id=%s, checkpoint_id=%s, mode=%s",
            command.execution_id,
            command.checkpoint_id,
            command.replay_mode,
        )
        await engine.restore_checkpoint(
            session_id=session_id,
            checkpoint_id=command.checkpoint_id,
            replay_mode=getattr(command.replay_mode, "value", command.replay_mode),
            execution_id=command.execution_id,
        )

    # ------------------------------------------------------------------
    # SetLogLevel
    # ------------------------------------------------------------------
    async def _handle_set_log_level(self, command: SetLogLevelCommand, session_id: str) -> None:
        resolved_level = set_log_level(command.level)
        self._logger.info(
            "Log level updated: requested=%s resolved=%s effective=%s",
            command.level,
            resolved_level,
            get_log_level(),
        )
        await self._send_event(
            session_id,
            LogLevelEvent(
                event_id=self._event_id(),
                session_id=session_id,
                level=resolved_level.lower(),
                requested_level=command.level,
            ),
        )
