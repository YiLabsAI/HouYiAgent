"""Execution lifecycle service for starting and managing executions."""

from __future__ import annotations

import logging
from typing import Any

from houyi.application.workflow.execution_backends import ExecutionBackend
from houyi.application.workflow.execution_lifecycle_service import ExecutionLifecycleManager
from houyi.interface.protocol.ir import ExecutionIR, PlanIR

from ..gateway.events import ExecutionStatusEvent
from .observation_service import ObservationService
from .stores import CheckpointStore, ExecutionStore, PlanStore

logger = logging.getLogger(__name__)


class ExecutionLifecycleService:
    """Service that manages execution lifecycle transitions."""

    def __init__(
        self,
        *,
        execution_store: ExecutionStore,
        checkpoint_store: CheckpointStore,
        plan_store: PlanStore,
        observation_service: ObservationService,
        execution_backend: ExecutionBackend,
        execution_tasks: dict[str, Any],
        llm_call_logs: dict[str, list],
        normalize_run_settings: Any,
        execute_plan: Any,
        restore_checkpoint: Any,
    ) -> None:
        self._execution_store = execution_store
        self._checkpoint_store = checkpoint_store
        self._plan_store = plan_store
        self._observation_service = observation_service
        self._execution_backend = execution_backend
        self._execution_tasks = execution_tasks
        self._llm_call_logs = llm_call_logs
        self._normalize_run_settings = normalize_run_settings
        self._execute_plan = execute_plan
        self._restore_checkpoint = restore_checkpoint

        self._manager = ExecutionLifecycleManager(
            execution_store=self._execution_store,
            checkpoint_store=self._checkpoint_store,
            plan_store=self._plan_store,
            execution_backend=self._execution_backend,
            execution_tasks=self._execution_tasks,
            llm_call_logs=self._llm_call_logs,
            normalize_run_settings=self._normalize_run_settings,
            execute_plan=self._execute_plan,
        )

    async def start_execution(
        self,
        session_id: str,
        plan: PlanIR,
        run_settings: dict[str, Any] | None = None,
    ) -> ExecutionIR:
        """Start executing a plan and return the new execution."""
        outcome = await self._manager.start_execution(
            session_id=session_id,
            plan=plan,
            run_settings=run_settings,
        )

        await self._observation_service.emit(
            ExecutionStatusEvent(
                event_id=outcome.status_event.event_id,
                session_id=outcome.status_event.session_id,
                execution_id=outcome.status_event.execution_id,
                status=outcome.status_event.status,
                message=outcome.status_event.message,
            )
        )

        return outcome.execution

    async def pause_execution(self, execution_id: str) -> None:
        """Pause an execution if it is currently running."""
        outcome = await self._manager.pause_execution(execution_id=execution_id)
        if outcome.status_event is None:
            return
        await self._observation_service.emit(
            ExecutionStatusEvent(
                event_id=outcome.status_event.event_id,
                session_id=outcome.status_event.session_id,
                execution_id=outcome.status_event.execution_id,
                status=outcome.status_event.status,
                message=outcome.status_event.message,
            )
        )

    async def resume_execution(self, execution_id: str) -> None:
        """Resume a paused execution using the latest plan."""
        outcome = await self._manager.resume_execution(execution_id=execution_id)
        if outcome.status_event is None:
            return
        await self._observation_service.emit(
            ExecutionStatusEvent(
                event_id=outcome.status_event.event_id,
                session_id=outcome.status_event.session_id,
                execution_id=outcome.status_event.execution_id,
                status=outcome.status_event.status,
                message=outcome.status_event.message,
            )
        )

    async def abort_execution(self, execution_id: str) -> None:
        """Abort an execution and emit status updates."""
        outcome = await self._manager.abort_execution(execution_id=execution_id)
        if outcome.status_event is None:
            logger.warning("abort_execution: no status_event returned for %s", execution_id)
            return
        logger.info(
            "abort_execution: emitting status=%s session=%s exec=%s",
            outcome.status_event.status,
            outcome.status_event.session_id,
            outcome.status_event.execution_id,
        )
        await self._observation_service.emit(
            ExecutionStatusEvent(
                event_id=outcome.status_event.event_id,
                session_id=outcome.status_event.session_id,
                execution_id=outcome.status_event.execution_id,
                status=outcome.status_event.status,
                message=outcome.status_event.message,
            )
        )

    async def restore_checkpoint(
        self,
        *,
        session_id: str,
        checkpoint_id: str,
        replay_mode: str = "deterministic",
        execution_id: str | None = None,
    ) -> None:
        """Restore an execution from a checkpoint."""
        await self._restore_checkpoint(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            replay_mode=replay_mode,
            execution_id=execution_id,
        )
