"""Execution lifecycle core logic.

This module intentionally avoids any server transport concerns
(e.g. FastAPI/WebSocket/event emission). The server layer should adapt outcomes
into its event protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from houyi.interface.protocol.ir import ExecutionIR, ExecutionStatus, PlanIR

logger = logging.getLogger(__name__)


class ExecutionStoreLike(Protocol):
    def get(self, execution_id: str) -> ExecutionIR | None: ...

    def save(self, execution: ExecutionIR) -> None: ...


class CheckpointStoreLike(Protocol):
    def init_execution(self, execution_id: str) -> None: ...


class PlanStoreLike(Protocol):
    def set(self, session_id: str, plan: PlanIR, persist: bool = True) -> None: ...

    def get_cached(self, session_id: str) -> PlanIR | None: ...


class ExecutionBackendLike(Protocol):
    async def start(
        self, session_id: str, execution: ExecutionIR, plan: PlanIR, execute_plan: Any
    ) -> Any: ...


@dataclass(slots=True)
class ExecutionStatusPayload:
    event_id: str
    session_id: str
    execution_id: str
    status: ExecutionStatus
    message: str


@dataclass(slots=True)
class StartExecutionOutcome:
    execution: ExecutionIR
    status_event: ExecutionStatusPayload


@dataclass(slots=True)
class SimpleStatusOutcome:
    status_event: ExecutionStatusPayload | None


class ExecutionLifecycleManager:
    """Manage execution lifecycle transitions."""

    def __init__(
        self,
        *,
        execution_store: ExecutionStoreLike,
        checkpoint_store: CheckpointStoreLike,
        plan_store: PlanStoreLike,
        execution_backend: ExecutionBackendLike,
        execution_tasks: dict[str, Any],
        llm_call_logs: dict[str, list[Any]],
        normalize_run_settings: Any,
        execute_plan: Any,
    ) -> None:
        self._execution_store = execution_store
        self._checkpoint_store = checkpoint_store
        self._plan_store = plan_store
        self._execution_backend = execution_backend
        self._execution_tasks = execution_tasks
        self._llm_call_logs = llm_call_logs
        self._normalize_run_settings = normalize_run_settings
        self._execute_plan = execute_plan

    async def start_execution(
        self,
        *,
        session_id: str,
        plan: PlanIR,
        run_settings: dict[str, Any] | None = None,
    ) -> StartExecutionOutcome:
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex[:6]}"
        normalized_run_settings = self._normalize_run_settings(run_settings)

        execution = ExecutionIR(
            execution_id=execution_id,
            plan_id=plan.plan_id,
            status=ExecutionStatus.RUNNING,
            node_executions={},
            context={},
            started_at=datetime.now(),
            completed_at=None,
            error=None,
            metadata={
                "session_id": session_id,
                "run_settings": normalized_run_settings,
            },
        )

        self._execution_store.save(execution)
        self._checkpoint_store.init_execution(execution_id)
        self._llm_call_logs[execution_id] = []
        self._plan_store.set(session_id, plan)

        status_event = ExecutionStatusPayload(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            execution_id=execution_id,
            status=ExecutionStatus.RUNNING,
            message="Execution started",
        )

        task = await self._execution_backend.start(session_id, execution, plan, self._execute_plan)
        self._execution_tasks[execution_id] = task

        logger.info("Started execution: %s", execution_id)
        return StartExecutionOutcome(execution=execution, status_event=status_event)

    async def pause_execution(self, *, execution_id: str) -> SimpleStatusOutcome:
        execution = self._execution_store.get(execution_id)
        if not execution:
            logger.warning("Execution not found: %s", execution_id)
            return SimpleStatusOutcome(status_event=None)

        if execution.status != ExecutionStatus.RUNNING:
            logger.warning("Cannot pause execution in state: %s", execution.status)
            return SimpleStatusOutcome(status_event=None)

        execution.status = ExecutionStatus.PAUSED
        logger.info("Paused execution: %s", execution_id)

        session_id = execution.metadata.get("session_id", "unknown")
        return SimpleStatusOutcome(
            status_event=ExecutionStatusPayload(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution_id,
                status=ExecutionStatus.PAUSED,
                message="Execution paused",
            )
        )

    async def resume_execution(self, *, execution_id: str) -> SimpleStatusOutcome:
        execution = self._execution_store.get(execution_id)
        if not execution:
            logger.warning("Execution not found: %s", execution_id)
            return SimpleStatusOutcome(status_event=None)

        if execution.status != ExecutionStatus.PAUSED:
            logger.warning("Cannot resume execution in state: %s", execution.status)
            return SimpleStatusOutcome(status_event=None)

        session_id = execution.metadata.get("session_id", "unknown")
        current_plan = self._plan_store.get_cached(session_id)
        if not current_plan:
            logger.error("No plan found for session: %s", session_id)
            return SimpleStatusOutcome(status_event=None)

        execution.status = ExecutionStatus.RUNNING
        logger.info("Resumed execution: %s with updated plan", execution_id)

        task = self._execution_tasks.get(execution_id)
        if not task or getattr(task, "done", lambda: True)():
            task = await self._execution_backend.start(
                session_id,
                execution,
                current_plan,
                self._execute_plan,
            )
            self._execution_tasks[execution_id] = task
            logger.info("Started execution task on resume: %s", execution_id)

        return SimpleStatusOutcome(
            status_event=ExecutionStatusPayload(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution_id,
                status=ExecutionStatus.RUNNING,
                message="Execution resumed with updated plan",
            )
        )

    async def abort_execution(self, *, execution_id: str) -> SimpleStatusOutcome:
        execution = self._execution_store.get(execution_id)
        if not execution:
            logger.warning("Execution not found: %s", execution_id)
            return SimpleStatusOutcome(status_event=None)

        execution.status = ExecutionStatus.ABORTED
        execution.completed_at = datetime.now()

        task = self._execution_tasks.get(execution_id)
        if task and not getattr(task, "done", lambda: True)():
            task.cancel()

        logger.info("Aborted execution: %s", execution_id)

        session_id = execution.metadata.get("session_id", "unknown")
        return SimpleStatusOutcome(
            status_event=ExecutionStatusPayload(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution_id,
                status=ExecutionStatus.ABORTED,
                message="Execution aborted",
            )
        )
