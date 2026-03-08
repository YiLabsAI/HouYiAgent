"""Plan execution loop for the console execution engine."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from uuid import uuid4

from houyi.application.workflow.plan_execution_loop import (
    PlanExecutionLoop as _SdkPlanExecutionLoop,
)
from houyi.interface.protocol.ir import ExecutionIR, ExecutionStatus, PlanIR

from ..gateway.events import ExecutionStatusEvent
from .context import ExecutionContext
from .observation_service import ObservationService

logger = logging.getLogger(__name__)

ContextFactory = Callable[[str, ExecutionIR, PlanIR], ExecutionContext]
PlanGetter = Callable[[str, PlanIR], PlanIR]
ExecutionOrderResolver = Callable[[PlanIR], list[str]]
NodeExecutor = Callable[[ExecutionContext, str], Awaitable[None]]
CheckpointCallback = Callable[..., Awaitable[None]]
LifecycleNotifier = Callable[..., Awaitable[None]]
SleepFunc = Callable[[float], Awaitable[None]]


class PlanExecutionLoop:
    """Execute plan nodes in order with pause/resume handling."""

    def __init__(
        self,
        *,
        plan_getter: PlanGetter,
        get_execution_order: ExecutionOrderResolver,
        node_executor: NodeExecutor,
        checkpoint_callback: CheckpointCallback,
        observation_service: ObservationService,
        notify_lifecycle: LifecycleNotifier,
        context_factory: ContextFactory,
        sleep_func: SleepFunc | None = None,
    ) -> None:
        self._observation_service = observation_service
        self._sdk_loop = _SdkPlanExecutionLoop(
            plan_getter=plan_getter,
            get_execution_order=get_execution_order,
            node_executor=node_executor,
            checkpoint_callback=checkpoint_callback,
            notify_lifecycle=notify_lifecycle,
            context_factory=context_factory,
            sleep_func=sleep_func,
        )

    def set_sleep_func(self, sleep_func: SleepFunc) -> None:
        self._sdk_loop.set_sleep_func(sleep_func)

    async def execute(self, session_id: str, execution: ExecutionIR, plan: PlanIR) -> None:
        try:
            await self._sdk_loop.execute(session_id, execution, plan)
            event = ExecutionStatusEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                status=ExecutionStatus.COMPLETED,
                message="Execution completed",
            )
            await self._observation_service.emit(event)
        except asyncio.CancelledError:
            logger.debug("Execution task cancelled: %s", execution.execution_id)
            return
        except Exception as exc:
            logger.error(
                "Execution failed: %s - %s",
                execution.execution_id,
                exc,
                exc_info=True,
            )
            execution.status = ExecutionStatus.FAILED
            execution.error = str(exc)
            execution.completed_at = datetime.now()

            try:
                event = ExecutionStatusEvent(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    execution_id=execution.execution_id,
                    status=ExecutionStatus.FAILED,
                    message=f"Execution failed: {exc!s}",
                )
                await self._observation_service.emit(event)
            except Exception as event_error:
                logger.error("Failed to send error event: %s", event_error)
