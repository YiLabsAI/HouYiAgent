"""Checkpoint service for execution persistence."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from houyi.checkpoint import CheckpointManager
from houyi.execution.execution_order_service import ExecutionOrderService
from houyi.protocol.ir import (
    CheckpointTrigger,
    ExecutionIR,
)

from .events import (
    CheckpointCreatedEvent,
    ExecutionStatusEvent,
    NodeStatusEvent,
    RestoreCheckpointResultEvent,
)
from .observation_service import ObservationService
from .stores import CheckpointStore, ExecutionStore, PlanStore

logger = logging.getLogger(__name__)


class CheckpointService:
    """Manage checkpoint creation and restoration for executions."""

    def __init__(
        self,
        *,
        checkpoint_store: CheckpointStore,
        execution_store: ExecutionStore,
        plan_store: PlanStore,
        observation_service: ObservationService,
        execution_tasks: dict[str, asyncio.Task],
        llm_call_logs: dict[str, list[Any]],
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._execution_store = execution_store
        self._plan_store = plan_store
        self._observation_service = observation_service
        self._execution_tasks = execution_tasks
        self._llm_call_logs = llm_call_logs

        self._execution_order_service = ExecutionOrderService()
        self._manager = CheckpointManager(
            checkpoint_store=self._checkpoint_store,
            execution_store=self._execution_store,
            plan_store=self._plan_store,
            execution_tasks=self._execution_tasks,
            llm_call_logs=self._llm_call_logs,
            get_execution_order=self._execution_order_service.get_execution_order,
        )

    async def create_checkpoint(
        self,
        session_id: str,
        execution: ExecutionIR,
        trigger: CheckpointTrigger,
        node_id: str | None = None,
    ) -> None:
        """Create and emit a checkpoint for the current execution state."""
        checkpoint, payload = await self._manager.create_checkpoint(
            session_id=session_id,
            execution=execution,
            trigger=trigger,
            node_id=node_id,
        )

        event = CheckpointCreatedEvent(
            event_id=payload.event_id,
            session_id=payload.session_id,
            checkpoint_id=payload.checkpoint_id,
            execution_id=payload.execution_id,
            sequence_number=payload.sequence_number,
            trigger=payload.trigger,
            llm_call_logs=payload.llm_call_logs,
        )
        await self._observation_service.emit(event)

        logger.info("Created checkpoint: %s", checkpoint.checkpoint_id)

    async def restore_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        replay_mode: str = "deterministic",
        execution_id: str | None = None,
    ) -> None:
        """Restore execution state from a checkpoint."""
        outcome = await self._manager.restore_checkpoint(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            replay_mode=replay_mode,
            execution_id=execution_id,
        )

        if outcome.execution_status is not None:
            await self._observation_service.emit(
                ExecutionStatusEvent(
                    event_id=outcome.execution_status.event_id,
                    session_id=outcome.execution_status.session_id,
                    execution_id=outcome.execution_status.execution_id,
                    status=outcome.execution_status.status,
                    message=outcome.execution_status.message,
                )
            )

        for node_payload in outcome.node_statuses:
            await self._observation_service.emit(
                NodeStatusEvent(
                    event_id=node_payload.event_id,
                    session_id=node_payload.session_id,
                    execution_id=node_payload.execution_id,
                    node_id=node_payload.node_id,
                    status=node_payload.status,
                    inputs=node_payload.inputs,
                    outputs=node_payload.outputs,
                    error=node_payload.error,
                )
            )

        await self._observation_service.emit(
            RestoreCheckpointResultEvent(
                event_id=outcome.result.event_id,
                session_id=outcome.result.session_id,
                checkpoint_id=outcome.result.checkpoint_id,
                execution_id=outcome.result.execution_id,
                replay_mode=outcome.result.replay_mode,
                success=outcome.result.success,
                message=outcome.result.message,
            )
        )
