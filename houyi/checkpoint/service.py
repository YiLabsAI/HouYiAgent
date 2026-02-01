from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from houyi.protocol.ir import (
    CheckpointIR,
    CheckpointTrigger,
    ExecutionIR,
    ExecutionStatus,
    NodeStatus,
    PlanIR,
)

logger = logging.getLogger(__name__)


class CheckpointStoreLike(Protocol):
    checkpoints: dict[str, list[CheckpointIR]]

    def get(self, execution_id: str) -> list[CheckpointIR]: ...

    def add(self, execution_id: str, checkpoint: CheckpointIR) -> None: ...


class ExecutionStoreLike(Protocol):
    def save(self, execution: ExecutionIR) -> None: ...


class PlanStoreLike(Protocol):
    def get(self, session_id: str) -> PlanIR | None: ...

    def set(self, session_id: str, plan: PlanIR, persist: bool = True) -> None: ...


class CancellableTaskLike(Protocol):
    def done(self) -> bool: ...

    def cancel(self) -> None: ...


@dataclass(slots=True)
class CheckpointCreatedPayload:
    event_id: str
    session_id: str
    checkpoint_id: str
    execution_id: str
    sequence_number: int
    trigger: CheckpointTrigger
    llm_call_logs: list[Any]


@dataclass(slots=True)
class ExecutionStatusPayload:
    event_id: str
    session_id: str
    execution_id: str
    status: ExecutionStatus
    message: str | None


@dataclass(slots=True)
class NodeStatusPayload:
    event_id: str
    session_id: str
    execution_id: str
    node_id: str
    status: NodeStatus
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    error: str | None


@dataclass(slots=True)
class RestoreCheckpointResultPayload:
    event_id: str
    session_id: str
    checkpoint_id: str
    execution_id: str | None
    replay_mode: str | None
    success: bool
    message: str | None


@dataclass(slots=True)
class RestoreCheckpointOutcome:
    execution: ExecutionIR | None
    execution_status: ExecutionStatusPayload | None
    node_statuses: list[NodeStatusPayload]
    result: RestoreCheckpointResultPayload


class CheckpointManager:
    def __init__(
        self,
        *,
        checkpoint_store: CheckpointStoreLike,
        execution_store: ExecutionStoreLike,
        plan_store: PlanStoreLike,
        execution_tasks: dict[str, CancellableTaskLike],
        llm_call_logs: dict[str, list[Any]],
        get_execution_order: Any,
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._execution_store = execution_store
        self._plan_store = plan_store
        self._execution_tasks = execution_tasks
        self._llm_call_logs = llm_call_logs
        self._get_execution_order = get_execution_order

    async def create_checkpoint(
        self,
        *,
        session_id: str,
        execution: ExecutionIR,
        trigger: CheckpointTrigger,
        node_id: str | None = None,
    ) -> tuple[CheckpointIR, CheckpointCreatedPayload]:
        checkpoint_id = f"cp_{len(self._checkpoint_store.get(execution.execution_id)) + 1}"
        llm_logs = self._llm_call_logs.get(execution.execution_id, [])

        checkpoint = CheckpointIR(
            checkpoint_id=checkpoint_id,
            execution_id=execution.execution_id,
            plan_id=execution.plan_id,
            sequence_number=len(self._checkpoint_store.get(execution.execution_id)) + 1,
            trigger=trigger,
            created_at=datetime.now(),
            execution_snapshot=execution.model_dump(),
            llm_call_logs=llm_logs,
            parent_checkpoint_id=None,
            delta=None,
            metadata={"trigger_node_id": node_id} if node_id else {},
        )

        self._checkpoint_store.add(execution.execution_id, checkpoint)

        payload = CheckpointCreatedPayload(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            execution_id=execution.execution_id,
            sequence_number=checkpoint.sequence_number,
            trigger=trigger,
            llm_call_logs=llm_logs,
        )

        logger.info("Created checkpoint: %s", checkpoint_id)
        return checkpoint, payload

    async def restore_checkpoint(
        self,
        *,
        session_id: str,
        checkpoint_id: str,
        replay_mode: str = "deterministic",
        execution_id: str | None = None,
    ) -> RestoreCheckpointOutcome:
        logger.info(
            "Restoring checkpoint: %s (mode: %s)",
            checkpoint_id,
            replay_mode,
        )

        checkpoint: CheckpointIR | None = None
        if execution_id:
            for cp in self._checkpoint_store.get(execution_id):
                if cp.checkpoint_id == checkpoint_id:
                    checkpoint = cp
                    break
        else:
            logger.warning(
                "restore_checkpoint called without execution_id; checkpoint_id may be ambiguous: %s",
                checkpoint_id,
            )
            for _exec_id, checkpoints in self._checkpoint_store.checkpoints.items():
                for cp in checkpoints:
                    if cp.checkpoint_id == checkpoint_id:
                        checkpoint = cp
                        break
                if checkpoint:
                    break

        replay_mode_value = getattr(replay_mode, "value", replay_mode)

        if not checkpoint:
            logger.error("Checkpoint not found: %s", checkpoint_id)
            return RestoreCheckpointOutcome(
                execution=None,
                execution_status=None,
                node_statuses=[],
                result=RestoreCheckpointResultPayload(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    checkpoint_id=checkpoint_id,
                    execution_id=execution_id,
                    replay_mode=str(replay_mode_value) if replay_mode_value else None,
                    success=False,
                    message="Checkpoint not found",
                ),
            )

        plan = self._plan_store.get(session_id)
        if not plan:
            snapshot_session_id = None
            if isinstance(checkpoint.execution_snapshot, dict):
                snapshot_metadata = checkpoint.execution_snapshot.get("metadata")
                if isinstance(snapshot_metadata, dict):
                    snapshot_session_id = snapshot_metadata.get("session_id")
            if snapshot_session_id and snapshot_session_id != session_id:
                recovered_plan = self._plan_store.get(str(snapshot_session_id))
                if recovered_plan:
                    self._plan_store.set(session_id, recovered_plan, persist=False)
                    plan = recovered_plan
                    logger.info(
                        "Recovered plan for restored checkpoint from snapshot session_id=%s -> current session_id=%s",
                        snapshot_session_id,
                        session_id,
                    )

        if not plan:
            logger.error("Plan not found for session: %s", session_id)
            return RestoreCheckpointOutcome(
                execution=None,
                execution_status=None,
                node_statuses=[],
                result=RestoreCheckpointResultPayload(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    checkpoint_id=checkpoint_id,
                    execution_id=execution_id,
                    replay_mode=str(replay_mode_value),
                    success=False,
                    message="Plan not found for session",
                ),
            )

        execution_data = checkpoint.execution_snapshot
        execution = ExecutionIR(**execution_data)

        parent_execution_id = execution.execution_id
        forked_execution_id = f"exec_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex[:6]}"
        execution.execution_id = forked_execution_id
        execution.plan_id = checkpoint.plan_id

        try:
            status_counts = Counter(
                str(getattr(node_exec, "status", "unknown"))
                for node_exec in (execution.node_executions or {}).values()
            )
            logger.debug(
                "Restored checkpoint snapshot node status distribution: checkpoint=%s execution=%s mode=%s counts=%s",
                checkpoint_id,
                execution.execution_id,
                replay_mode,
                dict(status_counts),
            )
        except Exception:
            logger.debug(
                "Failed to compute restored checkpoint node status distribution", exc_info=True
            )

        execution.metadata = dict(execution.metadata or {})
        execution.metadata["session_id"] = session_id
        execution.metadata["parent_execution_id"] = parent_execution_id
        execution.metadata["parent_checkpoint_id"] = checkpoint_id

        existing_task = self._execution_tasks.get(parent_execution_id)
        if existing_task and not existing_task.done():
            existing_task.cancel()
            logger.info("Cancelled existing execution task before restore: %s", parent_execution_id)

        execution.status = ExecutionStatus.PAUSED
        execution.metadata["replay_mode"] = str(replay_mode_value)

        if replay_mode_value == "deterministic":
            self._llm_call_logs[execution.execution_id] = checkpoint.llm_call_logs.copy()
            logger.info(
                "Restored %d LLM call logs for deterministic replay",
                len(checkpoint.llm_call_logs),
            )
        else:
            self._llm_call_logs[execution.execution_id] = []
            logger.info("Fresh replay mode: cleared LLM call logs")

        trigger_node_id = None
        if isinstance(checkpoint.metadata, dict):
            trigger_node_id = checkpoint.metadata.get("trigger_node_id")

        node_order = list(self._get_execution_order(plan))
        if trigger_node_id and trigger_node_id in node_order:
            reset_from_index = node_order.index(trigger_node_id) + 1
        else:
            reset_from_index = 0

        for node_id in node_order[reset_from_index:]:
            node_exec = (execution.node_executions or {}).get(node_id)
            if node_exec is None:
                continue
            node_exec.status = NodeStatus.PENDING
            node_exec.started_at = None
            node_exec.completed_at = None
            node_exec.outputs = {}
            node_exec.error = None
            node_exec.streaming_output = ""

        if replay_mode_value == "fresh":
            execution.started_at = None
            execution.completed_at = None
            execution.error = None

        self._execution_store.save(execution)

        parent_checkpoints = list(self._checkpoint_store.get(parent_execution_id))
        self._checkpoint_store.checkpoints[execution.execution_id] = parent_checkpoints

        execution_status = ExecutionStatusPayload(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            execution_id=execution.execution_id,
            status=execution.status,
            message=f"Restored from checkpoint {checkpoint_id}",
        )

        node_statuses: list[NodeStatusPayload] = []
        for node_id, node_exec in (execution.node_executions or {}).items():
            node_statuses.append(
                NodeStatusPayload(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    execution_id=execution.execution_id,
                    node_id=node_id,
                    status=node_exec.status,
                    inputs=node_exec.inputs,
                    outputs=node_exec.outputs,
                    error=node_exec.error,
                )
            )

        logger.info(
            "Checkpoint restored: %s (execution: %s, status: %s)",
            checkpoint_id,
            execution.execution_id,
            execution.status,
        )

        result = RestoreCheckpointResultPayload(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            execution_id=execution.execution_id,
            replay_mode=str(replay_mode_value),
            success=True,
            message=f"Restored from checkpoint {checkpoint_id}",
        )

        return RestoreCheckpointOutcome(
            execution=execution,
            execution_status=execution_status,
            node_statuses=node_statuses,
            result=result,
        )
