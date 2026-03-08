from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from houyi.interface.protocol.ir import (
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
            metadata=checkpoint.metadata if isinstance(checkpoint.metadata, dict) else {},
        )

        logger.debug("Created checkpoint: %s", checkpoint_id)
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
        replay_mode_value = getattr(replay_mode, "value", replay_mode)

        checkpoint = self._find_checkpoint(checkpoint_id=checkpoint_id, execution_id=execution_id)

        if not checkpoint:
            logger.error("Checkpoint not found: %s", checkpoint_id)
            return self._build_restore_failure(
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                execution_id=execution_id,
                replay_mode=replay_mode_value,
                message="Checkpoint not found",
            )

        plan = self._resolve_plan_for_restore(session_id=session_id, checkpoint=checkpoint)

        if not plan:
            logger.error("Plan not found for session: %s", session_id)
            return self._build_restore_failure(
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                execution_id=execution_id,
                replay_mode=replay_mode_value,
                message="Plan not found for session",
            )

        execution, parent_execution_id = self._fork_execution_from_checkpoint(
            checkpoint=checkpoint,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
        )
        self._log_restored_node_statuses(
            checkpoint_id=checkpoint_id,
            execution_id=execution.execution_id,
            replay_mode=replay_mode,
            execution=execution,
        )
        self._cancel_existing_task(parent_execution_id)
        self._apply_replay_mode(
            execution=execution, checkpoint=checkpoint, replay_mode_value=replay_mode_value
        )
        self._reset_execution_nodes(execution=execution, checkpoint=checkpoint, plan=plan)
        self._finalize_restored_execution(
            execution=execution, parent_execution_id=parent_execution_id
        )
        execution_status = self._build_execution_status(
            session_id=session_id, execution=execution, checkpoint_id=checkpoint_id
        )
        node_statuses = self._build_node_statuses(session_id=session_id, execution=execution)

        logger.info(
            "Checkpoint restored: %s (execution: %s, status: %s)",
            checkpoint_id,
            execution.execution_id,
            execution.status,
        )

        return RestoreCheckpointOutcome(
            execution=execution,
            execution_status=execution_status,
            node_statuses=node_statuses,
            result=self._build_restore_success(
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                execution_id=execution.execution_id,
                replay_mode=replay_mode_value,
            ),
        )

    def _find_checkpoint(
        self,
        *,
        checkpoint_id: str,
        execution_id: str | None,
    ) -> CheckpointIR | None:
        if execution_id:
            for checkpoint in self._checkpoint_store.get(execution_id):
                if checkpoint.checkpoint_id == checkpoint_id:
                    return checkpoint
            return None

        logger.warning(
            "restore_checkpoint called without execution_id; checkpoint_id may be ambiguous: %s",
            checkpoint_id,
        )
        for checkpoints in self._checkpoint_store.checkpoints.values():
            for checkpoint in checkpoints:
                if checkpoint.checkpoint_id == checkpoint_id:
                    return checkpoint
        return None

    def _resolve_plan_for_restore(
        self,
        *,
        session_id: str,
        checkpoint: CheckpointIR,
    ) -> PlanIR | None:
        plan = self._plan_store.get(session_id)
        if plan:
            return plan

        snapshot_session_id = None
        if isinstance(checkpoint.execution_snapshot, dict):
            snapshot_metadata = checkpoint.execution_snapshot.get("metadata")
            if isinstance(snapshot_metadata, dict):
                snapshot_session_id = snapshot_metadata.get("session_id")

        if not snapshot_session_id or snapshot_session_id == session_id:
            return None

        recovered_plan = self._plan_store.get(str(snapshot_session_id))
        if not recovered_plan:
            return None

        self._plan_store.set(session_id, recovered_plan, persist=False)
        logger.info(
            "Recovered plan for restored checkpoint from snapshot session_id=%s -> current session_id=%s",
            snapshot_session_id,
            session_id,
        )
        return recovered_plan

    def _build_restore_failure(
        self,
        *,
        session_id: str,
        checkpoint_id: str,
        execution_id: str | None,
        replay_mode: str | None,
        message: str,
    ) -> RestoreCheckpointOutcome:
        return RestoreCheckpointOutcome(
            execution=None,
            execution_status=None,
            node_statuses=[],
            result=RestoreCheckpointResultPayload(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                execution_id=execution_id,
                replay_mode=str(replay_mode) if replay_mode else None,
                success=False,
                message=message,
            ),
        )

    def _fork_execution_from_checkpoint(
        self,
        *,
        checkpoint: CheckpointIR,
        session_id: str,
        checkpoint_id: str,
    ) -> tuple[ExecutionIR, str]:
        execution = ExecutionIR(**checkpoint.execution_snapshot)
        parent_execution_id = execution.execution_id
        execution.execution_id = (
            f"exec_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex[:6]}"
        )
        execution.plan_id = checkpoint.plan_id
        execution.metadata = dict(execution.metadata or {})
        execution.metadata["session_id"] = session_id
        execution.metadata["parent_execution_id"] = parent_execution_id
        execution.metadata["parent_checkpoint_id"] = checkpoint_id
        return execution, parent_execution_id

    def _log_restored_node_statuses(
        self,
        *,
        checkpoint_id: str,
        execution_id: str,
        replay_mode: str,
        execution: ExecutionIR,
    ) -> None:
        try:
            status_counts = Counter(
                str(getattr(node_exec, "status", "unknown"))
                for node_exec in (execution.node_executions or {}).values()
            )
            logger.debug(
                "Restored checkpoint snapshot node status distribution: checkpoint=%s execution=%s mode=%s counts=%s",
                checkpoint_id,
                execution_id,
                replay_mode,
                dict(status_counts),
            )
        except Exception:
            logger.debug(
                "Failed to compute restored checkpoint node status distribution", exc_info=True
            )

    def _cancel_existing_task(self, parent_execution_id: str) -> None:
        existing_task = self._execution_tasks.get(parent_execution_id)
        if existing_task and not existing_task.done():
            existing_task.cancel()
            logger.info("Cancelled existing execution task before restore: %s", parent_execution_id)

    def _apply_replay_mode(
        self,
        *,
        execution: ExecutionIR,
        checkpoint: CheckpointIR,
        replay_mode_value: str | None,
    ) -> None:
        execution.status = ExecutionStatus.PAUSED
        execution.metadata["replay_mode"] = str(replay_mode_value)
        if replay_mode_value == "deterministic":
            self._llm_call_logs[execution.execution_id] = checkpoint.llm_call_logs.copy()
            logger.info(
                "Restored %d LLM call logs for deterministic replay",
                len(checkpoint.llm_call_logs),
            )
            return

        self._llm_call_logs[execution.execution_id] = []
        logger.info("Fresh replay mode: cleared LLM call logs")

    def _reset_execution_nodes(
        self,
        *,
        execution: ExecutionIR,
        checkpoint: CheckpointIR,
        plan: PlanIR,
    ) -> None:
        trigger_node_id = (
            checkpoint.metadata.get("trigger_node_id")
            if isinstance(checkpoint.metadata, dict)
            else None
        )
        node_order = list(self._get_execution_order(plan))
        reset_from_index = 0
        if trigger_node_id and trigger_node_id in node_order:
            reset_from_index = node_order.index(trigger_node_id) + 1

        nodes_to_reset = node_order[reset_from_index:]
        if not nodes_to_reset and trigger_node_id:
            logger.info(
                "Terminal checkpoint detected: checkpoint=%s trigger_node=%s — applying replay-all (resetting all %d nodes)",
                checkpoint.checkpoint_id,
                trigger_node_id,
                len(node_order),
            )
            nodes_to_reset = node_order
            execution.metadata["replay_all"] = True

        for node_id in nodes_to_reset:
            node_exec = (execution.node_executions or {}).get(node_id)
            if node_exec is None:
                continue
            node_exec.status = NodeStatus.PENDING
            node_exec.started_at = None
            node_exec.completed_at = None
            node_exec.outputs = {}
            node_exec.error = None
            node_exec.streaming_output = ""

    def _finalize_restored_execution(
        self, *, execution: ExecutionIR, parent_execution_id: str
    ) -> None:
        if execution.metadata.get("replay_mode") == "fresh":
            execution.started_at = None
            execution.completed_at = None
            execution.error = None

        self._execution_store.save(execution)
        parent_checkpoints = list(self._checkpoint_store.get(parent_execution_id))
        self._checkpoint_store.checkpoints[execution.execution_id] = parent_checkpoints

    def _build_execution_status(
        self,
        *,
        session_id: str,
        execution: ExecutionIR,
        checkpoint_id: str,
    ) -> ExecutionStatusPayload:
        return ExecutionStatusPayload(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            execution_id=execution.execution_id,
            status=execution.status,
            message=f"Restored from checkpoint {checkpoint_id}",
        )

    def _build_node_statuses(
        self,
        *,
        session_id: str,
        execution: ExecutionIR,
    ) -> list[NodeStatusPayload]:
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
                    outputs=node_exec.outputs,  # type: ignore[arg-type]
                    error=node_exec.error,
                )
            )
        return node_statuses

    def _build_restore_success(
        self,
        *,
        session_id: str,
        checkpoint_id: str,
        execution_id: str,
        replay_mode: str | None,
    ) -> RestoreCheckpointResultPayload:
        return RestoreCheckpointResultPayload(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            execution_id=execution_id,
            replay_mode=str(replay_mode),
            success=True,
            message=f"Restored from checkpoint {checkpoint_id}",
        )
