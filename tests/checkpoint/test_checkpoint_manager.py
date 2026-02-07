from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from houyi.checkpoint import CheckpointManager
from houyi.protocol.ir import (
    CheckpointIR,
    CheckpointTrigger,
    ExecutionIR,
    ExecutionStatus,
    NodeExecutionIR,
    NodeIR,
    NodeStatus,
    NodeType,
    PlanIR,
)
from houyi.protocol.ir.checkpoint_ir import LLMCallLog


@dataclass
class _CheckpointStore:
    checkpoints: dict[str, list[CheckpointIR]] = field(default_factory=dict)

    def get(self, execution_id: str) -> list[CheckpointIR]:
        return self.checkpoints.get(execution_id, [])

    def add(self, execution_id: str, checkpoint: CheckpointIR) -> None:
        self.checkpoints.setdefault(execution_id, []).append(checkpoint)


@dataclass
class _ExecutionStore:
    saved: dict[str, ExecutionIR] = field(default_factory=dict)

    def save(self, execution: ExecutionIR) -> None:
        self.saved[execution.execution_id] = execution


@dataclass
class _PlanStore:
    plans: dict[str, PlanIR] = field(default_factory=dict)

    def get(self, session_id: str) -> PlanIR | None:
        return self.plans.get(session_id)

    def set(self, session_id: str, plan: PlanIR, persist: bool = True) -> None:
        self.plans[session_id] = plan


class _Task:
    def __init__(self) -> None:
        self.cancelled = False

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        self.cancelled = True


def _build_plan() -> PlanIR:
    return PlanIR(
        plan_id="plan_1",
        version=1,
        nodes=[
            NodeIR(
                node_id="n1",
                node_type=NodeType.LLM,
                config={},
                inputs={},
                position={"x": 0, "y": 0},
            ),
            NodeIR(
                node_id="n2",
                node_type=NodeType.LLM,
                config={},
                inputs={},
                position={"x": 1, "y": 0},
            ),
        ],
        edges=[],
        entry_node_id="n1",
        metadata={},
    )


def _build_execution(*, execution_id: str, plan_id: str, session_id: str) -> ExecutionIR:
    now = datetime.now()
    return ExecutionIR(
        execution_id=execution_id,
        plan_id=plan_id,
        status=ExecutionStatus.RUNNING,
        node_executions={
            "n1": NodeExecutionIR(
                node_id="n1",
                status=NodeStatus.COMPLETED,
                started_at=now,
                completed_at=now,
                inputs={},
                outputs={"result": "ok1"},
                error=None,
                streaming_output="ok1",
                metadata={},
            ),
            "n2": NodeExecutionIR(
                node_id="n2",
                status=NodeStatus.COMPLETED,
                started_at=now,
                completed_at=now,
                inputs={},
                outputs={"result": "ok2"},
                error=None,
                streaming_output="ok2",
                metadata={},
            ),
        },
        context={},
        started_at=now,
        completed_at=None,
        error=None,
        metadata={"session_id": session_id},
    )


def test_create_checkpoint_persists_and_returns_payload() -> None:
    checkpoint_store = _CheckpointStore()
    execution_store = _ExecutionStore()
    plan_store = _PlanStore()
    tasks: dict[str, Any] = {}
    llm_call_logs: dict[str, list[Any]] = {
        "exec_1": [
            LLMCallLog(
                call_id="llm_0_n1",
                node_id="n1",
                timestamp=datetime.now(),
                model="test-model",
                prompt="hello",
                response="world",
                metadata={},
            )
        ]
    }

    manager = CheckpointManager(
        checkpoint_store=checkpoint_store,
        execution_store=execution_store,
        plan_store=plan_store,
        execution_tasks=tasks,
        llm_call_logs=llm_call_logs,
        get_execution_order=lambda plan: [node.node_id for node in plan.nodes],
    )

    execution = _build_execution(execution_id="exec_1", plan_id="plan_1", session_id="s1")

    checkpoint, payload = asyncio.run(
        manager.create_checkpoint(
            session_id="s1",
            execution=execution,
            trigger=CheckpointTrigger.USER_CHECKPOINT,
            node_id="n1",
        )
    )

    assert checkpoint.checkpoint_id == "cp_1"
    assert checkpoint_store.get("exec_1")[0].checkpoint_id == "cp_1"
    assert payload.checkpoint_id == "cp_1"
    assert payload.execution_id == "exec_1"
    assert payload.session_id == "s1"


def test_restore_checkpoint_returns_failure_when_not_found() -> None:
    checkpoint_store = _CheckpointStore()
    execution_store = _ExecutionStore()
    plan_store = _PlanStore()

    manager = CheckpointManager(
        checkpoint_store=checkpoint_store,
        execution_store=execution_store,
        plan_store=plan_store,
        execution_tasks={},
        llm_call_logs={},
        get_execution_order=lambda plan: [],
    )

    outcome = asyncio.run(
        manager.restore_checkpoint(
            session_id="s1",
            checkpoint_id="cp_missing",
            replay_mode="deterministic",
            execution_id="exec_1",
        )
    )

    assert outcome.result.success is False
    assert outcome.result.message == "Checkpoint not found"
    assert outcome.execution is None


def test_restore_checkpoint_forks_execution_resets_nodes_and_copies_checkpoints() -> None:
    checkpoint_store = _CheckpointStore()
    execution_store = _ExecutionStore()
    plan_store = _PlanStore()

    plan = _build_plan()
    plan_store.set("s1", plan, persist=False)

    parent_execution = _build_execution(
        execution_id="exec_parent", plan_id=plan.plan_id, session_id="s1"
    )

    # Parent checkpoints list must exist.
    parent_cp = CheckpointIR(
        checkpoint_id="cp_1",
        execution_id=parent_execution.execution_id,
        plan_id=parent_execution.plan_id,
        sequence_number=1,
        trigger=CheckpointTrigger.NODE_COMPLETED,
        created_at=datetime.now(),
        execution_snapshot=parent_execution.model_dump(),
        llm_call_logs=[
            LLMCallLog(
                call_id="llm_0_n1",
                node_id="n1",
                timestamp=datetime.now(),
                model="test-model",
                prompt="hello",
                response="world",
                metadata={},
            )
        ],
        parent_checkpoint_id=None,
        delta=None,
        metadata={"trigger_node_id": "n1"},
    )
    checkpoint_store.add(parent_execution.execution_id, parent_cp)

    task = _Task()
    execution_tasks = {parent_execution.execution_id: task}
    llm_call_logs: dict[str, list[Any]] = {}

    manager = CheckpointManager(
        checkpoint_store=checkpoint_store,
        execution_store=execution_store,
        plan_store=plan_store,
        execution_tasks=execution_tasks,
        llm_call_logs=llm_call_logs,
        get_execution_order=lambda _plan: ["n1", "n2"],
    )

    outcome = asyncio.run(
        manager.restore_checkpoint(
            session_id="s1",
            checkpoint_id="cp_1",
            replay_mode="deterministic",
            execution_id=parent_execution.execution_id,
        )
    )

    assert outcome.result.success is True
    assert outcome.execution is not None
    forked = outcome.execution
    assert forked.execution_id != parent_execution.execution_id

    # Task cancelled.
    assert task.cancelled is True

    # Metadata updated.
    assert forked.metadata.get("session_id") == "s1"
    assert forked.metadata.get("parent_execution_id") == parent_execution.execution_id
    assert forked.metadata.get("parent_checkpoint_id") == "cp_1"
    assert forked.metadata.get("replay_mode") == "deterministic"

    # Status paused.
    assert forked.status == ExecutionStatus.PAUSED

    # Only nodes AFTER the trigger node should be reset (n2).
    # The checkpoint captures state after trigger_node (n1) completed,
    # so n1 stays COMPLETED and execution resumes from n2.
    assert forked.node_executions["n1"].status == NodeStatus.COMPLETED
    assert forked.node_executions["n2"].status == NodeStatus.PENDING
    assert forked.node_executions["n2"].outputs == {}
    assert forked.node_executions["n2"].error is None

    # Execution persisted.
    assert forked.execution_id in execution_store.saved

    # Checkpoints copied onto forked execution id.
    assert checkpoint_store.get(forked.execution_id)
    assert checkpoint_store.get(forked.execution_id)[0].checkpoint_id == "cp_1"

    # LLM call logs copied for deterministic replay.
    assert llm_call_logs[forked.execution_id]
    assert getattr(llm_call_logs[forked.execution_id][0], "call_id", None) == "llm_0_n1"


def test_restore_terminal_checkpoint_replays_all_nodes() -> None:
    """Restoring from the last checkpoint (trigger_node is the last node)
    should reset ALL nodes (replay-all semantics) instead of failing."""
    checkpoint_store = _CheckpointStore()
    execution_store = _ExecutionStore()
    plan_store = _PlanStore()

    plan = _build_plan()
    plan_store.set("s1", plan, persist=False)

    parent_execution = _build_execution(
        execution_id="exec_parent", plan_id=plan.plan_id, session_id="s1"
    )

    # Checkpoint triggered by the LAST node (n2)
    terminal_cp = CheckpointIR(
        checkpoint_id="cp_terminal",
        execution_id=parent_execution.execution_id,
        plan_id=parent_execution.plan_id,
        sequence_number=2,
        trigger=CheckpointTrigger.NODE_COMPLETED,
        created_at=datetime.now(),
        execution_snapshot=parent_execution.model_dump(),
        llm_call_logs=[],
        parent_checkpoint_id=None,
        delta=None,
        metadata={"trigger_node_id": "n2"},
    )
    checkpoint_store.add(parent_execution.execution_id, terminal_cp)

    task = _Task()
    execution_tasks = {parent_execution.execution_id: task}
    llm_call_logs: dict[str, list[Any]] = {}

    manager = CheckpointManager(
        checkpoint_store=checkpoint_store,
        execution_store=execution_store,
        plan_store=plan_store,
        execution_tasks=execution_tasks,
        llm_call_logs=llm_call_logs,
        get_execution_order=lambda _plan: ["n1", "n2"],
    )

    outcome = asyncio.run(
        manager.restore_checkpoint(
            session_id="s1",
            checkpoint_id="cp_terminal",
            replay_mode="deterministic",
            execution_id=parent_execution.execution_id,
        )
    )

    # Should succeed (not reject)
    assert outcome.result.success is True
    assert outcome.execution is not None
    forked = outcome.execution

    # ALL nodes should be reset (replay-all)
    assert forked.node_executions["n1"].status == NodeStatus.PENDING
    assert forked.node_executions["n2"].status == NodeStatus.PENDING

    # Metadata should indicate replay-all
    assert forked.metadata.get("replay_all") is True
