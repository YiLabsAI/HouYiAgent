import asyncio

import pytest

from houyi.application.workflow.execution_lifecycle_service import ExecutionLifecycleManager
from houyi.application.workflow.execution_order_service import get_execution_order
from houyi.application.workflow.orchestration.plan import NodeType
from houyi.application.workflow.plan_execution_loop import PlanExecutionLoop
from houyi.interface.protocol.ir import EdgeIR, ExecutionIR, ExecutionStatus, NodeIR, PlanIR


def _make_plan(*, node_ids: list[str], edges: list[tuple[str, str]], entry_node_id: str) -> PlanIR:
    nodes = [NodeIR(node_id=node_id, node_type=NodeType.TOOL) for node_id in node_ids]
    edge_irs = [
        EdgeIR(edge_id=f"e_{src}_{dst}", source_node_id=src, target_node_id=dst)
        for src, dst in edges
    ]
    return PlanIR(
        plan_id="plan_1",
        version=1,
        nodes=nodes,
        edges=edge_irs,
        entry_node_id=entry_node_id,
        metadata={},
    )


def test_get_execution_order_prefers_entry_node_and_is_deterministic() -> None:
    plan = _make_plan(
        node_ids=["b", "a", "c"],
        edges=[("a", "b"), ("a", "c")],
        entry_node_id="a",
    )

    order = get_execution_order(plan)

    assert order[0] == "a"
    assert set(order) == {"a", "b", "c"}
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")


@pytest.mark.asyncio
async def test_plan_execution_loop_completes_in_order_and_invokes_checkpoint_callback() -> None:
    plan = _make_plan(
        node_ids=["n1", "n2", "n3"],
        edges=[("n1", "n2"), ("n2", "n3")],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.RUNNING,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []
    checkpoint_nodes: list[str] = []
    lifecycle_events: list[str] = []

    def plan_getter(session_id: str, base_plan: PlanIR) -> PlanIR:
        assert session_id == "s1"
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)

    async def checkpoint_callback(
        _session_id: str, _execution: ExecutionIR, _trigger: object, *, node_id=None
    ):
        checkpoint_nodes.append(node_id)

    async def notify_lifecycle(event_name: str, _context: object) -> None:
        lifecycle_events.append(event_name)

    def context_factory(_session_id: str, _execution: ExecutionIR, _plan: PlanIR) -> object:
        return object()

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
    )

    await loop.execute("s1", execution, plan)

    assert executed == ["n1", "n2", "n3"]
    assert checkpoint_nodes == ["n1", "n2", "n3"]
    assert lifecycle_events == ["on_execution_start", "on_execution_end"]
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.completed_at is not None


@pytest.mark.asyncio
async def test_plan_execution_loop_handles_pause_then_resume() -> None:
    plan = _make_plan(
        node_ids=["n1"],
        edges=[],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.PAUSED,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []

    def plan_getter(_session_id: str, base_plan: PlanIR) -> PlanIR:
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)

    async def checkpoint_callback(*_args, **_kwargs):
        return None

    async def notify_lifecycle(*_args, **_kwargs):
        return None

    def context_factory(*_args, **_kwargs):
        return object()

    sleep_calls: list[float] = []

    async def sleep_func(seconds: float) -> None:
        sleep_calls.append(seconds)
        execution.status = ExecutionStatus.RUNNING
        await asyncio.sleep(0)

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
        sleep_func=sleep_func,
    )

    await loop.execute("s1", execution, plan)

    assert sleep_calls
    assert executed == ["n1"]
    assert execution.status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_plan_execution_loop_branching_fork_join_orders_join_after_branches() -> None:
    plan = _make_plan(
        node_ids=["n1", "n2", "n3", "n4"],
        edges=[("n1", "n2"), ("n1", "n3"), ("n2", "n4"), ("n3", "n4")],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.RUNNING,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []

    def plan_getter(_session_id: str, base_plan: PlanIR) -> PlanIR:
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)

    async def checkpoint_callback(*_args, **_kwargs):
        return None

    async def notify_lifecycle(*_args, **_kwargs):
        return None

    def context_factory(*_args, **_kwargs):
        return object()

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
    )

    await loop.execute("s1", execution, plan)

    assert executed[0] == "n1"
    assert executed[-1] == "n4"
    assert execution.status == ExecutionStatus.COMPLETED

    idx_n2 = executed.index("n2")
    idx_n3 = executed.index("n3")
    idx_n4 = executed.index("n4")
    assert idx_n2 < idx_n4
    assert idx_n3 < idx_n4


@pytest.mark.asyncio
async def test_plan_execution_loop_failure_sets_failed_and_still_notifies_end() -> None:
    plan = _make_plan(
        node_ids=["n1", "n2"],
        edges=[("n1", "n2")],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.RUNNING,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []
    checkpoint_nodes: list[str] = []
    lifecycle_events: list[str] = []

    def plan_getter(_session_id: str, base_plan: PlanIR) -> PlanIR:
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)
        if node_id == "n2":
            raise RuntimeError("boom")

    async def checkpoint_callback(
        _session_id: str, _execution: ExecutionIR, _trigger: object, *, node_id=None
    ):
        checkpoint_nodes.append(node_id)

    async def notify_lifecycle(event_name: str, _context: object) -> None:
        lifecycle_events.append(event_name)

    def context_factory(*_args, **_kwargs):
        return object()

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await loop.execute("s1", execution, plan)

    assert executed == ["n1", "n2"]
    assert checkpoint_nodes == ["n1"]
    assert lifecycle_events == ["on_execution_start", "on_execution_end"]
    assert execution.status == ExecutionStatus.FAILED
    assert execution.error is not None
    assert execution.completed_at is not None


@pytest.mark.asyncio
async def test_stops_without_marking_completed_when_aborted() -> None:
    plan = _make_plan(
        node_ids=["n1", "n2", "n3"],
        edges=[("n1", "n2"), ("n2", "n3")],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.RUNNING,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []
    lifecycle_events: list[str] = []

    def plan_getter(_session_id: str, base_plan: PlanIR) -> PlanIR:
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)
        if node_id == "n2":
            execution.status = ExecutionStatus.ABORTED

    async def checkpoint_callback(*_args, **_kwargs):
        return None

    async def notify_lifecycle(event_name: str, _context: object) -> None:
        lifecycle_events.append(event_name)

    def context_factory(*_args, **_kwargs):
        return object()

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
    )

    await loop.execute("s1", execution, plan)

    assert executed == ["n1", "n2"]
    assert lifecycle_events == ["on_execution_start", "on_execution_end"]
    assert execution.status == ExecutionStatus.ABORTED
    assert execution.completed_at is None


@pytest.mark.asyncio
async def test_exits_without_executing_when_aborted_while_paused() -> None:
    plan = _make_plan(
        node_ids=["n1"],
        edges=[],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.PAUSED,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []

    def plan_getter(_session_id: str, base_plan: PlanIR) -> PlanIR:
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)

    async def checkpoint_callback(*_args, **_kwargs):
        return None

    async def notify_lifecycle(*_args, **_kwargs):
        return None

    def context_factory(*_args, **_kwargs):
        return object()

    async def sleep_func(_seconds: float) -> None:
        execution.status = ExecutionStatus.ABORTED
        await asyncio.sleep(0)

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
        sleep_func=sleep_func,
    )

    await loop.execute("s1", execution, plan)

    assert executed == []
    assert execution.status == ExecutionStatus.ABORTED


@pytest.mark.asyncio
async def test_treats_skipped_nodes_as_executed() -> None:
    plan = _make_plan(
        node_ids=["n1", "n2"],
        edges=[("n1", "n2")],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.RUNNING,
        node_executions={
            "n1": {
                "node_id": "n1",
                "status": "skipped",
            }
        },
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []

    def plan_getter(_session_id: str, base_plan: PlanIR) -> PlanIR:
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)

    async def checkpoint_callback(*_args, **_kwargs):
        return None

    async def notify_lifecycle(*_args, **_kwargs):
        return None

    def context_factory(*_args, **_kwargs):
        return object()

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
    )

    await loop.execute("s1", execution, plan)

    assert executed == ["n2"]
    assert execution.status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_does_not_execute_or_mark_completed_when_aborted_before_start() -> None:
    plan = _make_plan(
        node_ids=["n1"],
        edges=[],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.ABORTED,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []
    lifecycle_events: list[str] = []
    checkpoint_nodes: list[str | None] = []

    def plan_getter(_session_id: str, base_plan: PlanIR) -> PlanIR:
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)

    async def checkpoint_callback(
        _session_id: str, _execution: ExecutionIR, _trigger: object, *, node_id=None
    ):
        checkpoint_nodes.append(node_id)

    async def notify_lifecycle(event_name: str, _context: object) -> None:
        lifecycle_events.append(event_name)

    def context_factory(*_args, **_kwargs):
        return object()

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
    )

    await loop.execute("s1", execution, plan)

    assert executed == []
    assert checkpoint_nodes == []
    assert lifecycle_events == ["on_execution_start", "on_execution_end"]
    assert execution.status == ExecutionStatus.ABORTED
    assert execution.completed_at is None
    assert execution.error is None


@pytest.mark.asyncio
async def test_converges_to_aborted_without_completed_at_on_cancelled_error() -> None:
    plan = _make_plan(
        node_ids=["n1"],
        edges=[],
        entry_node_id="n1",
    )
    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=plan.plan_id,
        status=ExecutionStatus.RUNNING,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []
    lifecycle_events: list[str] = []

    def plan_getter(_session_id: str, base_plan: PlanIR) -> PlanIR:
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)
        raise asyncio.CancelledError()

    async def checkpoint_callback(*_args, **_kwargs):
        return None

    async def notify_lifecycle(event_name: str, _context: object) -> None:
        lifecycle_events.append(event_name)

    def context_factory(*_args, **_kwargs):
        return object()

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
    )

    await loop.execute("s1", execution, plan)

    assert executed == ["n1"]
    assert lifecycle_events == ["on_execution_start", "on_execution_end"]
    assert execution.status == ExecutionStatus.ABORTED
    assert execution.completed_at is None
    assert execution.error is None


@pytest.mark.asyncio
async def test_executes_newly_added_node_after_dynamic_plan_update() -> None:
    base_plan = _make_plan(
        node_ids=["n1", "n2"],
        edges=[("n1", "n2")],
        entry_node_id="n1",
    )
    updated_plan = _make_plan(
        node_ids=["n1", "n2", "n3"],
        edges=[("n1", "n2"), ("n2", "n3")],
        entry_node_id="n1",
    )

    execution = ExecutionIR(
        execution_id="exec_1",
        plan_id=base_plan.plan_id,
        status=ExecutionStatus.RUNNING,
        node_executions={},
        context={},
        started_at=None,
        completed_at=None,
        error=None,
        metadata={"session_id": "s1"},
    )

    executed: list[str] = []
    plans_returned: list[int] = []

    def plan_getter(_session_id: str, _base_plan: PlanIR) -> PlanIR:
        plans_returned.append(1)
        if executed and executed[-1] == "n2":
            return updated_plan
        return base_plan

    def order_resolver(current_plan: PlanIR) -> list[str]:
        return get_execution_order(current_plan)

    async def node_executor(_context: object, node_id: str) -> None:
        executed.append(node_id)

    async def checkpoint_callback(*_args, **_kwargs):
        return None

    async def notify_lifecycle(*_args, **_kwargs):
        return None

    def context_factory(*_args, **_kwargs):
        return object()

    loop = PlanExecutionLoop(
        plan_getter=plan_getter,
        get_execution_order=order_resolver,
        node_executor=node_executor,
        checkpoint_callback=checkpoint_callback,
        notify_lifecycle=notify_lifecycle,
        context_factory=context_factory,
    )

    await loop.execute("s1", execution, base_plan)

    assert executed == ["n1", "n2", "n3"]
    assert execution.status == ExecutionStatus.COMPLETED
    assert plans_returned


def test_orders_cycle_and_disconnected_nodes_deterministically() -> None:
    plan = _make_plan(
        node_ids=["a", "b", "c", "d"],
        edges=[("a", "b"), ("b", "a")],
        entry_node_id="a",
    )

    order = get_execution_order(plan)

    assert order == ["c", "d", "a", "b"]


class _InMemoryExecutionStore:
    def __init__(self) -> None:
        self._executions: dict[str, ExecutionIR] = {}

    def get(self, execution_id: str) -> ExecutionIR | None:
        return self._executions.get(execution_id)

    def save(self, execution: ExecutionIR) -> None:
        self._executions[execution.execution_id] = execution


class _InMemoryCheckpointStore:
    def __init__(self) -> None:
        self.inited: list[str] = []

    def init_execution(self, execution_id: str) -> None:
        self.inited.append(execution_id)


class _InMemoryPlanStore:
    def __init__(self) -> None:
        self._plans: dict[str, PlanIR] = {}

    def set(self, session_id: str, plan: PlanIR, persist: bool = True) -> None:
        self._plans[session_id] = plan

    def get_cached(self, session_id: str) -> PlanIR | None:
        return self._plans.get(session_id)


class _DummyTask:
    def __init__(self, *, done: bool = False) -> None:
        self._done = done
        self.cancel_called = False

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancel_called = True


class _DummyBackend:
    def __init__(self) -> None:
        self.starts: list[tuple[str, str]] = []

    async def start(self, session_id: str, execution: ExecutionIR, plan: PlanIR, execute_plan):
        self.starts.append((session_id, execution.execution_id))
        return _DummyTask()


@pytest.mark.asyncio
async def test_execution_lifecycle_start_pause_resume_abort() -> None:
    plan = _make_plan(
        node_ids=["n1"],
        edges=[],
        entry_node_id="n1",
    )

    execution_store = _InMemoryExecutionStore()
    checkpoint_store = _InMemoryCheckpointStore()
    plan_store = _InMemoryPlanStore()
    backend = _DummyBackend()

    tasks: dict[str, object] = {}
    llm_call_logs: dict[str, list[object]] = {}

    def normalize_run_settings(run_settings):
        return run_settings or {}

    async def execute_plan(*_args, **_kwargs):
        return None

    manager = ExecutionLifecycleManager(
        execution_store=execution_store,
        checkpoint_store=checkpoint_store,
        plan_store=plan_store,
        execution_backend=backend,
        execution_tasks=tasks,
        llm_call_logs=llm_call_logs,
        normalize_run_settings=normalize_run_settings,
        execute_plan=execute_plan,
    )

    outcome = await manager.start_execution(session_id="s1", plan=plan, run_settings={"x": 1})
    execution_id = outcome.execution.execution_id

    assert outcome.execution.status == ExecutionStatus.RUNNING
    assert outcome.status_event.status == ExecutionStatus.RUNNING
    assert execution_store.get(execution_id) is not None
    assert checkpoint_store.inited == [execution_id]
    assert plan_store.get_cached("s1") is not None
    assert backend.starts == [("s1", execution_id)]

    paused = await manager.pause_execution(execution_id=execution_id)
    assert paused.status_event is not None
    assert paused.status_event.status == ExecutionStatus.PAUSED

    resumed = await manager.resume_execution(execution_id=execution_id)
    assert resumed.status_event is not None
    assert resumed.status_event.status == ExecutionStatus.RUNNING

    aborted = await manager.abort_execution(execution_id=execution_id)
    assert aborted.status_event is not None
    assert aborted.status_event.status == ExecutionStatus.ABORTED
