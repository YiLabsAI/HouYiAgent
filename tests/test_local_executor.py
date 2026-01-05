"""Tests for execution/local_executor.py"""

import pytest

from houyi.execution.local_executor import ExecutionResult, LocalExecutor
from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.state import SessionState


@pytest.mark.asyncio
async def test_local_executor_basic():
    """Test basic LocalExecutor execution."""
    executor = LocalExecutor()

    # Create a simple plan with one LLM node (doesn't require skill_ref)
    node = IRNode(
        node_id="node1",
        node_type=NodeType.LLM,
        inputs={"prompt": "test"},
        outputs={"result": "$output"},
        metadata={"model": "test"},
    )

    plan = ExecutionPlan(plan_id="test_plan_1", nodes=[node], entry_node="node1")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.metadata["nodes_executed"] == 1


@pytest.mark.asyncio
async def test_local_executor_dag():
    """Test DAG execution with dependencies."""
    executor = LocalExecutor()

    # Create a DAG: node1 -> node2 -> node3
    node1 = IRNode(
        node_id="node1",
        node_type=NodeType.LLM,
        inputs={"prompt": "step1"},
        outputs={"result": "$step1"},
        metadata={},
    )

    node2 = IRNode(
        node_id="node2",
        node_type=NodeType.LLM,
        inputs={"prompt": "$step1"},
        outputs={"result": "$step2"},
        metadata={},
        dependencies=["node1"],
    )

    node3 = IRNode(
        node_id="node3",
        node_type=NodeType.LLM,
        inputs={"prompt": "$step2"},
        outputs={"result": "$answer"},
        metadata={},
        dependencies=["node2"],
    )

    plan = ExecutionPlan(plan_id="test_plan_dag", nodes=[node1, node2, node3], entry_node="node1")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert result.success is True
    assert result.metadata["nodes_executed"] == 3


@pytest.mark.asyncio
async def test_local_executor_parallel():
    """Test parallel execution of independent nodes."""
    executor = LocalExecutor()

    # Create parallel nodes (no dependencies)
    node1 = IRNode(
        node_id="node1",
        node_type=NodeType.LLM,
        inputs={"prompt": "task1"},
        outputs={"result": "$out1"},
        metadata={},
    )

    node2 = IRNode(
        node_id="node2",
        node_type=NodeType.LLM,
        inputs={"prompt": "task2"},
        outputs={"result": "$out2"},
        metadata={},
    )

    node3 = IRNode(
        node_id="node3",
        node_type=NodeType.LLM,
        inputs={"prompt": "task3"},
        outputs={"result": "$out3"},
        metadata={},
    )

    plan = ExecutionPlan(
        plan_id="test_plan_parallel", nodes=[node1, node2, node3], entry_node="node1"
    )
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert result.success is True
    assert result.metadata["nodes_executed"] == 3


@pytest.mark.asyncio
async def test_local_executor_circular_dependency():
    """Test detection of circular dependencies."""
    executor = LocalExecutor()

    # Create circular dependency: node1 -> node2 -> node1
    node1 = IRNode(
        node_id="node1",
        node_type=NodeType.LLM,
        inputs={},
        outputs={},
        metadata={},
        dependencies=["node2"],
    )

    node2 = IRNode(
        node_id="node2",
        node_type=NodeType.LLM,
        inputs={},
        outputs={},
        metadata={},
        dependencies=["node1"],
    )

    plan = ExecutionPlan(plan_id="test_plan_circular", nodes=[node1, node2], entry_node="node1")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    with pytest.raises(RuntimeError, match="Circular dependency"):
        await executor.execute(plan, state)


@pytest.mark.asyncio
async def test_local_executor_context_propagation():
    """Test context propagation between nodes."""
    executor = LocalExecutor()

    node1 = IRNode(
        node_id="node1",
        node_type=NodeType.LLM,
        inputs={"prompt": "step1"},
        outputs={"result": "$intermediate"},
        metadata={},
    )

    node2 = IRNode(
        node_id="node2",
        node_type=NodeType.LLM,
        inputs={"prompt": "$intermediate"},
        outputs={"result": "$answer"},
        metadata={},
        dependencies=["node1"],
    )

    plan = ExecutionPlan(plan_id="test_plan_context", nodes=[node1, node2], entry_node="node1")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert result.success is True
    assert "intermediate" in result.metadata["context"]


@pytest.mark.asyncio
async def test_local_executor_empty_plan():
    """Test execution with empty plan."""
    executor = LocalExecutor()

    plan = ExecutionPlan(plan_id="test_plan_empty", nodes=[], entry_node="")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert result.success is True
    assert result.metadata["nodes_executed"] == 0


def test_execution_result():
    """Test ExecutionResult class."""
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = ExecutionResult(
        success=True, output="test output", final_state=state, metadata={"key": "value"}
    )

    assert result.success is True
    assert result.output == "test output"
    assert result.final_state == state
    assert result.metadata["key"] == "value"


def test_execution_result_default_metadata():
    """Test ExecutionResult with default metadata."""
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = ExecutionResult(success=False, output=None, final_state=state)

    assert result.success is False
    assert result.output is None
    assert result.metadata == {}
