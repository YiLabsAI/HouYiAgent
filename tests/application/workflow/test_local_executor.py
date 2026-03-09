"""Tests for execution/local_executor.py"""

import pytest
from pydantic import BaseModel

from houyi.application.workflow.executor import ExecutionResult, LocalExecutor
from houyi.application.workflow.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.application.workflow.orchestration.state import SessionState, TaskStatus
from houyi.domain.skill.spec import SkillSpec


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


@pytest.mark.asyncio
async def test_local_executor_result_exposes_compat_and_structured_fields():
    """Test executor result supports legacy and structured fields together."""
    executor = LocalExecutor()

    node = IRNode(
        node_id="node1",
        node_type=NodeType.LLM,
        inputs={"prompt": "test"},
        outputs={"result": "$answer"},
        metadata={},
    )

    plan = ExecutionPlan(plan_id="test_plan_structured", nodes=[node], entry_node="node1")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert result.success is True
    assert result.status == TaskStatus.SUCCEEDED
    assert result.task_id.startswith("task_")
    assert result.trace_id.startswith("trace_")
    assert result.metrics.total_duration_ms >= 0
    assert result.metrics.node_durations["node1"] >= 0
    assert result.metadata["nodes_executed"] == 1
    assert result.error is None


@pytest.mark.asyncio
async def test_local_executor_tool_node_failure_returns_structured_error():
    """Test executor returns structured failure details for tool execution errors."""
    executor = LocalExecutor()

    class Input(BaseModel):
        task: str

    class Output(BaseModel):
        result: str

    def failing_skill(task: str):
        raise ValueError(f"Intentional error: {task}")

    skill = SkillSpec(
        name="failing_skill",
        description="A failing skill",
        input_schema=Input,
        output_schema=Output,
        executor=failing_skill,
    )

    node = IRNode(
        node_id="fail_node",
        node_type=NodeType.TOOL,
        skill_ref=skill,
        inputs={"task": "boom"},
        outputs={"result": "$answer"},
        metadata={"direct_execution": True},
    )

    plan = ExecutionPlan(plan_id="test_plan_failure", nodes=[node], entry_node="fail_node")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert result.success is False
    assert result.status == TaskStatus.FAILED
    assert result.error is not None
    assert "Intentional error" in result.error


@pytest.mark.asyncio
async def test_local_executor_tool_node_direct_execution_tracks_metrics():
    """Test direct tool execution keeps legacy context behavior and structured metrics."""
    executor = LocalExecutor()

    class EchoInput(BaseModel):
        task: str

    class EchoOutput(BaseModel):
        result: str

    def echo_skill(task: str):
        return {"result": f"echo:{task}"}

    skill = SkillSpec(
        name="echo",
        description="Echo a task",
        input_schema=EchoInput,
        output_schema=EchoOutput,
        executor=echo_skill,
    )

    node = IRNode(
        node_id="tool_node",
        node_type=NodeType.TOOL,
        skill_ref=skill,
        inputs={"task": "hello"},
        outputs={"result": "$answer"},
        metadata={"direct_execution": True},
    )

    plan = ExecutionPlan(plan_id="test_plan_tool_metrics", nodes=[node], entry_node="tool_node")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert result.success is True
    assert result.status == TaskStatus.SUCCEEDED
    assert result.metrics.node_durations["tool_node"] >= 0
    assert result.metadata["nodes_executed"] == 1
    assert result.output["result"] == "echo:hello"


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
