"""Tests for execution/local_executor.py"""

import sys
import types

import pytest
from pydantic import BaseModel

from houyi.application.workflow.executor import ExecutionMetrics, ExecutionResult, LocalExecutor
from houyi.application.workflow.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.application.workflow.orchestration.state import SessionState, TaskStatus
from houyi.assurance.verification.verifier import VerificationRule
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
async def test_circular_dependency():
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
async def test_context_propagation():
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
async def test_empty_plan():
    """Test execution with empty plan."""
    executor = LocalExecutor()

    plan = ExecutionPlan(plan_id="test_plan_empty", nodes=[], entry_node="")
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = await executor.execute(plan, state)

    assert result.success is True
    assert result.metadata["nodes_executed"] == 0


@pytest.mark.asyncio
async def test_exposes_compat_structured():
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
async def test_tool_node_failure():
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
async def test_tool_node_direct_execution():
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


def test_default_metadata():
    """Test ExecutionResult with default metadata."""
    state = SessionState(session_id="test_session", agent_id="test_agent")

    result = ExecutionResult(success=False, output=None, final_state=state)

    assert result.success is False
    assert result.output is None
    assert result.metadata == {}


@pytest.mark.asyncio
async def test_trace_path():
    class _Span:
        def __init__(self) -> None:
            self.attrs = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def set_attribute(self, key, value):
            self.attrs[key] = value

    class _Trace:
        def __init__(self) -> None:
            self.span = _Span()

        def start_span(self, *args, **kwargs):
            _ = (args, kwargs)
            return self.span

    executor = LocalExecutor(trace_manager=_Trace())
    node = IRNode(node_id="n1", node_type=NodeType.LLM, inputs={"prompt": "x"}, outputs={})

    result = await executor._execute_node(node, {}, ExecutionMetrics())

    assert "answer" in result


@pytest.mark.asyncio
async def test_llm_fallback(monkeypatch):
    module = types.ModuleType("houyi.adapters.llm.openai_adapter")

    class _OpenAIAdapter:
        def __init__(self):
            raise RuntimeError("missing")

    module.OpenAIAdapter = _OpenAIAdapter
    monkeypatch.setitem(sys.modules, "houyi.adapters.llm.openai_adapter", module)

    base_module = types.ModuleType("houyi.adapters.llm.base")

    class _LLMMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class _MessageRole:
        USER = "user"

    base_module.LLMMessage = _LLMMessage
    base_module.MessageRole = _MessageRole
    monkeypatch.setitem(sys.modules, "houyi.adapters.llm.base", base_module)

    executor = LocalExecutor()
    node = IRNode(
        node_id="n1",
        node_type=NodeType.LLM,
        inputs={"task": "hello"},
        outputs={},
        metadata={"use_real_llm": True},
    )

    result = await executor._execute_llm_node(node, {"task": "hello"})

    assert "Mock LLM response" in result["answer"]


@pytest.mark.asyncio
async def test_llm_success(monkeypatch):
    module = types.ModuleType("houyi.adapters.llm.openai_adapter")

    class _OpenAIAdapter:
        async def chat(self, messages):
            _ = messages
            return types.SimpleNamespace(content="real")

    module.OpenAIAdapter = _OpenAIAdapter
    monkeypatch.setitem(sys.modules, "houyi.adapters.llm.openai_adapter", module)

    base_module = types.ModuleType("houyi.adapters.llm.base")

    class _LLMMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class _MessageRole:
        USER = "user"

    base_module.LLMMessage = _LLMMessage
    base_module.MessageRole = _MessageRole
    monkeypatch.setitem(sys.modules, "houyi.adapters.llm.base", base_module)

    executor = LocalExecutor()
    node = IRNode(
        node_id="n1", node_type=NodeType.LLM, inputs={}, outputs={}, metadata={"use_real_llm": True}
    )

    result = await executor._execute_llm_node(node, {"task": "hello"})

    assert result == {"answer": "real"}


def test_extract_params_schema():
    class Input(BaseModel):
        query: str

    class Output(BaseModel):
        result: str

    skill = SkillSpec(
        name="search",
        description="Search",
        input_schema=Input,
        output_schema=Output,
        executor=lambda query: query,
    )
    executor = LocalExecutor()

    params = executor._extract_params_from_task("find docs", skill)

    assert params == {"query": "find docs"}


def test_extract_params_func():
    def _tool(values: list[str]):
        return values

    class Input(BaseModel):
        pass

    class Output(BaseModel):
        result: list[str]

    skill = SkillSpec(
        name="collect",
        description="Collect",
        input_schema=Input,
        output_schema=Output,
        executor=_tool,
    )
    skill._original_func = _tool  # type: ignore[attr-defined]
    executor = LocalExecutor()

    params = executor._extract_params_from_task("one", skill)

    assert params == {"values": ["one"]}


@pytest.mark.asyncio
async def test_tool_requires_skill():
    executor = LocalExecutor()
    node = IRNode(node_id="n1", node_type=NodeType.TOOL, inputs={}, outputs={})

    with pytest.raises(ValueError, match="has no skill_ref"):
        await executor._execute_tool_node(node, {})


@pytest.mark.asyncio
async def test_nested_params():
    class Input(BaseModel):
        task: str

    class Output(BaseModel):
        result: str

    skill = SkillSpec(
        name="echo",
        description="Echo",
        input_schema=Input,
        output_schema=Output,
        executor=lambda task: {"result": task},
    )
    node = IRNode(
        node_id="n1",
        node_type=NodeType.TOOL,
        skill_ref=skill,
        inputs={},
        outputs={},
        metadata={"direct_execution": True},
    )
    executor = LocalExecutor()

    result = await executor._execute_tool_node(node, {"params": {"task": 123}})

    assert result == {"result": {"result": "123"}}


@pytest.mark.asyncio
async def test_tool_params():
    class Input(BaseModel):
        task: str

    class Output(BaseModel):
        result: str

    skill = SkillSpec(
        name="echo",
        description="Echo",
        input_schema=Input,
        output_schema=Output,
        executor=lambda task: {"result": task},
    )
    node = IRNode(
        node_id="n1",
        node_type=NodeType.TOOL,
        skill_ref=skill,
        inputs={},
        outputs={},
        metadata={},
    )
    executor = LocalExecutor()

    result = await executor._execute_tool_node(node, {"params": {"task": "hi"}})

    assert result == {"result": {"result": "hi"}}


@pytest.mark.asyncio
async def test_tool_placeholder():
    class Input(BaseModel):
        task: str

    class Output(BaseModel):
        result: str

    skill = SkillSpec(
        name="echo",
        description="Echo",
        input_schema=Input,
        output_schema=Output,
        executor=None,
    )
    node = IRNode(node_id="n1", node_type=NodeType.TOOL, skill_ref=skill, inputs={}, outputs={})
    executor = LocalExecutor()

    result = await executor._execute_tool_node(node, {})

    assert result == {"result": "Result from echo"}


@pytest.mark.asyncio
async def test_verify_no_rules():
    executor = LocalExecutor()
    node = IRNode(node_id="n1", node_type=NodeType.VERIFY, inputs={}, outputs={})

    result = await executor._execute_verify_node(node, {})

    assert result == {"verified": True}


@pytest.mark.asyncio
async def test_verify_no_output():
    executor = LocalExecutor()
    node = IRNode(
        node_id="n1",
        node_type=NodeType.VERIFY,
        verification_rules=[VerificationRule(rule_id="r1", verifier_type="python")],
        inputs={},
        outputs={},
    )

    result = await executor._execute_verify_node(node, {})

    assert result == {"verified": False, "error": "No output to verify"}


@pytest.mark.asyncio
async def test_verify_collects_errors(monkeypatch):
    verify_module = types.ModuleType("houyi.assurance.verification")

    class _Verifier:
        async def verify(self, output, rule):
            _ = output
            return types.SimpleNamespace(
                passed=False,
                rule_id=rule.rule_id,
                error_type="assertion",
                error_message="failed",
            )

    verify_module.ConstraintChecker = _Verifier
    verify_module.PythonVerifier = _Verifier
    verify_module.SQLVerifier = _Verifier
    monkeypatch.setitem(sys.modules, "houyi.assurance.verification", verify_module)

    executor = LocalExecutor()
    node = IRNode(
        node_id="n1",
        node_type=NodeType.VERIFY,
        verification_rules=[VerificationRule(rule_id="r1", verifier_type="python")],
        inputs={},
        outputs={},
    )

    result = await executor._execute_verify_node(node, {"output": "bad"})

    assert result["verified"] is False
    assert result["errors"][0]["rule_id"] == "r1"


@pytest.mark.asyncio
async def test_verify_sql(monkeypatch):
    verify_module = types.ModuleType("houyi.assurance.verification")

    class _Verifier:
        async def verify(self, output, rule):
            _ = (output, rule)
            return types.SimpleNamespace(
                passed=True, rule_id="sql", error_type=None, error_message=None
            )

    verify_module.ConstraintChecker = _Verifier
    verify_module.PythonVerifier = _Verifier
    verify_module.SQLVerifier = _Verifier
    monkeypatch.setitem(sys.modules, "houyi.assurance.verification", verify_module)

    executor = LocalExecutor()
    node = IRNode(
        node_id="n1",
        node_type=NodeType.VERIFY,
        verification_rules=[VerificationRule(rule_id="sql", verifier_type="sql")],
        inputs={},
        outputs={},
    )

    result = await executor._execute_verify_node(node, {"output": "select 1"})

    assert result == {"verified": True, "errors": None}


@pytest.mark.asyncio
async def test_verify_constraint(monkeypatch):
    verify_module = types.ModuleType("houyi.assurance.verification")

    class _Verifier:
        async def verify(self, output, rule):
            _ = (output, rule)
            return types.SimpleNamespace(
                passed=False,
                rule_id="c1",
                error_type="constraint",
                error_message="bad",
            )

    verify_module.ConstraintChecker = _Verifier
    verify_module.PythonVerifier = _Verifier
    verify_module.SQLVerifier = _Verifier
    monkeypatch.setitem(sys.modules, "houyi.assurance.verification", verify_module)

    executor = LocalExecutor()
    node = IRNode(
        node_id="n1",
        node_type=NodeType.VERIFY,
        verification_rules=[VerificationRule(rule_id="c1", verifier_type="constraint")],
        inputs={},
        outputs={},
    )

    result = await executor._execute_verify_node(node, {"output": "bad"})

    assert result["verified"] is False
    assert result["errors"][0]["error_type"] == "constraint"


@pytest.mark.asyncio
async def test_node_type_error():
    executor = LocalExecutor()
    node = IRNode(node_id="n1", node_type=NodeType.LOGIC, inputs={}, outputs={})

    with pytest.raises(ValueError, match="Unsupported node type"):
        await executor._execute_node_impl(node, {})
