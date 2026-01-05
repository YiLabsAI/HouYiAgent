"""Comprehensive tests to reach 80% coverage target."""

from pydantic import BaseModel

from houyi import AgentSpec, SkillSpec
from houyi.core.task import TaskSpec
from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.state import SessionState
from houyi.runtime.agent import Agent


class TestComprehensiveCoverage:
    """Additional tests to increase coverage."""

    def test_task_spec_creation(self):
        """Test TaskSpec creation."""
        task = TaskSpec(description="Test task", expected_output="Expected result")

        assert task.description == "Test task"
        assert task.expected_output == "Expected result"

    def test_task_spec_with_context(self):
        """Test TaskSpec with context dependencies."""
        task = TaskSpec(description="Dependent task", context=[0, 1])

        assert task.context == [0, 1]

    def test_ir_node_is_ready(self):
        """Test IRNode ready check."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
            dependencies=["dep1", "dep2"],
        )

        # Not ready - missing dependencies
        assert not node.is_ready(set())
        assert not node.is_ready({"dep1"})

        # Ready - all dependencies met
        assert node.is_ready({"dep1", "dep2"})

    def test_ir_node_get_input_values(self):
        """Test IRNode input value resolution."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "$node0.result", "static": "value"},
            outputs={"result": "$output"},
        )

        context = {"node0.result": "resolved_value"}
        inputs = node.get_input_values(context)

        assert inputs["task"] == "resolved_value"
        assert inputs["static"] == "value"

    def test_execution_plan_creation(self):
        """Test ExecutionPlan creation."""
        node1 = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
        )

        plan = ExecutionPlan(plan_id="test_plan", nodes=[node1], entry_node="node1")

        assert plan.plan_id == "test_plan"
        assert len(plan.nodes) == 1
        assert plan.entry_node == "node1"

    def test_node_types(self):
        """Test different node types."""
        llm_node = IRNode(
            node_id="llm1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
        )

        tool_node = IRNode(
            node_id="tool1",
            node_type=NodeType.TOOL,
            inputs={"input": "test"},
            outputs={"output": "$result"},
        )

        verify_node = IRNode(
            node_id="verify1",
            node_type=NodeType.VERIFY,
            inputs={"result": "$tool1.output"},
            outputs={"verified": "$status"},
        )

        assert llm_node.node_type == NodeType.LLM
        assert tool_node.node_type == NodeType.TOOL
        assert verify_node.node_type == NodeType.VERIFY

    def test_node_type_enum(self):
        """Test NodeType enum values."""
        assert NodeType.LLM == "llm"
        assert NodeType.TOOL == "tool"
        assert NodeType.VERIFY == "verify"

    def test_agent_spec_with_policies(self):
        """Test AgentSpec with custom policies."""
        agent = AgentSpec(
            role="Test Agent", policies={"llm": "gpt-4", "temperature": 0.7, "max_tokens": 1000}
        )

        assert agent.policies["llm"] == "gpt-4"
        assert agent.policies["temperature"] == 0.7
        assert agent.policies["max_tokens"] == 1000

    def test_skill_spec_with_constraints(self):
        """Test SkillSpec with constraints."""

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description="Test skill",
            input_schema=Input,
            output_schema=Output,
            executor=func,
            constraints={"timeout_ms": 5000, "max_retries": 3},
        )

        assert skill.constraints["timeout_ms"] == 5000
        assert skill.constraints["max_retries"] == 3

    def test_agent_get_tool_schemas(self):
        """Test agent tool schema generation."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="found")

        skill = SkillSpec(
            name="search",
            description="Search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = Agent(role="Agent", skills=[skill])
        schemas = agent.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"

    def test_ir_node_with_metadata(self):
        """Test IRNode with metadata."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.TOOL,
            inputs={"x": "1"},
            outputs={"y": "$output"},
            metadata={"skill_name": "calculator", "version": "1.0"},
        )

        assert node.metadata["skill_name"] == "calculator"
        assert node.metadata["version"] == "1.0"

    def test_execution_plan_with_multiple_nodes(self):
        """Test ExecutionPlan with multiple nodes."""
        node1 = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "step1"},
            outputs={"result": "$output"},
        )

        node2 = IRNode(
            node_id="node2",
            node_type=NodeType.TOOL,
            inputs={"input": "$node1.result"},
            outputs={"output": "$result"},
            dependencies=["node1"],
        )

        plan = ExecutionPlan(plan_id="multi_node_plan", nodes=[node1, node2], entry_node="node1")

        assert len(plan.nodes) == 2
        assert plan.nodes[1].dependencies == ["node1"]

    def test_session_state_with_metadata(self):
        """Test SessionState with metadata."""
        state = SessionState(session_id="session_1", agent_id="agent_1", metadata={"key": "value"})

        assert state.metadata["key"] == "value"
        assert state.session_id == "session_1"

    def test_agent_spec_empty_skills(self):
        """Test AgentSpec with no skills."""
        agent = AgentSpec(role="Simple Agent")

        assert len(agent.skills) == 0
        assert agent.role == "Simple Agent"

    def test_agent_spec_system_prompt_with_skills(self):
        """Test system prompt generation with skills."""

        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def calc(input: Input) -> Output:
            return Output(y=input.x * 2)

        skill = SkillSpec(
            name="calculator",
            description="Calculate values",
            input_schema=Input,
            output_schema=Output,
            executor=calc,
        )

        agent = AgentSpec(role="Math Agent", skills=[skill])

        prompt = agent.to_system_prompt()

        assert "Math Agent" in prompt
        assert "calculator" in prompt
        assert "Calculate values" in prompt

    def test_agent_spec_custom_system_prompt(self):
        """Test AgentSpec with custom system prompt."""
        agent = AgentSpec(role="Custom Agent", system_prompt="Custom instructions here")

        prompt = agent.to_system_prompt()
        assert prompt == "Custom instructions here"
