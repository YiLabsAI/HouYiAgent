"""Tests for IRNode class."""

from houyi.application.workflow.orchestration.plan import IRNode, NodeType


class TestIRNodeCreation:
    """Test IRNode creation and initialization."""

    def test_ir_node_llm_type(self):
        """Test IRNode with LLM type."""
        node = IRNode(
            node_id="llm1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
        )

        assert node.node_id == "llm1"
        assert node.node_type == NodeType.LLM
        assert node.inputs["task"] == "test"

    def test_ir_node_tool_type(self):
        """Test IRNode with TOOL type."""
        node = IRNode(
            node_id="tool1",
            node_type=NodeType.TOOL,
            inputs={"input": "data"},
            outputs={"output": "$result"},
        )

        assert node.node_type == NodeType.TOOL

    def test_ir_node_verify_type(self):
        """Test IRNode with VERIFY type."""
        node = IRNode(
            node_id="verify1",
            node_type=NodeType.VERIFY,
            inputs={"result": "data"},
            outputs={"verified": "$status"},
        )

        assert node.node_type == NodeType.VERIFY

    def test_ir_node_with_dependencies(self):
        """Test IRNode with dependencies."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
            dependencies=["dep1", "dep2"],
        )

        assert len(node.dependencies) == 2
        assert "dep1" in node.dependencies

    def test_ir_node_no_dependencies(self):
        """Test IRNode without dependencies."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
        )

        assert len(node.dependencies) == 0

    def test_ir_node_with_metadata(self):
        """Test IRNode with metadata."""
        metadata = {"model": "gpt-4", "temperature": 0.7}

        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
            metadata=metadata,
        )

        assert node.metadata["model"] == "gpt-4"
        assert node.metadata["temperature"] == 0.7


class TestIRNodeReadiness:
    """Test IRNode readiness checking."""

    def test_is_ready_no_dependencies(self):
        """Test node is ready when it has no dependencies."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
        )

        assert node.is_ready(set())
        assert node.is_ready({"any", "deps"})

    def test_ready_with_satisfied_deps(self):
        """Test node is ready when dependencies are satisfied."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
            dependencies=["dep1", "dep2"],
        )

        assert node.is_ready({"dep1", "dep2"})
        assert node.is_ready({"dep1", "dep2", "extra"})

    def test_ready_with_unsatisfied_deps(self):
        """Test node is not ready when dependencies are not satisfied."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
            dependencies=["dep1", "dep2"],
        )

        assert not node.is_ready(set())
        assert not node.is_ready({"dep1"})
        assert not node.is_ready({"dep2"})
        assert not node.is_ready({"other"})

    def test_is_ready_partial_dependencies(self):
        """Test node is not ready with partial dependencies."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
            dependencies=["dep1", "dep2", "dep3"],
        )

        assert not node.is_ready({"dep1", "dep2"})


class TestIRNodeInputResolution:
    """Test IRNode input value resolution."""

    def test_get_input_values_static(self):
        """Test input resolution with static values."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "static_value", "param": "another"},
            outputs={"result": "$output"},
        )

        inputs = node.get_input_values({})

        assert inputs["task"] == "static_value"
        assert inputs["param"] == "another"

    def test_input_values_from_context(self):
        """Test input resolution from context."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "$node0.result"},
            outputs={"result": "$output"},
        )

        context = {"node0.result": "resolved_value"}
        inputs = node.get_input_values(context)

        assert inputs["task"] == "resolved_value"

    def test_get_input_values_mixed(self):
        """Test input resolution with mixed static and dynamic values."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "$node0.result", "static": "value", "another": "$node1.data"},
            outputs={"result": "$output"},
        )

        context = {"node0.result": "resolved1", "node1.data": "resolved2"}
        inputs = node.get_input_values(context)

        assert inputs["task"] == "resolved1"
        assert inputs["static"] == "value"
        assert inputs["another"] == "resolved2"

    def test_input_values_missing_context(self):
        """Test input resolution with missing context."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "$missing.value"},
            outputs={"result": "$output"},
        )

        inputs = node.get_input_values({})

        # Should return the reference as-is if not in context
        assert inputs["task"] == "$missing.value"

    def test_input_values_empty_inputs(self):
        """Test input resolution with empty inputs."""
        node = IRNode(
            node_id="node1", node_type=NodeType.LLM, inputs={}, outputs={"result": "$output"}
        )

        inputs = node.get_input_values({})

        assert inputs == {}


class TestIRNodeOutputs:
    """Test IRNode outputs configuration."""

    def test_ir_node_single_output(self):
        """Test IRNode with single output."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
        )

        assert len(node.outputs) == 1
        assert "result" in node.outputs

    def test_ir_node_multiple_outputs(self):
        """Test IRNode with multiple outputs."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.TOOL,
            inputs={"input": "data"},
            outputs={"output1": "$result1", "output2": "$result2", "output3": "$result3"},
        )

        assert len(node.outputs) == 3

    def test_ir_node_empty_outputs(self):
        """Test IRNode with empty outputs."""
        node = IRNode(node_id="node1", node_type=NodeType.LLM, inputs={"task": "test"}, outputs={})

        assert node.outputs == {}


class TestNodeTypeEnum:
    """Test NodeType enum values."""

    def test_node_type_enum_values(self):
        assert NodeType.LLM == "llm"
        assert NodeType.TOOL == "tool"
        assert NodeType.VERIFY == "verify"
