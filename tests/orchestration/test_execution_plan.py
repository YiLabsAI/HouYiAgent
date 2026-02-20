"""Tests for ExecutionPlan class."""

from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType


class TestExecutionPlanCreation:
    """Test ExecutionPlan creation and initialization."""

    def test_execution_plan_minimal(self):
        """Test ExecutionPlan with minimal configuration."""
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
        )

        plan = ExecutionPlan(plan_id="test_plan", nodes=[node], entry_node="node1")

        assert plan.plan_id == "test_plan"
        assert len(plan.nodes) == 1
        assert plan.entry_node == "node1"

    def test_execution_plan_multiple_nodes(self):
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

        plan = ExecutionPlan(plan_id="multi_node", nodes=[node1, node2], entry_node="node1")

        assert len(plan.nodes) == 2
        assert plan.nodes[0].node_id == "node1"
        assert plan.nodes[1].node_id == "node2"

    def test_execution_plan_dag_structure(self):
        """Test ExecutionPlan with DAG structure."""
        # Create a simple DAG: start -> [branch1, branch2] -> merge
        start = IRNode(
            node_id="start",
            node_type=NodeType.LLM,
            inputs={"task": "begin"},
            outputs={"result": "$output"},
        )

        branch1 = IRNode(
            node_id="branch1",
            node_type=NodeType.TOOL,
            inputs={"input": "$start.result"},
            outputs={"output": "$result"},
            dependencies=["start"],
        )

        branch2 = IRNode(
            node_id="branch2",
            node_type=NodeType.TOOL,
            inputs={"input": "$start.result"},
            outputs={"output": "$result"},
            dependencies=["start"],
        )

        merge = IRNode(
            node_id="merge",
            node_type=NodeType.LLM,
            inputs={"input1": "$branch1.output", "input2": "$branch2.output"},
            outputs={"result": "$output"},
            dependencies=["branch1", "branch2"],
        )

        plan = ExecutionPlan(
            plan_id="dag", nodes=[start, branch1, branch2, merge], entry_node="start"
        )

        assert len(plan.nodes) == 4
        assert plan.entry_node == "start"
        assert len(plan.nodes[3].dependencies) == 2

    def test_execution_plan_linear_chain(self):
        """Test ExecutionPlan with linear chain."""
        nodes = []
        for i in range(5):
            deps = [f"node{i - 1}"] if i > 0 else []
            node = IRNode(
                node_id=f"node{i}",
                node_type=NodeType.LLM,
                inputs={"task": f"step{i}"},
                outputs={"result": "$output"},
                dependencies=deps,
            )
            nodes.append(node)

        plan = ExecutionPlan(plan_id="chain", nodes=nodes, entry_node="node0")

        assert len(plan.nodes) == 5
        assert plan.nodes[0].dependencies == []
        assert plan.nodes[4].dependencies == ["node3"]


class TestExecutionPlanNodeAccess:
    """Test ExecutionPlan node access."""

    def test_execution_plan_node_by_index(self):
        """Test accessing nodes by index."""
        nodes = []
        for i in range(3):
            node = IRNode(
                node_id=f"node{i}",
                node_type=NodeType.LLM,
                inputs={"task": f"task{i}"},
                outputs={"result": "$output"},
            )
            nodes.append(node)

        plan = ExecutionPlan(plan_id="test", nodes=nodes, entry_node="node0")

        assert plan.nodes[0].node_id == "node0"
        assert plan.nodes[1].node_id == "node1"
        assert plan.nodes[2].node_id == "node2"

    def test_execution_plan_node_iteration(self):
        """Test iterating over nodes."""
        nodes = []
        for i in range(4):
            node = IRNode(
                node_id=f"node{i}",
                node_type=NodeType.LLM,
                inputs={"task": f"task{i}"},
                outputs={"result": "$output"},
            )
            nodes.append(node)

        plan = ExecutionPlan(plan_id="test", nodes=nodes, entry_node="node0")

        count = 0
        for node in plan.nodes:
            assert node.node_id == f"node{count}"
            count += 1

        assert count == 4


class TestExecutionPlanComplexScenarios:
    """Test ExecutionPlan complex scenarios."""

    def test_execution_plan_with_verify_nodes(self):
        """Test ExecutionPlan with VERIFY nodes."""
        llm_node = IRNode(
            node_id="llm",
            node_type=NodeType.LLM,
            inputs={"task": "generate"},
            outputs={"result": "$output"},
        )

        verify_node = IRNode(
            node_id="verify",
            node_type=NodeType.VERIFY,
            inputs={"result": "$llm.result"},
            outputs={"verified": "$status"},
            dependencies=["llm"],
        )

        plan = ExecutionPlan(plan_id="with_verify", nodes=[llm_node, verify_node], entry_node="llm")

        assert plan.nodes[1].node_type == NodeType.VERIFY

    def test_execution_plan_mixed_node_types(self):
        """Test ExecutionPlan with mixed node types."""
        nodes = [
            IRNode(
                node_id="llm1",
                node_type=NodeType.LLM,
                inputs={"task": "think"},
                outputs={"result": "$output"},
            ),
            IRNode(
                node_id="tool1",
                node_type=NodeType.TOOL,
                inputs={"input": "$llm1.result"},
                outputs={"output": "$result"},
                dependencies=["llm1"],
            ),
            IRNode(
                node_id="verify1",
                node_type=NodeType.VERIFY,
                inputs={"result": "$tool1.output"},
                outputs={"verified": "$status"},
                dependencies=["tool1"],
            ),
            IRNode(
                node_id="llm2",
                node_type=NodeType.LLM,
                inputs={"task": "$verify1.verified"},
                outputs={"result": "$output"},
                dependencies=["verify1"],
            ),
        ]

        plan = ExecutionPlan(plan_id="mixed", nodes=nodes, entry_node="llm1")

        assert len(plan.nodes) == 4
        assert plan.nodes[0].node_type == NodeType.LLM
        assert plan.nodes[1].node_type == NodeType.TOOL
        assert plan.nodes[2].node_type == NodeType.VERIFY
        assert plan.nodes[3].node_type == NodeType.LLM

    def test_execution_plan_parallel_branches(self):
        """Test ExecutionPlan with parallel branches."""
        start = IRNode(
            node_id="start",
            node_type=NodeType.LLM,
            inputs={"task": "begin"},
            outputs={"result": "$output"},
        )

        # Create 3 parallel branches
        branches = []
        for i in range(3):
            branch = IRNode(
                node_id=f"branch{i}",
                node_type=NodeType.TOOL,
                inputs={"input": "$start.result"},
                outputs={"output": "$result"},
                dependencies=["start"],
            )
            branches.append(branch)

        plan = ExecutionPlan(plan_id="parallel", nodes=[start, *branches], entry_node="start")

        assert len(plan.nodes) == 4
        # All branches depend only on start
        for i in range(3):
            assert plan.nodes[i + 1].dependencies == ["start"]

    def test_execution_plan_deep_chain(self):
        """Test ExecutionPlan with deep dependency chain."""
        nodes = []
        for i in range(10):
            deps = [f"node{i - 1}"] if i > 0 else []
            node = IRNode(
                node_id=f"node{i}",
                node_type=NodeType.LLM,
                inputs={"task": f"step{i}"},
                outputs={"result": "$output"},
                dependencies=deps,
            )
            nodes.append(node)

        plan = ExecutionPlan(plan_id="deep", nodes=nodes, entry_node="node0")

        assert len(plan.nodes) == 10
        assert plan.nodes[9].dependencies == ["node8"]
