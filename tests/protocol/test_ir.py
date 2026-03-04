"""Tests for IR definitions and conversions."""

from __future__ import annotations

from datetime import UTC

from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.protocol.ir import CheckpointIR, CheckpointTrigger, ExecutionIR, LLMCallLog, NodeStatus
from houyi.protocol.ir.converter import IRConverter
from houyi.protocol.ir.plan_ir import EdgeIR, NodeIR, PlanIR


class TestPlanIR:
    """Test PlanIR structure and validation."""

    def test_create_simple_plan(self) -> None:
        """Test creating a simple plan with nodes and edges."""
        plan = PlanIR(
            plan_id="test_plan",
            nodes=[
                NodeIR(node_id="node1", node_type=NodeType.LLM),
                NodeIR(node_id="node2", node_type=NodeType.TOOL),
            ],
            edges=[
                EdgeIR(
                    edge_id="edge1",
                    source_node_id="node1",
                    target_node_id="node2",
                )
            ],
            entry_node_id="node1",
        )

        assert plan.plan_id == "test_plan"
        assert len(plan.nodes) == 2
        assert len(plan.edges) == 1
        assert plan.version == 1

    def test_get_node(self) -> None:
        """Test getting a node by ID."""
        plan = PlanIR(
            plan_id="test_plan",
            nodes=[
                NodeIR(node_id="node1", node_type=NodeType.LLM),
                NodeIR(node_id="node2", node_type=NodeType.TOOL),
            ],
            edges=[],
            entry_node_id="node1",
        )

        node = plan.get_node("node1")
        assert node is not None
        assert node.node_id == "node1"

        assert plan.get_node("nonexistent") is None

    def test_get_dependencies(self) -> None:
        """Test getting node dependencies."""
        plan = PlanIR(
            plan_id="test_plan",
            nodes=[
                NodeIR(node_id="node1", node_type=NodeType.LLM),
                NodeIR(node_id="node2", node_type=NodeType.TOOL),
                NodeIR(node_id="node3", node_type=NodeType.VERIFY),
            ],
            edges=[
                EdgeIR(edge_id="e1", source_node_id="node1", target_node_id="node2"),
                EdgeIR(edge_id="e2", source_node_id="node2", target_node_id="node3"),
            ],
            entry_node_id="node1",
        )

        deps = plan.get_dependencies("node3")
        assert deps == ["node2"]

        deps = plan.get_dependencies("node2")
        assert deps == ["node1"]

        deps = plan.get_dependencies("node1")
        assert deps == []

    def test_validate_dag_valid(self) -> None:
        """Test DAG validation for a valid graph."""
        plan = PlanIR(
            plan_id="test_plan",
            nodes=[
                NodeIR(node_id="node1", node_type=NodeType.LLM),
                NodeIR(node_id="node2", node_type=NodeType.TOOL),
            ],
            edges=[EdgeIR(edge_id="e1", source_node_id="node1", target_node_id="node2")],
            entry_node_id="node1",
        )

        is_valid, error = plan.validate_dag()
        assert is_valid
        assert error is None

    def test_validate_dag_cycle(self) -> None:
        """Test DAG validation detects cycles."""
        plan = PlanIR(
            plan_id="test_plan",
            nodes=[
                NodeIR(node_id="node1", node_type=NodeType.LLM),
                NodeIR(node_id="node2", node_type=NodeType.TOOL),
            ],
            edges=[
                EdgeIR(edge_id="e1", source_node_id="node1", target_node_id="node2"),
                EdgeIR(edge_id="e2", source_node_id="node2", target_node_id="node1"),
            ],
            entry_node_id="node1",
        )

        is_valid, error = plan.validate_dag()
        assert not is_valid
        assert error is not None
        assert "Cycle" in error

    def test_soft_delete_node(self) -> None:
        """Test soft delete functionality (DECISION-004)."""
        from datetime import datetime

        node = NodeIR(node_id="node1", node_type=NodeType.LLM)
        assert node.deleted_at is None

        node.deleted_at = datetime.now(UTC)
        assert node.deleted_at is not None


class TestExecutionIR:
    """Test ExecutionIR structure and updates."""

    def test_create_execution(self) -> None:
        """Test creating execution state."""
        execution = ExecutionIR(
            execution_id="exec1",
            plan_id="plan1",
        )

        assert execution.execution_id == "exec1"
        assert execution.plan_id == "plan1"
        assert len(execution.node_executions) == 0

    def test_update_node_status(self) -> None:
        """Test updating node execution status."""
        execution = ExecutionIR(
            execution_id="exec1",
            plan_id="plan1",
        )

        execution.update_node_status("node1", NodeStatus.RUNNING)
        node_exec = execution.get_node_execution("node1")
        assert node_exec is not None
        assert node_exec.status == NodeStatus.RUNNING
        assert node_exec.started_at is not None

        execution.update_node_status(
            "node1",
            NodeStatus.COMPLETED,
            outputs={"result": "success"},
        )
        assert node_exec.status == NodeStatus.COMPLETED
        assert node_exec.completed_at is not None
        assert node_exec.outputs == {"result": "success"}

    def test_get_completed_nodes(self) -> None:
        """Test getting completed nodes."""
        execution = ExecutionIR(
            execution_id="exec1",
            plan_id="plan1",
        )

        execution.update_node_status("node1", NodeStatus.COMPLETED)
        execution.update_node_status("node2", NodeStatus.RUNNING)
        execution.update_node_status("node3", NodeStatus.SKIPPED)

        completed = execution.get_completed_nodes()
        assert completed == {"node1", "node3"}


class TestCheckpointIR:
    """Test CheckpointIR structure."""

    def test_create_checkpoint(self) -> None:
        """Test creating a checkpoint."""
        checkpoint = CheckpointIR(
            checkpoint_id="cp1",
            execution_id="exec1",
            plan_id="plan1",
            sequence_number=1,
            trigger=CheckpointTrigger.NODE_COMPLETED,
            execution_snapshot={"status": "running"},
        )

        assert checkpoint.checkpoint_id == "cp1"
        assert checkpoint.sequence_number == 1
        assert checkpoint.trigger == CheckpointTrigger.NODE_COMPLETED
        assert not checkpoint.is_incremental()

    def test_incremental_checkpoint(self) -> None:
        """Test incremental checkpoint (DECISION-001)."""
        checkpoint = CheckpointIR(
            checkpoint_id="cp2",
            execution_id="exec1",
            plan_id="plan1",
            sequence_number=2,
            trigger=CheckpointTrigger.NODE_COMPLETED,
            execution_snapshot={},
            parent_checkpoint_id="cp1",
            delta={"node2": {"status": "completed"}},
        )

        assert checkpoint.is_incremental()
        assert checkpoint.parent_checkpoint_id == "cp1"
        assert checkpoint.delta is not None

    def test_llm_call_log(self) -> None:
        """Test LLM call logging for deterministic replay."""
        log = LLMCallLog(
            call_id="call1",
            node_id="node1",
            model="gpt-4",
            prompt="What is 2+2?",
            response="4",
        )

        assert log.call_id == "call1"
        assert log.node_id == "node1"
        assert log.response == "4"

        checkpoint = CheckpointIR(
            checkpoint_id="cp1",
            execution_id="exec1",
            plan_id="plan1",
            sequence_number=1,
            trigger=CheckpointTrigger.NODE_COMPLETED,
            execution_snapshot={},
            llm_call_logs=[log],
        )

        node_calls = checkpoint.get_llm_calls_for_node("node1")
        assert len(node_calls) == 1
        assert node_calls[0].call_id == "call1"


class TestIRConverter:
    """Test bidirectional IR conversion."""

    def test_execution_plan_to_plan_ir(self) -> None:
        """Test converting ExecutionPlan to PlanIR."""
        exec_plan = ExecutionPlan(
            plan_id="plan1",
            nodes=[
                IRNode(
                    node_id="node1",
                    node_type=NodeType.LLM,
                    inputs={"task": "test"},
                    outputs={"result": "$result"},
                    dependencies=[],
                ),
                IRNode(
                    node_id="node2",
                    node_type=NodeType.TOOL,
                    inputs={"input": "$result"},
                    outputs={"output": "$output"},
                    dependencies=["node1"],
                ),
            ],
            entry_node="node1",
        )

        plan_ir = IRConverter.execution_plan_to_plan_ir(exec_plan)

        assert plan_ir.plan_id == "plan1"
        assert len(plan_ir.nodes) == 2
        assert len(plan_ir.edges) == 1
        assert plan_ir.entry_node_id == "node1"

        edge = plan_ir.edges[0]
        assert edge.source_node_id == "node1"
        assert edge.target_node_id == "node2"

    def test_plan_ir_to_execution_plan(self) -> None:
        """Test converting PlanIR to ExecutionPlan."""
        plan_ir = PlanIR(
            plan_id="plan1",
            nodes=[
                NodeIR(
                    node_id="node1",
                    node_type=NodeType.LLM,
                    inputs={"task": "test"},
                    outputs={"result": "$result"},
                ),
                NodeIR(
                    node_id="node2",
                    node_type=NodeType.TOOL,
                    inputs={"input": "$result"},
                    outputs={"output": "$output"},
                ),
            ],
            edges=[EdgeIR(edge_id="e1", source_node_id="node1", target_node_id="node2")],
            entry_node_id="node1",
        )

        exec_plan = IRConverter.plan_ir_to_execution_plan(plan_ir)

        assert exec_plan.plan_id == "plan1"
        assert len(exec_plan.nodes) == 2
        assert exec_plan.entry_node == "node1"

        node2 = exec_plan.get_node("node2")
        assert node2 is not None
        assert node2.dependencies == ["node1"]

    def test_bidirectional_conversion(self) -> None:
        """Test lossless bidirectional conversion."""
        exec_plan = ExecutionPlan(
            plan_id="plan1",
            nodes=[
                IRNode(
                    node_id="node1",
                    node_type=NodeType.LLM,
                    inputs={"task": "test"},
                    outputs={"result": "$result"},
                    dependencies=[],
                    metadata={"purpose": "main"},
                ),
            ],
            entry_node="node1",
            metadata={"task": "test_task"},
        )

        plan_ir = IRConverter.execution_plan_to_plan_ir(exec_plan)
        exec_plan_2 = IRConverter.plan_ir_to_execution_plan(plan_ir)

        assert exec_plan_2.plan_id == exec_plan.plan_id
        assert len(exec_plan_2.nodes) == len(exec_plan.nodes)
        assert exec_plan_2.entry_node == exec_plan.entry_node
        assert exec_plan_2.metadata == exec_plan.metadata

    def test_create_execution_ir(self) -> None:
        """Test creating ExecutionIR from PlanIR."""
        plan_ir = PlanIR(
            plan_id="plan1",
            nodes=[
                NodeIR(node_id="node1", node_type=NodeType.LLM),
                NodeIR(node_id="node2", node_type=NodeType.TOOL),
            ],
            edges=[],
            entry_node_id="node1",
        )

        execution_ir = IRConverter.create_execution_ir("exec1", plan_ir)

        assert execution_ir.execution_id == "exec1"
        assert execution_ir.plan_id == "plan1"
        assert len(execution_ir.node_executions) == 2

        for node_exec in execution_ir.node_executions.values():
            assert node_exec.status == NodeStatus.PENDING

    def test_soft_deleted_nodes_skipped(self) -> None:
        """Test soft-deleted nodes are marked as skipped."""
        from datetime import datetime

        plan_ir = PlanIR(
            plan_id="plan1",
            nodes=[
                NodeIR(node_id="node1", node_type=NodeType.LLM),
                NodeIR(
                    node_id="node2",
                    node_type=NodeType.TOOL,
                    deleted_at=datetime.now(UTC),
                ),
            ],
            edges=[],
            entry_node_id="node1",
        )

        execution_ir = IRConverter.create_execution_ir("exec1", plan_ir)

        assert execution_ir.node_executions["node1"].status == NodeStatus.PENDING
        assert execution_ir.node_executions["node2"].status == NodeStatus.SKIPPED

        exec_plan = IRConverter.plan_ir_to_execution_plan(plan_ir)
        assert len(exec_plan.nodes) == 1
        assert exec_plan.get_node("node2") is None


class TestToolCallTraceIR:
    """Test ToolCallTraceIR tooling IR fields."""

    def test_parallel_group_id_field(self) -> None:
        from houyi.protocol.ir.tooling_ir import ToolCallTraceIR

        ir = ToolCallTraceIR(
            tool_name="search",
            tool_call_id="c1",
            parallel_group_id="round_1",
            args={"q": "test"},
        )
        assert ir.parallel_group_id == "round_1"

    def test_parallel_group_id_default_none(self) -> None:
        from houyi.protocol.ir.tooling_ir import ToolCallTraceIR

        ir = ToolCallTraceIR(tool_name="search")
        assert ir.parallel_group_id is None

    def test_serialization_includes_parallel_group_id(self) -> None:
        from houyi.protocol.ir.tooling_ir import ToolCallTraceIR

        ir = ToolCallTraceIR(
            tool_name="search",
            parallel_group_id="round_3",
        )
        data = ir.model_dump()
        assert data["parallel_group_id"] == "round_3"
