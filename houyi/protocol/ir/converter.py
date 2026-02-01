"""IR Converter: Bidirectional conversion between internal types and IR."""

from __future__ import annotations

from typing import Any

from houyi.orchestration.plan import ExecutionPlan, IRNode

from .execution_ir import ExecutionIR, NodeExecutionIR, NodeStatus
from .plan_ir import EdgeIR, NodeIR, PlanIR


class IRConverter:
    """Converter between internal ExecutionPlan and frontend-backend PlanIR.

    Ensures lossless bidirectional conversion for isomorphic architecture.
    """

    @staticmethod
    def execution_plan_to_plan_ir(plan: ExecutionPlan) -> PlanIR:
        """Convert ExecutionPlan to PlanIR.

        Args:
            plan: Internal execution plan

        Returns:
            Frontend-backend shared PlanIR
        """
        # Convert nodes
        nodes: list[NodeIR] = []
        for idx, ir_node in enumerate(plan.nodes):
            node_ir = NodeIR(
                node_id=ir_node.node_id,
                node_type=ir_node.node_type,
                position={"x": idx * 200, "y": 0},  # Default layout
                config={
                    "timeout": 300,  # Default 5 min
                    "max_retries": 3,
                },
                inputs=ir_node.inputs,
                outputs=ir_node.outputs,
                metadata=ir_node.metadata,
            )
            nodes.append(node_ir)

        # Convert dependencies to edges
        edges: list[EdgeIR] = []
        edge_counter = 0
        for ir_node in plan.nodes:
            for dep_id in ir_node.dependencies:
                edge = EdgeIR(
                    edge_id=f"edge_{edge_counter}",
                    source_node_id=dep_id,
                    target_node_id=ir_node.node_id,
                )
                edges.append(edge)
                edge_counter += 1

        return PlanIR(
            plan_id=plan.plan_id,
            nodes=nodes,
            edges=edges,
            entry_node_id=plan.entry_node,
            metadata=plan.metadata,
        )

    @staticmethod
    def plan_ir_to_execution_plan(plan_ir: PlanIR) -> ExecutionPlan:
        """Convert PlanIR to ExecutionPlan.

        Args:
            plan_ir: Frontend-backend shared PlanIR

        Returns:
            Internal execution plan
        """
        # Convert nodes
        ir_nodes: list[IRNode] = []
        for node_ir in plan_ir.nodes:
            # Skip soft-deleted nodes
            if node_ir.deleted_at is not None:
                continue

            # Build dependencies from edges
            dependencies = plan_ir.get_dependencies(node_ir.node_id)

            ir_node = IRNode(
                node_id=node_ir.node_id,
                node_type=node_ir.node_type,
                inputs=node_ir.inputs,
                outputs=node_ir.outputs,
                dependencies=dependencies,
                metadata=node_ir.metadata,
            )
            ir_nodes.append(ir_node)

        return ExecutionPlan(
            plan_id=plan_ir.plan_id,
            nodes=ir_nodes,
            entry_node=plan_ir.entry_node_id,
            metadata=plan_ir.metadata,
        )

    @staticmethod
    def create_execution_ir(
        execution_id: str,
        plan_ir: PlanIR,
    ) -> ExecutionIR:
        """Create initial ExecutionIR for a plan.

        Args:
            execution_id: Unique execution identifier
            plan_ir: Plan to execute

        Returns:
            Initial execution state
        """
        # Initialize node executions
        node_executions: dict[str, NodeExecutionIR] = {}
        for node_ir in plan_ir.nodes:
            # Skip soft-deleted nodes
            if node_ir.deleted_at is not None:
                status = NodeStatus.SKIPPED
            else:
                status = NodeStatus.PENDING

            node_executions[node_ir.node_id] = NodeExecutionIR(
                node_id=node_ir.node_id,
                status=status,
            )

        return ExecutionIR(
            execution_id=execution_id,
            plan_id=plan_ir.plan_id,
            node_executions=node_executions,
        )

    @staticmethod
    def merge_execution_state(
        execution_ir: ExecutionIR,
        node_id: str,
        status: NodeStatus,
        **updates: Any,
    ) -> ExecutionIR:
        """Update execution state for a node.

        Args:
            execution_ir: Current execution state
            node_id: Node to update
            status: New status
            **updates: Additional fields to update

        Returns:
            Updated execution state
        """
        execution_ir.update_node_status(node_id, status, **updates)
        return execution_ir
