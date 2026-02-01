"""Node execution service for running single plan nodes."""

from __future__ import annotations

from typing import Any

from houyi.protocol.ir import ExecutionIR, PlanIR

from .execution_context import ExecutionContext
from .node_execution_flow import NodeExecutionFlow


class NodeExecutionService:
    """Service for executing individual nodes via NodeExecutionFlow."""

    def __init__(self, node_execution_flow: NodeExecutionFlow) -> None:
        self._node_execution_flow = node_execution_flow

    async def execute_node(
        self,
        *,
        context: ExecutionContext | None = None,
        node_id: str | None = None,
        session_id: str | None = None,
        execution: ExecutionIR | None = None,
        plan: PlanIR | None = None,
        sleep_func: Any | None = None,
    ) -> None:
        """Execute a node using the configured execution flow."""
        if sleep_func is not None:
            self._node_execution_flow.set_sleep_func(sleep_func)
        await self._node_execution_flow.execute(
            context,
            node_id,
            session_id=session_id,
            execution=execution,
            plan=plan,
        )
