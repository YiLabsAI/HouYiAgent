"""Node executor registry scaffolding."""

from __future__ import annotations

from typing import Protocol

from houyi.interface.protocol.ir.execution_ir import NodeExecutionIR
from houyi.interface.protocol.ir.plan_ir import NodeIR, NodeType

from .context import ExecutionContext


class NodeExecutor(Protocol):
    """Protocol for node executors."""

    async def execute(
        self,
        context: ExecutionContext,
        node: NodeIR,
        node_exec: NodeExecutionIR,
    ) -> None:
        """Execute a plan node."""


class NodeExecutorRegistry:
    """Registry for node executors."""

    def __init__(self) -> None:
        self._executors: dict[NodeType, NodeExecutor] = {}

    def register(self, node_type: NodeType, executor: NodeExecutor) -> None:
        self._executors[node_type] = executor

    def resolve(self, node_type: NodeType) -> NodeExecutor | None:
        return self._executors.get(node_type)

    def list(self) -> dict[NodeType, NodeExecutor]:
        return dict(self._executors)
