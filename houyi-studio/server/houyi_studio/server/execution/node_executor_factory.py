"""Factory for building NodeExecutorRegistry with defaults."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from houyi.application.workflow.config_service import ConfigService
from houyi.interface.protocol.ir.plan_ir import NodeType

from .node_executor_registry import NodeExecutorRegistry
from .node_executors import (
    LLMNodeExecutor,
    LogicNodeExecutor,
    RouteNodeExecutor,
    ToolNodeExecutor,
    VerifyNodeExecutor,
)

ExecuteLLMReal = Callable[..., Awaitable[None]]
ExecuteLLMMock = Callable[..., Awaitable[None]]


class NodeExecutorFactory:
    """Build a NodeExecutorRegistry with default executors."""

    def __init__(
        self,
        *,
        config_service: ConfigService,
        execute_llm_real: ExecuteLLMReal,
        execute_llm_mock: ExecuteLLMMock,
    ) -> None:
        self._config_service = config_service
        self._execute_llm_real = execute_llm_real
        self._execute_llm_mock = execute_llm_mock

    def build_registry(self) -> NodeExecutorRegistry:
        """Create and return a registry with built-in executors."""
        registry = NodeExecutorRegistry()
        registry.register(
            NodeType.LLM,
            LLMNodeExecutor(
                config_service=self._config_service,
                execute_llm_real=self._execute_llm_real,
                execute_llm_mock=self._execute_llm_mock,
            ),
        )
        registry.register(NodeType.TOOL, ToolNodeExecutor())
        registry.register(NodeType.VERIFY, VerifyNodeExecutor())
        registry.register(NodeType.ROUTE, RouteNodeExecutor())
        registry.register(NodeType.LOGIC, LogicNodeExecutor())
        return registry
