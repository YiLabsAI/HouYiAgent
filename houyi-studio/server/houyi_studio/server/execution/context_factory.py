"""Factory for building execution contexts."""

from __future__ import annotations

from typing import Any

from houyi.interface.protocol.ir import ExecutionIR, PlanIR

from ..rag import KnowledgeService
from .agent_comm_service import AgentCommService
from .context import ExecutionContext
from .context_service import ContextService
from .mcp_gateway import MCPGateway
from .memory_service import MemoryService
from .observation_service import ObservationService


class ExecutionContextFactory:
    """Build execution contexts with shared services."""

    def __init__(
        self,
        *,
        context_service: ContextService,
        memory_service: MemoryService,
        rag_service: KnowledgeService,
        mcp_gateway: MCPGateway,
        agent_comm_service: AgentCommService,
        observation_service: ObservationService | None = None,
        tool_cache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._context_service = context_service
        self._memory_service = memory_service
        self._rag_service = rag_service
        self._mcp_gateway = mcp_gateway
        self._agent_comm_service = agent_comm_service
        self._observation_service = observation_service
        self._tool_cache = tool_cache

    def build(self, session_id: str, execution: ExecutionIR, plan: PlanIR) -> ExecutionContext:
        """Build a shared execution context with all required services."""
        return ExecutionContext(
            session_id=session_id,
            execution=execution,
            plan=plan,
            run_settings=execution.metadata.get("run_settings") or {},
            context_service=self._context_service,
            memory_service=self._memory_service,
            rag_service=self._rag_service,
            mcp_gateway=self._mcp_gateway,
            agent_comm_service=self._agent_comm_service,
            observation_service=self._observation_service,
            tool_cache=self._tool_cache,
        )
