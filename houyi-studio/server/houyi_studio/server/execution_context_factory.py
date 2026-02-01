"""Factory for building execution contexts."""

from __future__ import annotations

from houyi.protocol.ir import ExecutionIR, PlanIR

from .agent_comm_service import AgentCommService
from .context_service import ContextService
from .execution_context import ExecutionContext
from .mcp_gateway import MCPGateway
from .memory_service import MemoryService
from .rag_service import RAGService


class ExecutionContextFactory:
    """Build execution contexts with shared services."""

    def __init__(
        self,
        *,
        context_service: ContextService,
        memory_service: MemoryService,
        rag_service: RAGService,
        mcp_gateway: MCPGateway,
        agent_comm_service: AgentCommService,
    ) -> None:
        self._context_service = context_service
        self._memory_service = memory_service
        self._rag_service = rag_service
        self._mcp_gateway = mcp_gateway
        self._agent_comm_service = agent_comm_service

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
        )
