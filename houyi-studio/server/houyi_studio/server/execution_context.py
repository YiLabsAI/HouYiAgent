"""Execution context for console execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from houyi.protocol.ir import ExecutionIR, PlanIR

from .agent_comm_service import AgentCommService
from .context_bundle import ContextBundle
from .context_service import ContextService
from .mcp_gateway import MCPGateway
from .memory_service import MemoryService
from .rag_service import RAGService

if TYPE_CHECKING:
    from houyi.observability import Span


@dataclass(slots=True)
class ExecutionContext:
    session_id: str
    execution: ExecutionIR
    plan: PlanIR
    run_settings: dict[str, Any] = field(default_factory=dict)
    context_service: ContextService | None = None
    memory_service: MemoryService | None = None
    rag_service: RAGService | None = None
    mcp_gateway: MCPGateway | None = None
    agent_comm_service: AgentCommService | None = None
    context_bundle: ContextBundle | None = None
    root_span: Span | None = None

    def update_plan(self, plan: PlanIR) -> None:
        self.plan = plan
