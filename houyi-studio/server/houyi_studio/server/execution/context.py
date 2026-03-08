"""Execution context for console execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from houyi.interface.protocol.ir import ExecutionIR, PlanIR

from ..rag import KnowledgeService
from .agent_comm_service import AgentCommService
from .context_bundle import ContextBundle
from .context_service import ContextService
from .mcp_gateway import MCPGateway
from .memory_service import MemoryService

if TYPE_CHECKING:
    from houyi.infrastructure.observability import Span

    from .observation_service import ObservationService


@dataclass(slots=True)
class ExecutionContext:
    session_id: str
    execution: ExecutionIR
    plan: PlanIR
    run_settings: dict[str, Any] = field(default_factory=dict)
    context_service: ContextService | None = None
    memory_service: MemoryService | None = None
    rag_service: KnowledgeService | None = None
    mcp_gateway: MCPGateway | None = None
    agent_comm_service: AgentCommService | None = None
    context_bundle: ContextBundle | None = None
    root_span: Span | None = None
    observation_service: ObservationService | None = None
    tool_cache: dict[str, dict[str, Any]] | None = None

    def update_plan(self, plan: PlanIR) -> None:
        self.plan = plan
