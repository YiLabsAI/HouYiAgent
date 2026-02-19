"""Context service stub for execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..rag import KnowledgeService
from .context_bundle import ContextBundle
from .memory_service import MemoryService


@dataclass(slots=True)
class ContextService:
    memory_service: MemoryService
    rag_service: KnowledgeService

    async def resolve(self, context: Any, query: str | None = None) -> ContextBundle:
        memory_items = await self.memory_service.retrieve(context, query)
        rag_chunks: list[Any] = []
        if query and hasattr(self.rag_service, "search_knowledge"):
            try:
                result = await self.rag_service.search_knowledge(query)
                rag_chunks = result.get("results", []) if isinstance(result, dict) else []
            except Exception:
                pass
        run_settings = getattr(context, "run_settings", {}) if context else {}
        return ContextBundle(
            run_settings=run_settings,
            memory_items=memory_items,
            rag_chunks=rag_chunks,
        )
