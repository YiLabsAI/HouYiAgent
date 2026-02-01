"""Context service stub for execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context_bundle import ContextBundle
from .memory_service import MemoryService
from .rag_service import RAGService


@dataclass(slots=True)
class ContextService:
    memory_service: MemoryService
    rag_service: RAGService

    async def resolve(self, context: Any, query: str | None = None) -> ContextBundle:
        memory_items = await self.memory_service.retrieve(context, query)
        rag_chunks = await self.rag_service.search(context, query)
        run_settings = getattr(context, "run_settings", {}) if context else {}
        return ContextBundle(
            run_settings=run_settings,
            memory_items=memory_items,
            rag_chunks=rag_chunks,
        )
