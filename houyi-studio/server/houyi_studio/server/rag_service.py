"""RAG service stub for execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RAGService:
    async def search(self, _context: Any, _query: str | None = None) -> list[Any]:
        return []
