"""Memory service stub for execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MemoryService:
    async def retrieve(self, _context: Any, _query: str | None = None) -> list[Any]:
        return []

    async def write(self, _context: Any, _items: list[Any]) -> None:
        return None
