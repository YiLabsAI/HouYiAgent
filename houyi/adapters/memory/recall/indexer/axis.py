"""IndexAxis protocol + query / result value types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AxisQuery:
    axis: str
    key: str
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AxisResult:
    axis: str
    matched_ids: frozenset[str]
    total_scanned: int = 0
    cost_ms: float = 0.0


class IndexAxis(Protocol):
    @property
    def name(self) -> str: ...

    async def query(self, q: AxisQuery, *, deadline_ms: int | None = None) -> AxisResult: ...


__all__ = ["AxisQuery", "AxisResult", "IndexAxis"]
