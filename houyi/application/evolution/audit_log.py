"""Daemon audit log primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AuditEntry:
    consumer: str
    action: str
    cursor_before: int
    cursor_after: int
    events_consumed: int
    skipped: bool
    reason: str
    promotion_level: str | None = None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


class EvolutionAuditLog(Protocol):
    def append_audit(self, entry: AuditEntry) -> None: ...

    def read_audit(
        self,
        *,
        consumer: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEntry]: ...


@dataclass(slots=True)
class InMemoryEvolutionAuditLog:
    _entries: list[AuditEntry] = field(default_factory=list)

    def append_audit(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    def read_audit(
        self,
        *,
        consumer: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEntry]:
        rows: list[AuditEntry] = self._entries
        if consumer is not None:
            rows = [entry for entry in rows if entry.consumer == consumer]
        if limit is not None:
            rows = rows[-limit:]
        return list(rows)

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "AuditEntry",
    "EvolutionAuditLog",
    "InMemoryEvolutionAuditLog",
]
