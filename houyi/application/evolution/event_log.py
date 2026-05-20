from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from houyi.application.evolution.events import EvolutionEvent


class EvolutionEventLog(Protocol):
    def append(self, event: EvolutionEvent) -> None: ...

    def read_since(
        self,
        cursor: int,
        *,
        limit: int | None = None,
    ) -> tuple[list[EvolutionEvent], int]: ...


@dataclass(slots=True)
class InMemoryEvolutionEventLog:
    _events: list[EvolutionEvent] = field(default_factory=list)

    def append(self, event: EvolutionEvent) -> None:
        self._events.append(event)

    def read_since(
        self, cursor: int, *, limit: int | None = None
    ) -> tuple[list[EvolutionEvent], int]:
        if cursor < 0:
            raise ValueError("cursor must be >= 0")
        end = len(self._events) if limit is None else min(len(self._events), cursor + limit)
        return list(self._events[cursor:end]), end

    def __len__(self) -> int:
        return len(self._events)
