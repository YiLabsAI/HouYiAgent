from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class EvolutionCursorStore(Protocol):
    def get_cursor(self, consumer: str) -> int: ...

    def set_cursor(self, consumer: str, cursor: int) -> None: ...


@dataclass(slots=True)
class InMemoryEvolutionCursorStore:
    _cursors: dict[str, int] = field(default_factory=dict)

    def get_cursor(self, consumer: str) -> int:
        return self._cursors.get(consumer, 0)

    def set_cursor(self, consumer: str, cursor: int) -> None:
        if cursor < 0:
            raise ValueError("cursor must be >= 0")
        self._cursors[consumer] = cursor


class DurableProviderNotConfiguredError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DurableProviderConfig:
    provider_name: str
    connection_uri: str


def require_durable_provider(config: DurableProviderConfig) -> None:
    raise DurableProviderNotConfiguredError(
        f"durable evolution provider '{config.provider_name}' is not configured: "
        f"{config.connection_uri}"
    )
