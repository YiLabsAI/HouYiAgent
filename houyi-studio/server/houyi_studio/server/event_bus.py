"""Lightweight in-process event bus."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any


class EventBus:
    """Minimal event bus for in-process publish/subscribe."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Any], Awaitable[None] | None]] = []

    def subscribe(self, handler: Callable[[Any], Awaitable[None] | None]) -> None:
        self._subscribers.append(handler)

    async def publish(self, event: Any) -> None:
        for handler in list(self._subscribers):
            result = handler(event)
            if inspect.isawaitable(result):
                await result
