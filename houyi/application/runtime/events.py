"""Typed event emitter for agent runtime observability."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentEventType(str, Enum):
    """Event types emitted during agent execution."""

    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    LLM_STARTED = "llm.started"
    LLM_CHUNK = "llm.chunk"
    LLM_COMPLETED = "llm.completed"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    TEAM_AGENT_SPAWNED = "team_agent.spawned"
    TEAM_AGENT_COMPLETED = "team_agent.completed"
    PROGRESS = "progress"


class AgentEvent(BaseModel):
    """Structured event emitted during agent execution.

    Aligns with A2A tasks/sendSubscribe SSE event model.
    """

    event_type: AgentEventType
    agent_id: str = ""
    agent_name: str = ""
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


EventHandler = Callable[[AgentEvent], Coroutine[Any, Any, None]]


class EventEmitter:
    """Async-safe typed event emitter with multi-listener support.

    Listeners are invoked concurrently via asyncio.gather and never
    block the caller—exceptions in individual handlers are logged but
    do not propagate.

    Example::

        emitter = EventEmitter()
        emitter.on(AgentEventType.AGENT_STARTED, my_handler)
        await emitter.emit(AgentEvent(event_type=AgentEventType.AGENT_STARTED))
    """

    def __init__(self) -> None:
        self._handlers: dict[AgentEventType, list[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: list[EventHandler] = []

    def on(self, event_type: AgentEventType, handler: EventHandler) -> None:
        """Register *handler* for a specific event type."""
        self._handlers[event_type].append(handler)

    def on_any(self, handler: EventHandler) -> None:
        """Register *handler* for **all** event types (wildcard)."""
        self._wildcard_handlers.append(handler)

    def off(self, event_type: AgentEventType, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        handlers = self._handlers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    def off_any(self, handler: EventHandler) -> None:
        """Remove a previously registered wildcard handler."""
        if handler in self._wildcard_handlers:
            self._wildcard_handlers.remove(handler)

    async def emit(self, event: AgentEvent) -> None:
        """Emit *event* to all matching listeners concurrently.

        Handlers that raise are logged and suppressed so that a single
        broken listener never disrupts the runtime pipeline.
        """
        targets = [*self._handlers.get(event.event_type, []), *self._wildcard_handlers]
        if not targets:
            return
        results = await asyncio.gather(
            *(self._safe_call(h, event) for h in targets),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("Event handler error for %s: %s", event.event_type, r)

    async def emit_sync(self, event: AgentEvent) -> None:
        """Emit *event* sequentially—useful for ordered side-effects."""
        for h in [*self._handlers.get(event.event_type, []), *self._wildcard_handlers]:
            try:
                await h(event)
            except Exception:
                logger.warning("Event handler error for %s", event.event_type, exc_info=True)

    @staticmethod
    async def _safe_call(handler: EventHandler, event: AgentEvent) -> None:
        await handler(event)
