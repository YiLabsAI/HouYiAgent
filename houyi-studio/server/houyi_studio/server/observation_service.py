"""Observation service for publishing events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .event_bus import EventBus


@dataclass(slots=True)
class ObservationService:
    """Dispatch events to event bus and connection manager."""

    connection_manager: Any
    event_bus: EventBus

    async def emit(self, event: Any) -> None:
        await self.event_bus.publish(event)
        await self.connection_manager.send_event(event.session_id, event)
