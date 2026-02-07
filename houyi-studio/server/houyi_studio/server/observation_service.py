"""Observation service for publishing events."""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from typing import Any

from .event_bus import EventBus

logger = logging.getLogger(__name__)

# Max buffered span events per session to prevent unbounded memory growth
_MAX_SPAN_BUFFER_PER_SESSION = 2000


@dataclass(slots=True)
class ObservationService:
    """Dispatch events to event bus and connection manager."""

    connection_manager: Any
    event_bus: EventBus
    # Per-session ring buffer of SpanUpdateEvent for replay on reconnect
    _span_buffers: dict[str, collections.deque] = field(default_factory=dict)

    async def emit(self, event: Any) -> None:
        event_type = getattr(event, "event_type", "?")
        session_id = getattr(event, "session_id", "?")
        et_str = str(event_type).lower()
        if "streaming_output" in et_str or "span_update" in et_str or "node_status" in et_str:
            logger.debug("[ObservationService] emit: type=%s session=%s", event_type, session_id)
        else:
            logger.info("[ObservationService] emit: type=%s session=%s", event_type, session_id)
        await self.event_bus.publish(event)
        await self.connection_manager.send_event(event.session_id, event)
        # Buffer span events for replay on reconnect
        if getattr(event, "event_type", None) and str(event.event_type).endswith("span_update"):
            sid = event.session_id
            if sid not in self._span_buffers:
                self._span_buffers[sid] = collections.deque(maxlen=_MAX_SPAN_BUFFER_PER_SESSION)
            self._span_buffers[sid].append(event)

    def get_buffered_spans(self, session_id: str) -> list[Any]:
        """Return buffered span events for a session (for replay on reconnect)."""
        buf = self._span_buffers.get(session_id)
        return list(buf) if buf else []

    def clear_session_buffer(self, session_id: str) -> None:
        """Clear buffered span events for a session."""
        self._span_buffers.pop(session_id, None)
