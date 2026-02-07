"""WebSocket connection manager."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from .events import ServerEvent
from .logging_config import truncate_payload

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for console sessions.

    Handles:
    - Connection lifecycle (connect, disconnect)
    - Broadcasting events to session connections
    - Receiving commands from clients
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        # Map session_id -> list of WebSocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}

        # Map connection -> session_id for cleanup
        self.connection_sessions: dict[WebSocket, str] = {}

        # Per-session counter for dropped events (rate-limit log spam)
        self._drop_counts: dict[str, int] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            session_id: Session identifier
        """
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = []

        # Log summary of events dropped while disconnected
        dropped = self._drop_counts.pop(session_id, 0)
        if dropped:
            logger.info(
                "[WebSocket] Session %s reconnected, %d events were dropped while disconnected",
                session_id,
                dropped,
            )

        self.active_connections[session_id].append(websocket)
        self.connection_sessions[websocket] = session_id

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        session_id = self.connection_sessions.get(websocket)
        if session_id and session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)

            # Clean up empty session
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

        if websocket in self.connection_sessions:
            del self.connection_sessions[websocket]

    async def send_event(
        self,
        session_id: str,
        event: ServerEvent,
    ) -> None:
        """Send an event to all connections in a session.

        Args:
            session_id: Target session
            event: Event to send
        """
        if session_id not in self.active_connections:
            count = self._drop_counts.get(session_id, 0)
            if count == 0:
                logger.warning(
                    "[WebSocket] No active connections for session %s, dropping event %s (further drops will be silent)",
                    session_id,
                    getattr(event, "event_type", "?"),
                )
            self._drop_counts[session_id] = count + 1
            return

        # Serialize event
        event_json = event.model_dump_json()
        logger.debug(
            "[WebSocket] Sending event type=%s to session=%s (%d connections)",
            getattr(event, "event_type", "?"),
            session_id,
            len(self.active_connections[session_id]),
        )

        # Send to all connections in session
        disconnected = []
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_text(event_json)
            except Exception:
                # Connection is dead, mark for removal
                disconnected.append(connection)

        # Clean up dead connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_event(self, event: ServerEvent) -> None:
        """Broadcast an event to all sessions.

        Args:
            event: Event to broadcast
        """
        for session_id in list(self.active_connections.keys()):
            await self.send_event(session_id, event)

    async def receive_command(self, websocket: WebSocket) -> dict[str, Any] | None:
        """Receive a command from a WebSocket connection.

        Args:
            websocket: WebSocket connection

        Returns:
            Parsed command dict, or None if connection closed
        """
        try:
            data = await websocket.receive_text()
            logger = logging.getLogger(__name__)
            logger.debug("[WebSocket] Received raw data: %s", truncate_payload(data))
            parsed = json.loads(data)
            logger.debug("[WebSocket] Parsed command type: %s", parsed.get("command_type"))
            return parsed
        except WebSocketDisconnect:
            return None
        except RuntimeError:
            return None
        except json.JSONDecodeError as e:
            logger.error("[WebSocket] JSON decode error: %s", e)
            return None

    def get_session_count(self, session_id: str) -> int:
        """Get number of active connections for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(session_id, []))


# Global connection manager instance
connection_manager = ConnectionManager()


__all__ = [
    "ConnectionManager",
    "connection_manager",
]
