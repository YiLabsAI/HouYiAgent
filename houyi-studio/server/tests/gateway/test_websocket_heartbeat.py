"""Tests for WebSocket heartbeat and session management."""

from __future__ import annotations

import pytest


class TestHeartbeatConstants:
    """Verify heartbeat/session constants are sensible."""

    def test_heartbeat_interval_within_proxy_timeout(self) -> None:
        """Heartbeat interval must be well below typical proxy idle timeouts (60-120s)."""
        from houyi_studio.server.gateway.app import _HEARTBEAT_INTERVAL_S

        assert _HEARTBEAT_INTERVAL_S <= 30, "Heartbeat interval too long for proxy idle timeouts"
        assert _HEARTBEAT_INTERVAL_S >= 5, "Heartbeat interval too aggressive"

    def test_client_timeout_is_multiple_of_heartbeat(self) -> None:
        """Client timeout should allow at least 2 missed pings before declaring dead."""
        from houyi_studio.server.gateway.app import _CLIENT_TIMEOUT_S, _HEARTBEAT_INTERVAL_S

        assert _CLIENT_TIMEOUT_S >= _HEARTBEAT_INTERVAL_S * 2, (
            "Client timeout should be at least 2x heartbeat interval"
        )

    def test_grace_period_covers_reconnect_backoff(self) -> None:
        """Grace period must be long enough for ReconnectingWebSocket exponential backoff."""
        from houyi_studio.server.gateway.app import _DISCONNECT_GRACE_S

        # ReconnectingWebSocket first retry is 500ms-2s; 15s covers many retry cycles
        assert _DISCONNECT_GRACE_S >= 10, "Grace period too short for reconnection"


class TestPongHandling:
    """Test that pong commands are recognized and not treated as unknown commands."""

    def test_pong_is_not_parsed_as_command(self) -> None:
        """parse_command should return None for pong (it's handled before parse_command)."""
        from houyi_studio.server.gateway.app import parse_command

        result = parse_command({"command_type": "pong"})
        assert result is None, "pong should not be parsed as a valid command"


class TestConnectionManagerSessionCount:
    """Test ConnectionManager session counting for grace period logic."""

    @pytest.mark.asyncio
    async def test_session_count_zero_after_disconnect(self) -> None:
        from houyi_studio.server.gateway.websocket import ConnectionManager

        class FakeWS:
            async def accept(self) -> None:
                pass

        manager = ConnectionManager()
        ws = FakeWS()
        await manager.connect(ws, "s1")  # type: ignore[arg-type]
        assert manager.get_session_count("s1") == 1

        manager.disconnect(ws)  # type: ignore[arg-type]
        assert manager.get_session_count("s1") == 0

    @pytest.mark.asyncio
    async def test_multiple_connections_same_session(self) -> None:
        from houyi_studio.server.gateway.websocket import ConnectionManager

        class FakeWS:
            async def accept(self) -> None:
                pass

        manager = ConnectionManager()
        ws1 = FakeWS()
        ws2 = FakeWS()
        await manager.connect(ws1, "s1")  # type: ignore[arg-type]
        await manager.connect(ws2, "s1")  # type: ignore[arg-type]
        assert manager.get_session_count("s1") == 2

        # Disconnect one — session still alive
        manager.disconnect(ws1)  # type: ignore[arg-type]
        assert manager.get_session_count("s1") == 1

        # Disconnect second — session gone
        manager.disconnect(ws2)  # type: ignore[arg-type]
        assert manager.get_session_count("s1") == 0
