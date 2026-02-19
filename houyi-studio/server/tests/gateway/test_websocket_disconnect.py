from __future__ import annotations

import pytest


class _DummyWebSocket:
    async def receive_text(self) -> str:
        raise RuntimeError('WebSocket is not connected. Need to call "accept" first.')


@pytest.mark.asyncio
async def test_receive_command_runtimeerror_returns_none() -> None:
    """Regression: websocket receive_command should treat RuntimeError as closed connection."""

    from houyi_studio.server.gateway.websocket import ConnectionManager

    manager = ConnectionManager()
    result = await manager.receive_command(_DummyWebSocket())  # type: ignore[arg-type]
    assert result is None
