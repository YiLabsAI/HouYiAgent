import pytest

from houyi.adapters.mcp import MCPGateway


@pytest.mark.asyncio
async def test_gateway_returns_not_configured() -> None:
    gateway = MCPGateway()
    result = await gateway.invoke("tool.ref", {"x": 1})
    assert result == {"error": "mcp_not_configured"}
