import pytest

from houyi.adapters.mcp import MCPGateway


@pytest.mark.asyncio
async def test_mcp_gateway_default_returns_not_configured_error() -> None:
    gateway = MCPGateway()
    result = await gateway.invoke("tool.ref", {"x": 1})
    assert result == {"error": "mcp_not_configured"}
