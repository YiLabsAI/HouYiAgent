"""MCP gateway stub for execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MCPGateway:
    async def invoke(self, _tool_ref: Any, _args: dict[str, Any]) -> dict[str, Any]:
        return {"error": "mcp_not_configured"}
