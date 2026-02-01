"""Agent communication service stub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AgentCommService:
    async def send(self, _target_agent_id: str, _message: str) -> None:
        return None

    async def broadcast(self, _channel: str, _message: str) -> None:
        return None
