"""Coordinator for tool-call caches and service creation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .tool_call_service import ToolCallService


class ToolCallCoordinator:
    """Encapsulates tool-call caches and service wiring."""

    def __init__(self) -> None:
        self.tool_call_cache: dict[str, dict[str, Any]] = {}
        self.llm_tool_call_cache: dict[str, Any] = {}

    def build_service(
        self,
        *,
        connection_manager: Any,
        record_llm_call: Callable[..., None],
    ) -> ToolCallService:
        """Create a ToolCallService with managed caches."""
        return ToolCallService(
            connection_manager=connection_manager,
            record_llm_call=record_llm_call,
            tool_call_cache=self.tool_call_cache,
            llm_tool_call_cache=self.llm_tool_call_cache,
        )


__all__ = [
    "ToolCallCoordinator",
]
