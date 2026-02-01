"""Coordinator for tool-call caches and service creation."""

from __future__ import annotations


class ToolCallCoordinator:  # noqa: F401
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "ToolCallCoordinator is not available in this build. "
            "Migrate to houyi.execution.tool_call_runner.ToolCallRunner "
            "or houyi.execution.tool_call_runner_service.ToolCallRunnerService."
        )


__all__ = [
    "ToolCallCoordinator",
]
