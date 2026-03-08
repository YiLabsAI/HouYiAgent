"""Tool-calling event dispatch helpers."""

from __future__ import annotations

import contextlib
from typing import Any

from houyi.application.tool_calling.runner_models import _HookCtx
from houyi.application.tool_calling.tool_results import ToolResultBuilder


class _ToolCallEventDispatcher:
    """Dispatch tool-calling lifecycle events and hook callbacks."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def dispatch(self, name: str, attributes: dict[str, Any]) -> None:
        trace_manager = self._runner.trace_manager
        if not trace_manager:
            return
        span = getattr(trace_manager, "current_span", None)
        if span is None:
            return
        with contextlib.suppress(Exception):
            span.add_event(name, attributes)

    def emit_usage_blocked(
        self,
        *,
        tool_call_id: str | None,
        tool_name: str,
        reason: str,
    ) -> None:
        self.dispatch(
            "ToolUsageBlocked",
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "parallel_group_id": None,
                "reason": reason,
            },
        )

    async def emit_usage_outcome(
        self,
        *,
        tool_hooks: list[Any],
        hook_context: _HookCtx,
        result: dict[str, Any],
        tool_call_id: str | None,
        tool_name: str | None,
        requested_tool_name: str | None,
        parallel_group_id: str | None,
        cache_hit_for_reporting: bool,
        cache_key: str | None,
        latency_ms: Any,
    ) -> None:
        if ToolResultBuilder.is_error(result):
            self.dispatch(
                "ToolUsageError",
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "requested_tool_name": requested_tool_name,
                    "parallel_group_id": parallel_group_id,
                    "error": result.get("raw"),
                    "latency_ms": latency_ms,
                },
            )
            hook_name = "on_tool_error"
        else:
            self.dispatch(
                "ToolUsageFinished",
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "requested_tool_name": requested_tool_name,
                    "parallel_group_id": parallel_group_id,
                    "result": result.get("raw"),
                    "cache_hit": cache_hit_for_reporting,
                    "cache_key": cache_key,
                    "latency_ms": latency_ms,
                },
            )
            hook_name = "after_tool_call"

        for hook in tool_hooks:
            callback = getattr(hook, hook_name, None)
            if callback is not None:
                await self._runner._execution_service._invoke_hook(callback, hook_context, result)


__all__ = ["_ToolCallEventDispatcher"]
