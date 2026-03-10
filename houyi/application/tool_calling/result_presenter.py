"""Tool-call result presentation helpers."""

from __future__ import annotations

from houyi.application.tool_calling.budget import MessageBudget
from houyi.application.tool_calling.runner_models import (
    _BlockedToolCallPresentationRequest,
    _ToolCallPresentationRequest,
)
from houyi.application.tool_calling.tool_results import ToolResultBuilder


class _ToolCallResultPresenter:
    """Build trace entries and tool messages for tool-call outcomes."""

    def build_blocked_trace_and_message(
        self,
        request: _BlockedToolCallPresentationRequest,
    ) -> tuple[dict[str, object], dict[str, object]]:
        error_result = ToolResultBuilder.build(
            {
                "error": request.error_code,
                "message": request.message,
            },
            call_id=request.tool_call_id,
            metadata={"tool_name": request.tool_name, "policy_blocked": True},
        )
        trace_entry: dict[str, object] = {
            "tool_name": request.tool_name,
            "requested_tool_name": request.tool_name,
            "tool_call_id": request.tool_call_id,
            "args": request.args,
            "result": error_result,
            "policy_blocked": True,
            "block_reason": request.block_reason,
        }
        tool_message: dict[str, object] = {
            "role": "tool",
            "tool_call_id": request.tool_call_id,
            "name": request.tool_name,
            "content": ToolResultBuilder.format(error_result),
        }
        return trace_entry, tool_message

    def build_trace_and_message(
        self,
        request: _ToolCallPresentationRequest,
    ) -> tuple[dict[str, object], dict[str, object]]:
        trace_entry: dict[str, object] = {
            "tool_name": request.tool_name,
            "requested_tool_name": request.requested_tool_name,
            "tool_call_id": request.tool_call_id,
            "round_index": request.round_index_value,
            "parallel_group_id": request.parallel_group_id,
            "duration_ms": request.duration_ms,
            "args": request.args,
            "result": request.result,
            "tool_override": (
                {
                    "from": request.requested_tool_name,
                    "to": request.attempted_tool_name,
                    "allowed": request.allow_tool_replace,
                    "applied": request.allow_tool_replace
                    and request.attempted_tool_name != request.requested_tool_name,
                }
                if request.attempted_tool_name
                else None
            ),
        }
        tool_message: dict[str, object] = {
            "role": "tool",
            "tool_call_id": request.tool_call_id,
            "name": request.tool_name,
            "content": ToolResultBuilder.format(request.result),
            "metadata": {
                "round_index": request.round_index_value,
                "parallel_group_id": request.parallel_group_id,
                "duration_ms": request.duration_ms,
            },
        }
        if not request.tool_result_summary_enabled:
            return trace_entry, tool_message

        summarized_content, summarized = MessageBudget.summarize_tool_result(
            str(tool_message["content"]),
            max_chars=request.tool_result_summary_max_chars,
            max_items=request.tool_result_summary_max_items,
        )
        if summarized:
            tool_message["content"] = summarized_content
            result_meta = dict(request.result.get("metadata") or {})
            result_meta["result_summarized"] = True
            result_meta["result_summary_max_chars"] = request.tool_result_summary_max_chars
            result_meta["result_summary_max_items"] = request.tool_result_summary_max_items
            request.result["metadata"] = result_meta
        return trace_entry, tool_message


__all__ = ["_ToolCallResultPresenter"]
