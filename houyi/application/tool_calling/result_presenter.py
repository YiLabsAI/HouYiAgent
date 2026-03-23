"""Tool-call result presentation helpers."""

from __future__ import annotations

from typing import Any

from houyi.application.tool_calling.budget import MessageBudget
from houyi.application.tool_calling.runner_models import (
    _BlockedToolCallPresentationRequest,
    _ToolCallPresentationRequest,
)
from houyi.application.tool_calling.tool_results import ToolResultBuilder


class _ToolCallResultPresenter:
    """Build trace entries and tool messages for tool-call outcomes."""

    @staticmethod
    def _format_recovery_guidance(result: dict[str, Any]) -> str | None:
        raw = result.get("raw")
        if not isinstance(raw, dict):
            return None
        guidance = raw.get("recovery_guidance")
        if not isinstance(guidance, dict):
            return None
        lines: list[str] = []
        title = guidance.get("title")
        if isinstance(title, str) and title.strip():
            lines.append(f"Recovery: {title.strip()}")
        similar_tools = guidance.get("similar_tools")
        if isinstance(similar_tools, list) and similar_tools:
            lines.append("Similar tools: " + ", ".join(str(item) for item in similar_tools[:3]))
        required_fields = guidance.get("required_fields")
        if isinstance(required_fields, list) and required_fields:
            lines.append("Required fields: " + ", ".join(str(item) for item in required_fields[:5]))
        next_steps = guidance.get("next_steps")
        if isinstance(next_steps, list) and next_steps:
            lines.extend(f"- {step!s}" for step in next_steps[:3])
        return "\n".join(lines) if lines else None

    def _build_footer(
        self, request: _ToolCallPresentationRequest, result: dict[str, Any]
    ) -> str | None:
        footer_parts: list[str] = []
        duration_ms = request.duration_ms
        if isinstance(duration_ms, (int, float)):
            footer_parts.append(f"duration_ms={duration_ms:.2f}")
        payload_size = ToolResultBuilder.content_length(result)
        footer_parts.append(f"content_chars={payload_size}")
        metadata = result.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("cache_hit") is True:
            footer_parts.append("cache_hit=true")
        if request.parallel_group_id:
            footer_parts.append(f"parallel_group_id={request.parallel_group_id}")
        return " | ".join(footer_parts) if footer_parts else None

    def _present_result_content(
        self, request: _ToolCallPresentationRequest
    ) -> tuple[str, dict[str, Any]]:
        result = request.result
        presentation_meta: dict[str, Any] = {
            "is_binary": False,
            "error_detail_attached": False,
            "footer_attached": False,
            "recovery_guidance_attached": False,
            "result_artifact_candidate": False,
            "content_chars": ToolResultBuilder.content_length(result),
        }
        if ToolResultBuilder.is_binary_like_content(result):
            presentation_meta["is_binary"] = True
            return "Binary-like tool result omitted from inline expansion.", presentation_meta

        content = ToolResultBuilder.format(result)
        if ToolResultBuilder.is_error(result):
            error_detail = ToolResultBuilder.extract_error_detail(result)
            if error_detail:
                content = f"Error: {error_detail}\n\n{content}"
                presentation_meta["error_detail_attached"] = True
            recovery_guidance = self._format_recovery_guidance(result)
            if recovery_guidance:
                content = f"{content}\n\n{recovery_guidance}"
                presentation_meta["recovery_guidance_attached"] = True

        footer = self._build_footer(request, result)
        if footer:
            content = f"{content}\n\n[{footer}]"
            presentation_meta["footer_attached"] = True

        if presentation_meta["content_chars"] > request.tool_result_summary_max_chars:
            presentation_meta["result_artifact_candidate"] = True
        return content, presentation_meta

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
        presented_content, presentation_meta = self._present_result_content(request)
        result_meta = dict(request.result.get("metadata") or {})
        result_meta.update(
            {
                "content_chars": presentation_meta["content_chars"],
                "is_binary": presentation_meta["is_binary"],
                "error_detail_attached": presentation_meta["error_detail_attached"],
                "footer_attached": presentation_meta["footer_attached"],
                "recovery_guidance_attached": presentation_meta["recovery_guidance_attached"],
                "result_artifact_candidate": presentation_meta["result_artifact_candidate"],
            }
        )
        request.result["metadata"] = result_meta
        trace_entry: dict[str, object] = {
            "tool_name": request.tool_name,
            "requested_tool_name": request.requested_tool_name,
            "tool_call_id": request.tool_call_id,
            "round_index": request.round_index_value,
            "parallel_group_id": request.parallel_group_id,
            "duration_ms": request.duration_ms,
            "args": request.args,
            "result": request.result,
            "presentation": dict(result_meta),
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
            "content": presented_content,
            "metadata": {
                "round_index": request.round_index_value,
                "parallel_group_id": request.parallel_group_id,
                "duration_ms": request.duration_ms,
                "tool_args": request.args,
                "presentation": dict(result_meta),
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
            result_meta["result_summarized"] = True
            result_meta["result_summary_max_chars"] = request.tool_result_summary_max_chars
            result_meta["result_summary_max_items"] = request.tool_result_summary_max_items
            request.result["metadata"] = result_meta
            trace_entry["presentation"] = dict(result_meta)
            metadata = tool_message.get("metadata")
            if isinstance(metadata, dict):
                metadata["presentation"] = dict(result_meta)
        return trace_entry, tool_message


__all__ = ["_ToolCallResultPresenter"]
