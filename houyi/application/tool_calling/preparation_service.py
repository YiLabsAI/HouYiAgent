"""Preparation orchestrator for tool-call execution inputs."""

from __future__ import annotations

from typing import Any

from houyi.application.tool_calling.arg_coercion import coerce_args
from houyi.application.tool_calling.placeholder_resolver import PlaceholderResolver
from houyi.application.tool_calling.runner_models import (
    _PreparedToolCall,
    _ToolCallPreparationRequest,
)
from houyi.application.tool_calling.tool_results import ToolResultBuilder
from houyi.domain.skill.spec import SkillSpec


class _ToolCallPreparationService:
    """Prepare tool execution inputs before runtime dispatch."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def prepare(
        self,
        request: _ToolCallPreparationRequest,
    ) -> _PreparedToolCall | tuple[int, dict[str, Any], dict[str, Any], float]:
        """Resolve inputs, apply consent/hook policies, and build prepared call payload."""
        tool_name, tool_call_id, args, skill = self._resolve_tool_call_inputs(
            tool_call=request.tool_call,
            parsed_args=request.parsed_args,
            resolved_outputs=request.resolved_outputs,
            skills_by_name=request.skills_by_name,
        )
        requested_name = tool_name
        consent_rejection = await self._runner._preparation_policy_service.handle_consent_rejection(
            tool_name=tool_name,
            args=args,
            tool_call_id=tool_call_id,
            index=request.index,
            round_index_value=request.round_index_value,
            parallel_group_id=request.parallel_group_id,
            requested_tool_name=requested_name,
        )
        if consent_rejection is not None:
            return consent_rejection

        (
            hook_context,
            attempted_tool_name,
        ) = await self._runner._preparation_hook_service.apply_before_tool_hooks(
            tool_name=tool_name,
            args=args,
            skill=skill,
            tool_call_id=tool_call_id,
            tool_hooks=request.tool_hooks,
            allow_tool_replace=request.allow_tool_replace,
        )
        current_tool_name = hook_context["tool_name"]
        current_args = hook_context["args"]
        current_skill = hook_context["skill"]
        return _PreparedToolCall(
            requested_tool_name=requested_name,
            tool_name=current_tool_name,
            tool_call_id=tool_call_id,
            args=current_args,
            skill=current_skill,
            hook_context=hook_context,
            attempted_tool_name=attempted_tool_name,
            cache_key=self._runner._execution_service._build_tool_cache_key(
                current_tool_name,
                current_args,
                current_skill,
            ),
        )

    def _resolve_tool_call_inputs(
        self,
        *,
        tool_call: Any,
        parsed_args: dict[str, Any] | None,
        resolved_outputs: dict[str, Any] | None,
        skills_by_name: dict[str, SkillSpec],
    ) -> tuple[str | None, str | None, dict[str, Any], SkillSpec | None]:
        """Parse tool-call payload and resolve arguments into executable skill inputs."""
        tool_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        tool_name = tool_payload.get("name")
        tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
        args = (
            parsed_args
            if parsed_args is not None
            else ToolResultBuilder.parse_arguments(tool_payload.get("arguments"))
        )
        if resolved_outputs is not None:
            args = PlaceholderResolver.resolve(args, resolved_outputs)
            if tool_name:
                args = coerce_args(tool_name, args, resolved_outputs)
        skill = skills_by_name.get(tool_name) if tool_name else None
        return tool_name, tool_call_id, args, skill


__all__ = ["_ToolCallPreparationService"]
