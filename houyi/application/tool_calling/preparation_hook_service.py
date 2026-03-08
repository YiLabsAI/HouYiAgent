"""Hook-path helpers for tool-call preparation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from houyi.application.tool_calling.runner_models import _HookCtx
from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


class _ToolCallPreparationHookService:
    """Apply before-tool hooks and PRE_TOOL_USE notifications during preparation."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def apply_before_tool_hooks(
        self,
        *,
        tool_name: str | None,
        args: dict[str, Any],
        skill: SkillSpec | None,
        tool_call_id: str | None,
        tool_hooks: list[Any],
        allow_tool_replace: bool,
    ) -> tuple[_HookCtx, str | None]:
        hook_context: _HookCtx = {
            "tool_name": tool_name,
            "args": args,
            "skill": skill,
            "tool_call_id": tool_call_id,
        }
        attempted_tool_name: str | None = None

        if tool_name:
            await self.trigger_pre_tool_use_hook(tool_name, args, skill)

        for hook in tool_hooks:
            before_hook = getattr(hook, "before_tool_call", None)
            if before_hook is None:
                continue
            updated = await self._runner._execution_service._invoke_hook(before_hook, hook_context)
            if not isinstance(updated, dict):
                continue
            if "tool_name" in updated and updated["tool_name"] != hook_context["tool_name"]:
                attempted_tool_name = updated["tool_name"]
                if allow_tool_replace:
                    hook_context["tool_name"] = updated["tool_name"]
            if "args" in updated:
                hook_context["args"] = updated["args"]

        return hook_context, attempted_tool_name

    async def trigger_pre_tool_use_hook(
        self,
        tool_name: str,
        args: dict[str, Any],
        skill: SkillSpec | None,
    ) -> str | None:
        if not self._runner.skill_hooks_manager:
            return None
        from houyi.domain.skill.hooks import HookContext, HookEvent

        skill_hook_ctx = HookContext(
            tool_name=tool_name,
            tool_args=args,
            skill=skill,
            skill_name=skill.name if skill else None,
            cwd=Path.cwd(),
            skill_dir=skill.skill_dir if skill else None,
        )
        hook_result = await self._runner.skill_hooks_manager.trigger_hook(
            HookEvent.PRE_TOOL_USE,
            skill_hook_ctx,
            tool_name=tool_name,
        )
        if hook_result.output:
            logger.debug(
                "[ToolCallRunner] PreToolUse hook output: %s",
                hook_result.output[:100] if hook_result.output else None,
            )
            return hook_result.output
        return None


__all__ = ["_ToolCallPreparationHookService"]
