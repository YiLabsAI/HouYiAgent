"""Lifecycle and preprocessing collaborators for tool-calling runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


class _ToolCallLifecycleService:
    """Handle session lifecycle hooks, router filtering, and preprocessors."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def run_preprocessors(self, preprocessors: list[Any], messages: list[Any]) -> list[Any]:
        """Run preprocessors before the first LLM call and inject outputs into messages."""
        from houyi.domain.skill.preprocessor import PreprocessorPipeline

        pipeline = PreprocessorPipeline(preprocessors)
        try:
            pp_results = await pipeline.run()
            messages = pipeline.inject(messages, pp_results)
            logger.debug(
                "Preprocessors executed: %d total, %d successful",
                len(pp_results),
                sum(1 for r in pp_results if r.success),
            )
        except Exception:
            logger.warning("Preprocessor pipeline error (non-fatal)", exc_info=True)
        return messages

    async def trigger_session_start_hook(
        self,
        max_rounds: int,
        tool_count: int,
        skill_count: int,
    ) -> None:
        """Trigger SessionStart hook with run-level counts."""
        if not self._runner.skill_hooks_manager:
            return
        from houyi.domain.skill.hooks import HookContext, HookEvent

        session_ctx = HookContext(
            tool_name="__session__",
            tool_args={
                "max_rounds": max_rounds,
                "tool_count": tool_count,
                "skill_count": skill_count,
            },
        )
        try:
            await self._runner.skill_hooks_manager.trigger_hook(
                HookEvent.SESSION_START, session_ctx
            )
        except Exception:
            logger.debug("SessionStart hook error (non-fatal)", exc_info=True)

    def apply_tool_router(
        self,
        skills: list[SkillSpec],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter available tools using ToolRouter restrictions."""
        from houyi.domain.skill.tool_router import ToolRouter

        tool_router = ToolRouter(skills, self._runner.policy_enforcer)
        if not tool_router.has_restrictions:
            return tools
        original_count = len(tools)
        filtered_tools = tool_router.filter_tools(tools)
        logger.debug(
            "ToolRouter: filtered %d → %d tools",
            original_count,
            len(filtered_tools),
        )
        return filtered_tools

    async def trigger_post_tool_use_hook(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        skill: SkillSpec | None,
    ) -> None:
        """Trigger PostToolUse hook after tool result is produced."""
        if not self._runner.skill_hooks_manager:
            return
        from houyi.domain.skill.hooks import HookContext, HookEvent

        skill_hook_ctx = HookContext(
            tool_name=tool_name,
            tool_args=args,
            tool_result=result.get("raw"),
            skill=skill,
            skill_name=skill.name if skill else None,
            cwd=Path.cwd(),
            skill_dir=skill.skill_dir if skill else None,
        )
        post_hook_result = await self._runner.skill_hooks_manager.trigger_hook(
            HookEvent.POST_TOOL_USE,
            skill_hook_ctx,
            tool_name=tool_name,
        )
        if post_hook_result.output:
            logger.debug(
                "[ToolCallRunner] PostToolUse hook output: %s",
                post_hook_result.output[:100] if post_hook_result.output else None,
            )

    async def trigger_stop_hook(self, tool_trace: list[dict[str, Any]]) -> None:
        """Trigger Stop hook at the end of a tool-calling session."""
        if not self._runner.skill_hooks_manager:
            return
        from houyi.domain.skill.hooks import HookContext, HookEvent

        stop_ctx = HookContext(
            tool_name="__session__",
            tool_args={"tool_trace_length": len(tool_trace)},
        )
        try:
            await self._runner.skill_hooks_manager.trigger_hook(HookEvent.STOP, stop_ctx)
        except Exception:
            logger.debug("Stop hook error (non-fatal)", exc_info=True)


__all__ = ["_ToolCallLifecycleService"]
