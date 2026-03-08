"""Skill hooks system for lifecycle management."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """Hook event types (compatible with Claude Code)."""

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    SESSION_START = "SessionStart"
    PRE_EXECUTION = "PreExecution"
    POST_EXECUTION = "PostExecution"


class HookType(str, Enum):
    """Hook execution type."""

    COMMAND = "command"
    HANDLER = "handler"


@dataclass
class SkillHook:
    """Skill hook definition."""

    event: HookEvent
    matcher: str | None = None
    hook_type: HookType = HookType.HANDLER
    command: str | None = None
    handler_path: str | None = None
    handler: Callable[..., Any] | None = None

    def matches_tool(self, tool_name: str) -> bool:
        if not self.matcher:
            return True
        try:
            return bool(re.search(self.matcher, tool_name))
        except re.error:
            logger.warning("Invalid matcher regex: %s", self.matcher)
            return False


@dataclass
class HookContext:
    """Context passed to hook handlers."""

    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    skill: Any = None
    skill_name: str | None = None
    execution_id: str | None = None
    session_id: str | None = None
    cwd: Path | None = None
    skill_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    output: str | None = None


@dataclass
class HookResult:
    """Result from hook execution."""

    success: bool = True
    output: str | None = None
    error: str | None = None
    exit_code: int = 0
    should_block: bool = False
    inject_to_prompt: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillHooksManager:
    """Manages skill hooks registration and execution."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[SkillHook]] = {event: [] for event in HookEvent}
        self._skill_hooks: dict[str, list[SkillHook]] = {}

    def register_hooks(self, skill: SkillSpec) -> None:
        hooks = getattr(skill, "hooks", None)
        if not hooks:
            return

        skill_hooks: list[SkillHook] = []
        for hook in hooks:
            self._hooks[hook.event].append(hook)
            skill_hooks.append(hook)
            logger.debug(
                "Registered hook: skill=%s event=%s matcher=%s",
                skill.name,
                hook.event.value,
                hook.matcher,
            )

        self._skill_hooks[skill.name] = skill_hooks

    def unregister_hooks(self, skill_name: str) -> None:
        hooks = self._skill_hooks.pop(skill_name, [])
        for hook in hooks:
            with contextlib.suppress(ValueError):
                self._hooks[hook.event].remove(hook)
        logger.debug("Unregistered hooks for skill: %s", skill_name)

    async def trigger_hook(
        self,
        event: HookEvent,
        context: HookContext,
        tool_name: str | None = None,
    ) -> HookResult:
        hooks = self._hooks.get(event, [])
        if not hooks:
            return HookResult()

        combined_output: list[str] = []
        should_block = False

        for hook in hooks:
            if tool_name and not hook.matches_tool(tool_name):
                continue

            try:
                result = await self._execute_hook(hook, context)
                if result.output:
                    combined_output.append(result.output)
                if result.should_block:
                    should_block = True
                if not result.success:
                    logger.warning(
                        "Hook failed: event=%s error=%s",
                        event.value,
                        result.error,
                    )
            except Exception as e:
                logger.error("Hook execution failed: event=%s error=%s", event.value, e)

        return HookResult(
            success=True,
            output="\n".join(combined_output) if combined_output else None,
            should_block=should_block,
            inject_to_prompt=bool(combined_output),
        )

    async def _execute_hook(self, hook: SkillHook, context: HookContext) -> HookResult:
        if hook.hook_type == HookType.COMMAND:
            return await self._execute_command_hook(hook, context)
        if hook.hook_type == HookType.HANDLER:
            return await self._execute_handler_hook(hook, context)
        return HookResult(success=False, error=f"Unknown hook type: {hook.hook_type}")

    async def _execute_command_hook(self, hook: SkillHook, context: HookContext) -> HookResult:
        if not hook.command:
            return HookResult(success=False, error="No command specified")

        env = os.environ.copy()
        if context.skill_dir:
            env["SKILL_DIR"] = str(context.skill_dir)
            env["CLAUDE_PLUGIN_ROOT"] = str(context.skill_dir)
        if context.cwd:
            env["CWD"] = str(context.cwd)
            env["WORKSPACE"] = str(context.cwd)

        cwd = context.cwd or context.skill_dir or Path.cwd()

        try:
            process = await asyncio.create_subprocess_shell(
                hook.command,
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)

            exit_code = process.returncode or 0
            output = stdout.decode("utf-8", errors="replace").strip()
            error_output = stderr.decode("utf-8", errors="replace").strip()
            should_block = hook.event == HookEvent.STOP and exit_code != 0

            return HookResult(
                success=exit_code == 0,
                output=output if output else None,
                error=error_output if error_output and exit_code != 0 else None,
                exit_code=exit_code,
                should_block=should_block,
                inject_to_prompt=bool(output),
            )
        except TimeoutError:
            return HookResult(success=False, error="Command timed out")
        except Exception as e:
            return HookResult(success=False, error=str(e))

    async def _execute_handler_hook(self, hook: SkillHook, context: HookContext) -> HookResult:
        handler = hook.handler

        if not handler and hook.handler_path:
            handler = self._load_handler(hook.handler_path)
            if handler:
                hook.handler = handler

        if not handler:
            return HookResult(success=False, error=f"No handler found: {hook.handler_path}")

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(context)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, handler, context)

            if isinstance(result, HookResult):
                return result
            if isinstance(result, str):
                return HookResult(success=True, output=result, inject_to_prompt=True)
            if isinstance(result, dict):
                return HookResult(
                    success=result.get("success", True),
                    output=result.get("output"),
                    error=result.get("error"),
                    should_block=result.get("should_block", False),
                    inject_to_prompt=result.get("inject_to_prompt", False),
                )
            return HookResult(success=True)

        except Exception as e:
            logger.exception("Handler execution failed: %s", hook.handler_path)
            return HookResult(success=False, error=str(e))

    def _load_handler(self, handler_path: str) -> Callable[..., Any] | None:
        try:
            if ":" in handler_path:
                module_path, func_name = handler_path.rsplit(":", 1)
            else:
                module_path, func_name = handler_path.rsplit(".", 1)

            import importlib

            module = importlib.import_module(module_path)
            return getattr(module, func_name, None)
        except (ValueError, ImportError, AttributeError) as e:
            logger.warning("Failed to load handler %s: %s", handler_path, e)
            return None

    def get_registered_hooks(self, event: HookEvent | None = None) -> list[SkillHook]:
        if event:
            return list(self._hooks.get(event, []))
        return [hook for hooks in self._hooks.values() for hook in hooks]

    def clear(self) -> None:
        self._hooks = {event: [] for event in HookEvent}
        self._skill_hooks.clear()


DEFAULT_HOOKS_MANAGER = SkillHooksManager()

__all__ = [
    "DEFAULT_HOOKS_MANAGER",
    "HookContext",
    "HookEvent",
    "HookResult",
    "HookType",
    "SkillHook",
    "SkillHooksManager",
]
