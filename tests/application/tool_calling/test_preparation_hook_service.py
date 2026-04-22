from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from houyi.application.tool_calling.preparation_hook_service import (
    _ToolCallPreparationHookService,
)
from houyi.domain.skill.hooks import HookEvent, HookType, SkillHook, SkillHooksManager
from houyi.domain.skill.spec import SkillSpec


class _EmptyInput(BaseModel):
    pass


class _SimpleOutput(BaseModel):
    ok: bool = True


@dataclass
class _HookResult:
    output: str | None = None


class _FakeExecutionService:
    async def _invoke_hook(self, hook: Any, hook_context: dict[str, Any]) -> Any:
        return await hook(dict(hook_context))


class _FakeRunner:
    def __init__(self, skill_hooks_manager: SkillHooksManager | None = None) -> None:
        self.skill_hooks_manager = skill_hooks_manager
        self._execution_service = _FakeExecutionService()


def _make_skill(name: str) -> SkillSpec:
    return SkillSpec(
        name=name,
        description=f"skill {name}",
        input_schema=_EmptyInput,
        output_schema=_SimpleOutput,
        executor=lambda _: _SimpleOutput(ok=True),
    )


class TestToolCallPreparationHookService:
    @pytest.mark.asyncio
    async def test_before_hooks_patch_args(self) -> None:
        class _PatchArgsHook:
            async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
                return {"args": {"patched": tool_call["args"]["value"] + 1}}

        skill = _make_skill("patch_args")
        service = _ToolCallPreparationHookService(_FakeRunner())

        hook_context, attempted_tool_name = await service.apply_before_tool_hooks(
            tool_name="patch_args",
            args={"value": 2},
            skill=skill,
            tool_call_id="call_patch",
            tool_hooks=[_PatchArgsHook()],
            allow_tool_replace=False,
        )

        assert attempted_tool_name is None
        assert hook_context["tool_name"] == "patch_args"
        assert hook_context["args"] == {"patched": 3}
        assert hook_context["skill"] is skill
        assert hook_context["tool_call_id"] == "call_patch"

    @pytest.mark.asyncio
    async def test_before_hooks_records_attempt(
        self,
    ) -> None:
        class _ReplaceHook:
            async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
                _ = tool_call
                return {"tool_name": "tool2"}

        skill = _make_skill("tool1")
        service = _ToolCallPreparationHookService(_FakeRunner())

        hook_context, attempted_tool_name = await service.apply_before_tool_hooks(
            tool_name="tool1",
            args={},
            skill=skill,
            tool_call_id="call_replace_blocked",
            tool_hooks=[_ReplaceHook()],
            allow_tool_replace=False,
        )

        assert attempted_tool_name == "tool2"
        assert hook_context["tool_name"] == "tool1"
        assert hook_context["args"] == {}
        assert hook_context["skill"] is skill

    @pytest.mark.asyncio
    async def test_before_hooks_replace_tool(self) -> None:
        class _ReplaceHook:
            async def before_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
                _ = tool_call
                return {"tool_name": "tool2", "args": {"from_hook": True}}

        original_skill = _make_skill("tool1")
        service = _ToolCallPreparationHookService(_FakeRunner())

        hook_context, attempted_tool_name = await service.apply_before_tool_hooks(
            tool_name="tool1",
            args={},
            skill=original_skill,
            tool_call_id="call_replace_allowed",
            tool_hooks=[_ReplaceHook()],
            allow_tool_replace=True,
        )

        assert attempted_tool_name == "tool2"
        assert hook_context["tool_name"] == "tool2"
        assert hook_context["args"] == {"from_hook": True}
        assert hook_context["skill"] is original_skill

    @pytest.mark.asyncio
    async def test_before_ignores_non_dict(self) -> None:
        class _NoopHook:
            async def before_tool_call(self, tool_call: dict[str, Any]) -> list[str]:
                _ = tool_call
                return ["ignored"]

        skill = _make_skill("tool1")
        service = _ToolCallPreparationHookService(_FakeRunner())

        hook_context, attempted_tool_name = await service.apply_before_tool_hooks(
            tool_name="tool1",
            args={"x": 1},
            skill=skill,
            tool_call_id="call_ignored",
            tool_hooks=[_NoopHook()],
            allow_tool_replace=True,
        )

        assert attempted_tool_name is None
        assert hook_context["tool_name"] == "tool1"
        assert hook_context["args"] == {"x": 1}

    @pytest.mark.asyncio
    async def test_trigger_passes_context(self) -> None:
        seen: list[dict[str, Any]] = []

        async def on_pre_tool_use(ctx: Any) -> dict[str, Any]:
            seen.append(
                {
                    "tool_name": ctx.tool_name,
                    "tool_args": dict(ctx.tool_args),
                    "skill_name": ctx.skill_name,
                }
            )
            return {"success": True, "output": "noted"}

        hooks = SkillHooksManager()
        hooks.register_hooks(
            SkillSpec(
                name="hooked",
                description="hooked skill hooks",
                input_schema=_EmptyInput,
                output_schema=_SimpleOutput,
                hooks=[
                    SkillHook(
                        event=HookEvent.PRE_TOOL_USE,
                        hook_type=HookType.HANDLER,
                        handler=on_pre_tool_use,
                    )
                ],
            )
        )

        skill = _make_skill("hooked")
        service = _ToolCallPreparationHookService(_FakeRunner(skill_hooks_manager=hooks))

        output = await service.trigger_pre_tool_use_hook("hooked", {}, skill)

        assert output == "noted"
        assert seen == [{"tool_name": "hooked", "tool_args": {}, "skill_name": "hooked"}]

    @pytest.mark.asyncio
    async def test_trigger_none_without_manager(self) -> None:
        service = _ToolCallPreparationHookService(_FakeRunner(skill_hooks_manager=None))

        output = await service.trigger_pre_tool_use_hook("hooked", {"x": 1}, _make_skill("hooked"))

        assert output is None
