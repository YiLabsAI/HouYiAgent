from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from houyi.application.tool_calling.arg_coercion import _REGISTRY, register_arg_coercion
from houyi.application.tool_calling.preparation_service import _ToolCallPreparationService
from houyi.application.tool_calling.runner_models import (
    _PreparedToolCall,
    _ToolCallPreparationRequest,
)
from houyi.domain.skill.spec import SkillSpec


class _EmptyInput(BaseModel):
    pass


class _ValueInput(BaseModel):
    value: int


class _SimpleOutput(BaseModel):
    ok: bool = True


class _FakeExecutionService:
    def __init__(self) -> None:
        self.cache_key_calls: list[tuple[str | None, dict[str, Any], SkillSpec | None]] = []

    def _build_tool_cache_key(
        self,
        tool_name: str | None,
        args: dict[str, Any],
        skill: SkillSpec | None,
    ) -> str | None:
        self.cache_key_calls.append((tool_name, dict(args), skill))
        if tool_name is None:
            return None
        return f"cache::{tool_name}::{json.dumps(args, sort_keys=True)}"


class _FakePreparationPolicyService:
    def __init__(
        self, rejection: tuple[int, dict[str, Any], dict[str, Any], float] | None = None
    ) -> None:
        self.rejection = rejection
        self.calls: list[dict[str, Any]] = []

    async def handle_consent_rejection(
        self, **kwargs: Any
    ) -> tuple[int, dict[str, Any], dict[str, Any], float] | None:
        self.calls.append(kwargs)
        return self.rejection


class _FakePreparationHookService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.hook_context: dict[str, Any] | None = None
        self.attempted_tool_name: str | None = None

    async def apply_before_tool_hooks(self, **kwargs: Any) -> tuple[dict[str, Any], str | None]:
        self.calls.append(kwargs)
        if self.hook_context is not None:
            return self.hook_context, self.attempted_tool_name
        return (
            {
                "tool_name": kwargs["tool_name"],
                "args": kwargs["args"],
                "skill": kwargs["skill"],
                "tool_call_id": kwargs["tool_call_id"],
            },
            self.attempted_tool_name,
        )


class _FakeRunner:
    def __init__(
        self,
        *,
        rejection: tuple[int, dict[str, Any], dict[str, Any], float] | None = None,
    ) -> None:
        self._execution_service = _FakeExecutionService()
        self._preparation_policy_service = _FakePreparationPolicyService(rejection=rejection)
        self._preparation_hook_service = _FakePreparationHookService()


def _make_skill(name: str, input_schema: type[BaseModel] = _EmptyInput) -> SkillSpec:
    return SkillSpec(
        name=name,
        description=f"skill {name}",
        input_schema=input_schema,
        output_schema=_SimpleOutput,
        executor=lambda _: _SimpleOutput(ok=True),
    )


def _make_request(
    *,
    tool_call: dict[str, Any],
    parsed_args: dict[str, Any] | None,
    resolved_outputs: dict[str, Any] | None,
    skills_by_name: dict[str, SkillSpec],
    tool_hooks: list[Any] | None = None,
    allow_tool_replace: bool = False,
    index: int = 0,
    round_index_value: int | None = 1,
    parallel_group_id: str | None = None,
) -> _ToolCallPreparationRequest:
    return _ToolCallPreparationRequest(
        tool_call=tool_call,
        parsed_args=parsed_args,
        resolved_outputs=resolved_outputs,
        skills_by_name=skills_by_name,
        tool_hooks=tool_hooks or [],
        allow_tool_replace=allow_tool_replace,
        index=index,
        round_index_value=round_index_value,
        parallel_group_id=parallel_group_id,
    )


class TestToolCallPreparationService:
    @pytest.mark.asyncio
    async def test_prepare_builds_prepared_tool_call_from_resolved_inputs(self) -> None:
        skill = _make_skill("echo", _ValueInput)
        runner = _FakeRunner()
        service = _ToolCallPreparationService(runner)
        request = _make_request(
            tool_call={
                "id": "call_1",
                "function": {"name": "echo", "arguments": json.dumps({"value": 1})},
            },
            parsed_args={"value": 1},
            resolved_outputs=None,
            skills_by_name={"echo": skill},
            tool_hooks=[],
        )

        prepared = await service.prepare(request)

        assert isinstance(prepared, _PreparedToolCall)
        assert prepared.requested_tool_name == "echo"
        assert prepared.tool_name == "echo"
        assert prepared.tool_call_id == "call_1"
        assert prepared.args == {"value": 1}
        assert prepared.skill is skill
        assert prepared.hook_context == {
            "tool_name": "echo",
            "args": {"value": 1},
            "skill": skill,
            "tool_call_id": "call_1",
        }
        assert prepared.attempted_tool_name is None
        assert prepared.cache_key == 'cache::echo::{"value": 1}'
        assert runner._execution_service.cache_key_calls == [("echo", {"value": 1}, skill)]

    @pytest.mark.asyncio
    async def test_prepare_returns_policy_rejection_without_invoking_hook_service(self) -> None:
        rejection = (
            0,
            {"tool_name": "blocked"},
            {"role": "tool", "content": "blocked"},
            0.0,
        )
        skill = _make_skill("blocked")
        runner = _FakeRunner(rejection=rejection)
        service = _ToolCallPreparationService(runner)
        request = _make_request(
            tool_call={"id": "call_blocked", "function": {"name": "blocked", "arguments": "{}"}},
            parsed_args={},
            resolved_outputs=None,
            skills_by_name={"blocked": skill},
        )

        prepared = await service.prepare(request)

        assert prepared == rejection
        assert len(runner._preparation_policy_service.calls) == 1
        assert runner._preparation_hook_service.calls == []
        assert runner._execution_service.cache_key_calls == []

    @pytest.mark.asyncio
    async def test_prepare_uses_hook_service_outputs_for_final_payload(self) -> None:
        skill = _make_skill("tool1")
        replacement_skill = _make_skill("tool2")
        runner = _FakeRunner()
        runner._preparation_hook_service.hook_context = {
            "tool_name": "tool2",
            "args": {"from_hook": True},
            "skill": replacement_skill,
            "tool_call_id": "call_replace",
        }
        runner._preparation_hook_service.attempted_tool_name = "tool2"
        service = _ToolCallPreparationService(runner)
        request = _make_request(
            tool_call={"id": "call_replace", "function": {"name": "tool1", "arguments": "{}"}},
            parsed_args={},
            resolved_outputs=None,
            skills_by_name={"tool1": skill, "tool2": replacement_skill},
            allow_tool_replace=True,
        )

        prepared = await service.prepare(request)

        assert isinstance(prepared, _PreparedToolCall)
        assert prepared.requested_tool_name == "tool1"
        assert prepared.tool_name == "tool2"
        assert prepared.args == {"from_hook": True}
        assert prepared.skill is replacement_skill
        assert prepared.attempted_tool_name == "tool2"
        assert prepared.cache_key == 'cache::tool2::{"from_hook": true}'
        assert runner._execution_service.cache_key_calls == [
            ("tool2", {"from_hook": True}, replacement_skill)
        ]

    @pytest.mark.asyncio
    async def test_prepare_resolves_placeholders_and_coerces_args_before_policy_and_hooks(
        self,
    ) -> None:
        def _coerce_registered_echo(
            args: dict[str, Any], resolved_outputs: dict[str, Any]
        ) -> dict[str, Any]:
            _ = resolved_outputs
            return {"value": int(args["value"])}

        register_arg_coercion("registered_echo", _coerce_registered_echo)
        skill = _make_skill("registered_echo", _ValueInput)
        runner = _FakeRunner()
        service = _ToolCallPreparationService(runner)
        try:
            request = _make_request(
                tool_call={
                    "id": "call_resolve",
                    "function": {
                        "name": "registered_echo",
                        "arguments": json.dumps({"value": "$tool.seed.value"}),
                    },
                },
                parsed_args={"value": "$tool.seed.value"},
                resolved_outputs={"seed": {"value": "7"}},
                skills_by_name={"registered_echo": skill},
            )

            prepared = await service.prepare(request)

            assert isinstance(prepared, _PreparedToolCall)
            assert prepared.args == {"value": 7}
            assert runner._preparation_policy_service.calls[0]["args"] == {"value": 7}
            hook_call = runner._preparation_hook_service.calls[0]
            assert hook_call["args"] == {"value": 7}
            assert hook_call["tool_name"] == "registered_echo"
            assert prepared.cache_key == 'cache::registered_echo::{"value": 7}'
        finally:
            _REGISTRY.pop("registered_echo", None)

    @pytest.mark.asyncio
    async def test_resolve_tool_call_inputs_parses_arguments_when_parsed_args_missing(self) -> None:
        skill = _make_skill("echo", _ValueInput)
        service = _ToolCallPreparationService(_FakeRunner())

        tool_name, tool_call_id, args, resolved_skill = service._resolve_tool_call_inputs(
            tool_call={
                "id": "call_parse",
                "function": {"name": "echo", "arguments": json.dumps({"value": 3})},
            },
            parsed_args=None,
            resolved_outputs=None,
            skills_by_name={"echo": skill},
        )

        assert tool_name == "echo"
        assert tool_call_id == "call_parse"
        assert args == {"value": 3}
        assert resolved_skill is skill
