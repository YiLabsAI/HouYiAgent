"""Cross-module integration tests for ToolCallRunner ownership paths.

These tests cover real collaboration across policy, consent, hooks, metrics,
and ToolCallRunner orchestration while keeping placement aligned with the
tool_calling integration ownership.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from houyi.application.tool_calling.runner import ToolCallRunner
from houyi.domain.skill.consent import (
    ConsentManager,
    InMemoryConsentStore,
    PolicyBasedConsentHandler,
)
from houyi.domain.skill.hooks import (
    HookEvent,
    HookType,
    SkillHook,
    SkillHooksManager,
)
from houyi.domain.skill.metrics import MetricsStore
from houyi.domain.skill.policy import (
    InvocationPolicy,
    ModelAutoInvoke,
    Permissions,
    PolicyEnforcer,
    SideEffect,
)
from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec


class EmptyInput(BaseModel):
    pass


class SimpleOutput(BaseModel):
    result: str = "success"


@dataclass
class FakeResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_copy(self, deep: bool = False) -> FakeResponse:
        return FakeResponse(
            content=self.content,
            tool_calls=json.loads(json.dumps(self.tool_calls)) if deep else list(self.tool_calls),
            metadata=json.loads(json.dumps(self.metadata)) if deep else dict(self.metadata),
        )


class FakeAdapter:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.model = "fake-model"
        self.base_url = "http://fake.local"

    async def chat(
        self, _messages: list[Any], tools: list[dict[str, Any]] | None = None, **_kwargs: Any
    ) -> FakeResponse:
        self.calls += 1
        assert tools is None or isinstance(tools, list)
        if self._responses:
            return self._responses.pop(0)
        return FakeResponse(content="done", tool_calls=[])


class FakeExecutor:
    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.results = results or {}
        self.max_retries = 1
        self.timeout = 10.0
        self.executed: list[str] = []

    async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
        self.executed.append(skill.name)
        if skill.name in self.results:
            return self.results[skill.name]
        return {"result": "ok", "skill": skill.name, "args": args}


class TestToolCallRunnerIntegration:
    @pytest.mark.asyncio
    async def test_full_flow_policy_consent(self) -> None:
        hooks_manager = SkillHooksManager()
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW,
            user_invocable=True,
            side_effect=SideEffect.NONE,
        )
        policy_enforcer.register_skill_policy("test-skill", policy)

        consent_manager = ConsentManager(
            store=InMemoryConsentStore(),
            handler=PolicyBasedConsentHandler(default_grant=True),
        )
        metrics_store = MetricsStore()
        hook_called = {"pre": False, "post": False}

        async def pre_hook(_ctx: Any) -> dict[str, Any]:
            hook_called["pre"] = True
            return {"success": True, "output": "pre-hook executed"}

        async def post_hook(_ctx: Any) -> dict[str, Any]:
            hook_called["post"] = True
            return {"success": True, "output": "post-hook executed"}

        skill = SkillSpec(
            name="test-skill",
            description="Integration test skill",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            executor=lambda _: SimpleOutput(result="executed"),
            hooks=[
                SkillHook(
                    event=HookEvent.PRE_TOOL_USE,
                    hook_type=HookType.HANDLER,
                    handler=pre_hook,
                ),
                SkillHook(
                    event=HookEvent.POST_TOOL_USE,
                    hook_type=HookType.HANDLER,
                    handler=post_hook,
                ),
            ],
        )
        hooks_manager.register_hooks(skill)

        runner = ToolCallRunner(
            skill_hooks_manager=hooks_manager,
            policy_enforcer=policy_enforcer,
            consent_manager=consent_manager,
            metrics_store=metrics_store,
        )
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "test-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        _response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(tool_trace) == 1
        assert tool_trace[0]["tool_name"] == "test-skill"
        assert hook_called == {"pre": True, "post": True}

        metrics = metrics_store.aggregate("test-skill")
        assert metrics is not None
        assert metrics.reliability.total_count >= 1
        assert metrics.latency.samples >= 1

    @pytest.mark.asyncio
    async def test_policy_blocked_visible_trace(self) -> None:
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.DENY,
            user_invocable=False,
            side_effect=SideEffect.MIXED,
        )
        policy_enforcer.register_skill_policy("blocked-skill", policy)

        skill = SkillSpec(
            name="blocked-skill",
            description="Blocked skill",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            executor=lambda _: SimpleOutput(result="should not execute"),
        )
        executor = FakeExecutor()
        runner = ToolCallRunner(policy_enforcer=policy_enforcer)
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "blocked-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        _response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=1,
        )

        assert executor.executed == []
        assert len(tool_trace) == 1
        assert tool_trace[0]["policy_blocked"] is True
        assert "denies model auto-invocation" in tool_trace[0].get("block_reason", "")

    @pytest.mark.asyncio
    async def test_hooks_fire_in_order(self) -> None:
        events: list[str] = []

        async def on_session_start(_ctx: Any) -> dict[str, Any]:
            events.append("SessionStart")
            return {"success": True, "output": "ok"}

        async def on_pre_tool(_ctx: Any) -> dict[str, Any]:
            events.append("PreToolUse")
            return {"success": True, "output": "ok"}

        async def on_post_tool(_ctx: Any) -> dict[str, Any]:
            events.append("PostToolUse")
            return {"success": True, "output": "ok"}

        async def on_stop(_ctx: Any) -> dict[str, Any]:
            events.append("Stop")
            return {"success": True, "output": "ok"}

        skill = SkillSpec(
            name="hooked-skill",
            description="Skill with all hooks",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            hooks=[
                SkillHook(
                    event=HookEvent.SESSION_START,
                    hook_type=HookType.HANDLER,
                    handler=on_session_start,
                ),
                SkillHook(
                    event=HookEvent.PRE_TOOL_USE,
                    hook_type=HookType.HANDLER,
                    handler=on_pre_tool,
                ),
                SkillHook(
                    event=HookEvent.POST_TOOL_USE,
                    hook_type=HookType.HANDLER,
                    handler=on_post_tool,
                ),
                SkillHook(event=HookEvent.STOP, hook_type=HookType.HANDLER, handler=on_stop),
            ],
        )
        hooks_mgr = SkillHooksManager()
        hooks_mgr.register_hooks(skill)

        runner = ToolCallRunner(skill_hooks_manager=hooks_mgr)
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "hooked-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert events == ["SessionStart", "PreToolUse", "PostToolUse", "Stop"]


class TestPolicyConsentMetricsIntegration:
    @pytest.mark.asyncio
    async def test_allow_flow_records_metrics(self) -> None:
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW,
            side_effect=SideEffect.NONE,
        )
        policy_enforcer.register_skill_policy("allowed-skill", policy)
        metrics_store = MetricsStore()

        skill = SkillSpec(
            name="allowed-skill",
            description="Allowed skill",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            executor=lambda _: SimpleOutput(),
            invocation_policy=policy,
        )

        runner = ToolCallRunner(
            policy_enforcer=policy_enforcer,
            metrics_store=metrics_store,
        )
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "allowed-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0]["tool_name"] == "allowed-skill"
        metrics = metrics_store.aggregate("allowed-skill")
        assert metrics is not None
        assert metrics.reliability.total_count >= 1

    @pytest.mark.asyncio
    async def test_deny_flow_marks_blocked(self) -> None:
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.DENY,
            user_invocable=False,
            side_effect=SideEffect.EXEC,
        )
        policy_enforcer.register_skill_policy("denied-skill", policy)

        skill = SkillSpec(
            name="denied-skill",
            description="Denied skill",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            executor=lambda _: SimpleOutput(),
            invocation_policy=policy,
        )

        runner = ToolCallRunner(policy_enforcer=policy_enforcer)
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "denied-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0]["policy_blocked"] is True

    def test_write_access_needs_consent(self) -> None:
        perms = Permissions.from_dict(
            {
                "filesystem": {"read": True, "write": True, "paths": ["./data/**"]},
            }
        )

        assert perms.requires_consent() is True
        descriptions = perms.describe()
        assert any("Write" in description for description in descriptions)

    def test_read_only_no_consent(self) -> None:
        perms = Permissions.from_dict(
            {
                "filesystem": {"read": True, "write": False},
            }
        )

        assert perms.requires_consent() is False


class TestHooksLifecycleIntegration:
    @pytest.mark.asyncio
    async def test_hook_exception_not_abort(self) -> None:
        async def failing_hook(_ctx: Any) -> dict[str, Any]:
            raise RuntimeError("hook explosion")

        skill = SkillSpec(
            name="fail-hook",
            description="Failing hooks",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            hooks=[
                SkillHook(
                    event=HookEvent.SESSION_START,
                    hook_type=HookType.HANDLER,
                    handler=failing_hook,
                ),
                SkillHook(event=HookEvent.STOP, hook_type=HookType.HANDLER, handler=failing_hook),
            ],
        )

        hooks_mgr = SkillHooksManager()
        hooks_mgr.register_hooks(skill)

        runner = ToolCallRunner(skill_hooks_manager=hooks_mgr)
        response, _trace = await runner.run(
            adapter=FakeAdapter([FakeResponse(content="ok")]),
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert response is not None


class TestToolCallRunnerErrorPaths:
    @pytest.mark.asyncio
    async def test_unregistered_records_trace(self) -> None:
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ghost",
                            "type": "function",
                            "function": {"name": "ghost-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        runner = ToolCallRunner()
        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "ghost-skill", "description": "X", "parameters": {}},
                }
            ],
            skills=[],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0]["tool_name"] == "ghost-skill"

    @pytest.mark.asyncio
    async def test_executor_exception_in_trace(self) -> None:
        class FailingExecutor:
            max_retries = 1
            timeout = 5.0

            async def execute(self, skill: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
                _ = (skill, args)
                raise RuntimeError("executor boom")

        skill = SkillSpec(
            name="boom-skill",
            description="Boom",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_boom",
                            "type": "function",
                            "function": {"name": "boom-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        runner = ToolCallRunner()
        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FailingExecutor(),
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0]["tool_name"] == "boom-skill"
        assert trace[0].get("error") or trace[0].get("status") in ("error", "failed", None)

    @pytest.mark.asyncio
    async def test_policy_blocked_not_executed(self) -> None:
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.DENY,
            user_invocable=False,
        )
        policy_enforcer.register_skill_policy("forbidden", policy)

        skill = SkillSpec(
            name="forbidden",
            description="Forbidden skill",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            invocation_policy=policy,
        )

        executor = FakeExecutor()
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_f",
                            "type": "function",
                            "function": {"name": "forbidden", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        runner = ToolCallRunner(policy_enforcer=policy_enforcer)
        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=1,
        )

        assert executor.executed == []
        assert len(trace) == 1
        assert trace[0]["policy_blocked"] is True


class TestConsentIntegration:
    @pytest.mark.asyncio
    async def test_consent_blocks_without_manager(self) -> None:
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
            user_invocable=True,
            side_effect=SideEffect.FILESYSTEM,
        )
        policy_enforcer.register_skill_policy("consent-skill", policy)

        skill = SkillSpec(
            name="consent-skill",
            description="Needs consent",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            invocation_policy=policy,
        )
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_c",
                            "type": "function",
                            "function": {"name": "consent-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        executor = FakeExecutor()
        runner = ToolCallRunner(policy_enforcer=policy_enforcer)
        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0]["policy_blocked"] is True
        assert executor.executed == []

    @pytest.mark.asyncio
    async def test_consent_executes_when_granted(self) -> None:
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
            user_invocable=True,
            side_effect=SideEffect.NETWORK,
        )
        policy_enforcer.register_skill_policy("consent-ok", policy)

        consent_manager = ConsentManager(
            store=InMemoryConsentStore(),
            handler=PolicyBasedConsentHandler(auto_grant_skills={"consent-ok"}),
        )
        skill = SkillSpec(
            name="consent-ok",
            description="Auto-granted consent",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            invocation_policy=policy,
        )
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ok",
                            "type": "function",
                            "function": {"name": "consent-ok", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        executor = FakeExecutor()
        runner = ToolCallRunner(
            policy_enforcer=policy_enforcer,
            consent_manager=consent_manager,
        )
        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0].get("policy_blocked") is not True
        assert executor.executed == ["consent-ok"]

    @pytest.mark.asyncio
    async def test_consent_blocks_when_denied(self) -> None:
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
            side_effect=SideEffect.EXEC,
        )
        policy_enforcer.register_skill_policy("consent-denied", policy)

        consent_manager = ConsentManager(
            store=InMemoryConsentStore(),
            handler=PolicyBasedConsentHandler(auto_deny_skills={"consent-denied"}),
        )
        skill = SkillSpec(
            name="consent-denied",
            description="Denied consent",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            invocation_policy=policy,
        )
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_d",
                            "type": "function",
                            "function": {"name": "consent-denied", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        executor = FakeExecutor()
        runner = ToolCallRunner(
            policy_enforcer=policy_enforcer,
            consent_manager=consent_manager,
        )
        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0]["policy_blocked"] is True
        assert executor.executed == []

    @pytest.mark.asyncio
    async def test_non_interactive_blocks(self) -> None:
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
            side_effect=SideEffect.MIXED,
        )
        policy_enforcer.register_skill_policy("ni-skill", policy)

        skill = SkillSpec(
            name="ni-skill",
            description="Non-interactive",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            invocation_policy=policy,
        )
        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ni",
                            "type": "function",
                            "function": {"name": "ni-skill", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        executor = FakeExecutor()
        runner = ToolCallRunner(
            policy_enforcer=policy_enforcer,
            consent_manager=ConsentManager(interactive=False),
        )
        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0]["policy_blocked"] is True
        assert executor.executed == []


class TestUnloadThenCallIntegration:
    @pytest.mark.asyncio
    async def test_unloaded_returns_trace(self) -> None:
        registry = SkillRegistry()

        from houyi.skills.weather import get_weather

        registry.register(get_weather, overwrite=True)
        assert "get_weather" in [skill.name for skill in registry.list()]

        registry.unregister("get_weather")
        assert "get_weather" not in [skill.name for skill in registry.list()]

        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_unloaded",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                )
            ]
        )

        runner = ToolCallRunner()
        _response, trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather", "description": "W", "parameters": {}},
                }
            ],
            skills=[],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(trace) == 1
        assert trace[0]["tool_name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_get_none_after_unload(self) -> None:
        registry = SkillRegistry()

        from houyi.skills.weather import get_date

        registry.register(get_date, overwrite=True)
        assert registry.get("get_date") is not None

        registry.unregister("get_date")
        assert registry.get("get_date") is None
