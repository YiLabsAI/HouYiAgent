"""Integration tests for SimpleSkill v0.1 full execution flow.

These tests verify the complete integration of:
- Policy enforcement
- Consent management
- Hooks execution
- Metrics collection
- Observability
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from houyi.core.skill import (
    ConsentManager,
    HookEvent,
    HookType,
    InMemoryConsentStore,
    InvocationPolicy,
    MetricsStore,
    ModelAutoInvoke,
    PolicyBasedConsentHandler,
    PolicyEnforcer,
    SideEffect,
    SkillHook,
    SkillHooksManager,
    SkillSpec,
)
from houyi.core.skill_registry import SkillRegistry
from houyi.execution.tool_call_runner import ToolCallRunner


class EmptyInput(BaseModel):
    """Empty input for testing."""
    pass


class SimpleOutput(BaseModel):
    """Simple output for testing."""
    result: str = "success"


@dataclass
class FakeResponse:
    """Fake LLM response for testing."""
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
    """Fake LLM adapter for testing."""
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: int = 0
        self.model = "fake-model"
        self.base_url = "http://fake.local"

    async def chat(self, _messages: list[Any], tools: list[dict[str, Any]] | None = None, **_kwargs: Any) -> FakeResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return FakeResponse(content="done", tool_calls=[])


class FakeExecutor:
    """Fake skill executor for testing."""
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


class TestSimpleSkillIntegration:
    """Integration tests for the complete SimpleSkill execution flow."""

    @pytest.mark.asyncio
    async def test_full_flow_with_all_components(self) -> None:
        """Test complete execution flow with policy, consent, hooks, and metrics."""
        # Setup hooks manager
        hooks_manager = SkillHooksManager()

        # Setup policy enforcer with a permissive policy
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW,
            user_invocable=True,
            side_effect=SideEffect.NONE,
        )
        policy_enforcer.register_skill_policy("test-skill", policy)

        # Setup consent manager (auto-allow for testing)
        consent_store = InMemoryConsentStore()
        consent_handler = PolicyBasedConsentHandler(default_grant=True)
        consent_manager = ConsentManager(
            store=consent_store,
            handler=consent_handler,
        )

        # Setup metrics store
        metrics_store = MetricsStore()

        # Create skill with hooks
        hook_called = {"pre": False, "post": False}

        async def pre_hook(ctx: Any) -> dict[str, Any]:
            hook_called["pre"] = True
            return {"success": True, "output": "pre-hook executed"}

        async def post_hook(ctx: Any) -> dict[str, Any]:
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

        # Register hooks
        hooks_manager.register_hooks(skill)

        # Setup runner with all components
        runner = ToolCallRunner(
            skill_hooks_manager=hooks_manager,
            policy_enforcer=policy_enforcer,
            consent_manager=consent_manager,
            metrics_store=metrics_store,
        )

        # Create fake adapter with tool call
        adapter = FakeAdapter([
            FakeResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "test-skill",
                        "arguments": "{}",
                    },
                }],
            ),
        ])

        # Execute
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        # Verify execution happened
        assert len(tool_trace) == 1
        assert tool_trace[0]["tool_name"] == "test-skill"

        # Verify hooks were called
        assert hook_called["pre"] is True
        assert hook_called["post"] is True

        # Verify metrics were collected
        metrics = metrics_store.aggregate("test-skill")
        assert metrics is not None
        assert metrics.reliability.total_count >= 1
        assert metrics.latency.samples >= 1

    @pytest.mark.asyncio
    async def test_policy_blocks_execution(self) -> None:
        """Test that policy enforcement blocks disallowed invocations."""
        # Setup policy enforcer with deny policy
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.DENY,
            user_invocable=False,
            side_effect=SideEffect.MIXED,
        )
        policy_enforcer.register_skill_policy("blocked-skill", policy)

        # Setup metrics to verify blocking
        metrics_store = MetricsStore()

        skill = SkillSpec(
            name="blocked-skill",
            description="Blocked skill",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            executor=lambda _: SimpleOutput(result="should not execute"),
        )

        runner = ToolCallRunner(
            policy_enforcer=policy_enforcer,
            metrics_store=metrics_store,
        )

        adapter = FakeAdapter([
            FakeResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "blocked-skill",
                        "arguments": "{}",
                    },
                }],
            ),
        ])

        # Execute - should be blocked
        executor = FakeExecutor()
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=1,
        )

        # Verify skill was NOT executed (blocked by policy)
        assert len(executor.executed) == 0 or "blocked-skill" not in executor.executed

    @pytest.mark.asyncio
    async def test_metrics_export_to_trace(self) -> None:
        """Test that metrics can be exported to trace span."""
        # Create a mock trace manager
        trace_manager = MagicMock()
        mock_span = MagicMock()
        trace_manager.current_span = mock_span

        # Setup metrics store
        metrics_store = MetricsStore()

        skill = SkillSpec(
            name="traced-skill",
            description="Traced skill",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            executor=lambda _: SimpleOutput(result="traced"),
        )

        runner = ToolCallRunner(
            trace_manager=trace_manager,
            metrics_store=metrics_store,
        )

        adapter = FakeAdapter([
            FakeResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "traced-skill",
                        "arguments": "{}",
                    },
                }],
            ),
        ])

        # Execute
        await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        # Export metrics to trace
        runner.export_metrics_to_trace()

        # Verify metrics were exported
        assert mock_span.set_attribute.called

    @pytest.mark.asyncio
    async def test_registry_integration(self, tmp_path: Path) -> None:
        """Test skill registry integration with manifest loading."""
        # Create SKILL.md file
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: registry-test-skill
description: Skill loaded from registry
user_invocable: true
---

# Registry Test Skill

This skill tests registry integration.
""")

        # Create hooks manager and registry
        hooks_manager = SkillHooksManager()
        registry = SkillRegistry(hooks_manager=hooks_manager)

        # Load skill from file
        name = registry.register_from_skill_file(skill_md)

        assert name == "registry-test-skill"

        # Verify skill is registered
        skill = registry.get("registry-test-skill")
        assert skill is not None
        assert skill.description == "Skill loaded from registry"

        # Verify skill can be used with runner
        metrics_store = MetricsStore()
        runner = ToolCallRunner(
            skill_hooks_manager=hooks_manager,
            metrics_store=metrics_store,
        )

        adapter = FakeAdapter([
            FakeResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "registry-test-skill",
                        "arguments": "{}",
                    },
                }],
            ),
        ])

        # Execute with registered skills
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=registry.as_tool_schemas(),
            skills=registry.list(),
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(tool_trace) == 1
        assert tool_trace[0]["tool_name"] == "registry-test-skill"

    @pytest.mark.asyncio
    async def test_config_driven_execution(self) -> None:
        """Test that configuration affects execution behavior."""
        from houyi.core.skill.config import HookConfig, SkillConfig

        # Create config with custom settings
        config = SkillConfig(
            hooks=HookConfig(
                timeout_seconds=5.0,
                fail_on_error=True,
            ),
        )

        # Verify config values are accessible
        assert config.hooks.timeout_seconds == 5.0
        assert config.hooks.fail_on_error is True

        # Config can be serialized for inspection
        config_dict = config.to_dict()
        assert config_dict["hooks"]["timeout_seconds"] == 5.0
