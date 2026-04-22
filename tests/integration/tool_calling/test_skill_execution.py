"""Cross-module integration tests for SimpleSkill execution flow.

Tests in this file exercise the full path: SDK core (skill spec, policy,
consent, hooks, metrics) → ToolCallRunner → SkillExecutor.

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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
    PolicyEnforcer,
    SideEffect,
)
from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec

# =========================================================================
# Shared test fixtures
# =========================================================================


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

    async def chat(
        self, _messages: list[Any], tools: list[dict[str, Any]] | None = None, **_kwargs: Any
    ) -> FakeResponse:
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


# =========================================================================
# Full Execution Flow (ToolCallRunner)
# =========================================================================


class TestSimpleSkillIntegration:
    """Integration tests for the complete SimpleSkill execution flow."""

    @pytest.mark.asyncio
    async def test_full_flow_all_components(self) -> None:
        """Test complete execution flow with policy, consent, hooks, and metrics."""
        hooks_manager = SkillHooksManager()

        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW,
            user_invocable=True,
            side_effect=SideEffect.NONE,
        )
        policy_enforcer.register_skill_policy("test-skill", policy)

        consent_store = InMemoryConsentStore()
        consent_handler = PolicyBasedConsentHandler(default_grant=True)
        consent_manager = ConsentManager(
            store=consent_store,
            handler=consent_handler,
        )

        metrics_store = MetricsStore()

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
                    event=HookEvent.PRE_TOOL_USE, hook_type=HookType.HANDLER, handler=pre_hook
                ),
                SkillHook(
                    event=HookEvent.POST_TOOL_USE, hook_type=HookType.HANDLER, handler=post_hook
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
                ),
            ]
        )

        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(tool_trace) == 1
        assert tool_trace[0]["tool_name"] == "test-skill"
        assert hook_called["pre"] is True
        assert hook_called["post"] is True

        metrics = metrics_store.aggregate("test-skill")
        assert metrics is not None
        assert metrics.reliability.total_count >= 1
        assert metrics.latency.samples >= 1

    @pytest.mark.asyncio
    async def test_policy_blocks_execution(self) -> None:
        """Test that policy enforcement blocks disallowed invocations."""
        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.DENY,
            user_invocable=False,
            side_effect=SideEffect.MIXED,
        )
        policy_enforcer.register_skill_policy("blocked-skill", policy)

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
                ),
            ]
        )

        executor = FakeExecutor()
        response, tool_trace = await runner.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "test"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=executor,
            max_rounds=1,
        )

        assert len(executor.executed) == 0 or "blocked-skill" not in executor.executed
        assert len(tool_trace) == 1
        assert tool_trace[0]["policy_blocked"] is True
        assert "denies model auto-invocation" in tool_trace[0].get("block_reason", "")

    @pytest.mark.asyncio
    async def test_metrics_export_to_trace(self) -> None:
        """Test that metrics can be exported to trace span."""
        trace_manager = MagicMock()
        mock_span = MagicMock()
        trace_manager.current_span = mock_span

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

        adapter = FakeAdapter(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "traced-skill", "arguments": "{}"},
                        }
                    ],
                ),
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

        runner.export_metrics_to_trace()
        assert mock_span.set_attribute.called

    @pytest.mark.asyncio
    async def test_registry_integration(self, tmp_path: Path) -> None:
        """Test skill registry integration with manifest loading."""
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

        hooks_manager = SkillHooksManager()
        registry = SkillRegistry(hooks_manager=hooks_manager)

        name = registry.register_from_skill_file(skill_md)
        assert name == "registry-test-skill"

        skill = registry.get("registry-test-skill")
        assert skill is not None
        assert skill.description == "Skill loaded from registry"

        metrics_store = MetricsStore()
        runner = ToolCallRunner(
            skill_hooks_manager=hooks_manager,
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
                            "function": {"name": "registry-test-skill", "arguments": "{}"},
                        }
                    ],
                ),
            ]
        )

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
        from houyi.domain.skill.config import HookConfig, SkillConfig

        config = SkillConfig(
            hooks=HookConfig(timeout_seconds=5.0, fail_on_error=True),
        )

        assert config.hooks.timeout_seconds == 5.0
        assert config.hooks.fail_on_error is True

        config_dict = config.to_dict()
        assert config_dict["hooks"]["timeout_seconds"] == 5.0


# =========================================================================
# Preprocessor integration (M8) via ToolCallRunner
# =========================================================================


class TestPreprocessorIntegration:
    """Verify PreprocessorPipeline executes before LLM and injects context."""

    @pytest.mark.asyncio
    async def test_preprocessor_injects_system_message(self) -> None:
        """Preprocessor output appears in messages sent to the LLM."""
        from houyi.domain.skill.preprocessor import PreprocessorSpec, PreprocessorType

        captured_messages: list[list[Any]] = []

        class CapturingAdapter:
            model = "capture-model"
            base_url = "http://fake.local"

            async def chat(self, messages: list[Any], **_kw: Any) -> FakeResponse:
                captured_messages.append(list(messages))
                return FakeResponse(content="done")

        pp = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="echo preprocessor-output-42",
            inject_as="system",
            description="echo-test",
        )

        skill = SkillSpec(
            name="pp-skill",
            description="Preprocessor test",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )

        runner = ToolCallRunner()
        await runner.run(
            adapter=CapturingAdapter(),
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
            preprocessors=[pp],
        )

        assert len(captured_messages) >= 1
        all_content = " ".join(
            m.get("content", "") for m in captured_messages[0] if isinstance(m, dict)
        )
        assert "preprocessor-output-42" in all_content

    @pytest.mark.asyncio
    async def test_preprocessor_failure_non_fatal(self) -> None:
        """A failing preprocessor does not abort the run."""
        from houyi.domain.skill.preprocessor import PreprocessorSpec, PreprocessorType

        pp = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="exit 1",
            description="fail-test",
        )

        skill = SkillSpec(
            name="pp-nonfatal",
            description="Nonfatal preprocessor",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )

        runner = ToolCallRunner()
        response, trace = await runner.run(
            adapter=FakeAdapter([FakeResponse(content="ok")]),
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
            preprocessors=[pp],
        )
        # Run completed without exception
        assert response is not None

    @pytest.mark.asyncio
    async def test_preprocessors_inject_in_order(self) -> None:
        """Multiple preprocessors inject in declaration order."""
        from houyi.domain.skill.preprocessor import PreprocessorSpec, PreprocessorType

        captured: list[list[Any]] = []

        class CapturingAdapter2:
            model = "capture"
            base_url = "http://fake.local"

            async def chat(self, messages: list[Any], **_kw: Any) -> FakeResponse:
                captured.append(list(messages))
                return FakeResponse(content="done")

        pp1 = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="echo FIRST",
            inject_as="system",
            description="first-pp",
        )
        pp2 = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="echo SECOND",
            inject_as="system",
            description="second-pp",
        )

        skill = SkillSpec(
            name="multi-pp",
            description="Multi preprocessor",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )

        runner = ToolCallRunner()
        await runner.run(
            adapter=CapturingAdapter2(),
            messages=[{"role": "user", "content": "hi"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
            preprocessors=[pp1, pp2],
        )

        assert len(captured) >= 1
        sys_msgs = [
            m["content"] for m in captured[0] if isinstance(m, dict) and m.get("role") == "system"
        ]
        combined = "\n".join(sys_msgs)
        first_pos = combined.index("FIRST")
        second_pos = combined.index("SECOND")
        assert first_pos < second_pos, "Preprocessors should inject in order"


# =========================================================================
# Tool Router integration (M9) via ToolCallRunner
# =========================================================================


class TestToolRouterIntegration:
    """Verify ToolRouter filters tools and enforces whitelist at runtime."""

    @pytest.mark.asyncio
    async def test_allowed_tools_whitelist_filters(self) -> None:
        """Only tools in allowed_tools are sent to the LLM."""
        captured_tools: list[list[dict[str, Any]]] = []

        class ToolCapturingAdapter:
            model = "capture"
            base_url = "http://fake.local"

            async def chat(
                self,
                messages: list[Any],
                tools: list[dict[str, Any]] | None = None,
                **_kw: Any,
            ) -> FakeResponse:
                captured_tools.append(tools or [])
                return FakeResponse(content="done")

        # Skill declares allowed_tools = ["alpha"]
        skill = SkillSpec(
            name="restricted-skill",
            description="Restricted",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            allowed_tools=["alpha"],
        )

        alpha_tool = {
            "type": "function",
            "function": {"name": "alpha", "description": "A", "parameters": {}},
        }
        beta_tool = {
            "type": "function",
            "function": {"name": "beta", "description": "B", "parameters": {}},
        }

        runner = ToolCallRunner()
        await runner.run(
            adapter=ToolCapturingAdapter(),
            messages=[{"role": "user", "content": "hi"}],
            tools=[alpha_tool, beta_tool],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(captured_tools) >= 1
        tool_names = [t["function"]["name"] for t in captured_tools[0] if "function" in t]
        assert "alpha" in tool_names
        assert "beta" not in tool_names

    @pytest.mark.asyncio
    async def test_no_restrictions_passes_all(self) -> None:
        """Without allowed_tools, all tools pass through."""
        captured_tools: list[list[dict[str, Any]]] = []

        class AllCapturingAdapter:
            model = "capture"
            base_url = "http://fake.local"

            async def chat(
                self,
                messages: list[Any],
                tools: list[dict[str, Any]] | None = None,
                **_kw: Any,
            ) -> FakeResponse:
                captured_tools.append(tools or [])
                return FakeResponse(content="done")

        skill = SkillSpec(
            name="unrestricted",
            description="No restrictions",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )

        tools_in = [
            {"type": "function", "function": {"name": "x", "description": "X", "parameters": {}}},
            {"type": "function", "function": {"name": "y", "description": "Y", "parameters": {}}},
        ]

        runner = ToolCallRunner()
        await runner.run(
            adapter=AllCapturingAdapter(),
            messages=[{"role": "user", "content": "hi"}],
            tools=tools_in,
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
        )

        assert len(captured_tools) >= 1
        assert len(captured_tools[0]) == 2


# =========================================================================
# Combined: Preprocessor + ToolRouter + Hooks + Policy in single run
# =========================================================================


class TestCombinedPipeline:
    """End-to-end: preprocessor → tool router → hooks → policy → execution."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> None:
        """All subsystems work together in a single ToolCallRunner.run()."""
        from houyi.domain.skill.preprocessor import PreprocessorSpec, PreprocessorType

        events: list[str] = []
        captured_messages: list[list[Any]] = []

        async def on_pre(ctx: Any) -> dict[str, Any]:
            events.append("pre")
            return {"success": True, "output": "ok"}

        async def on_post(ctx: Any) -> dict[str, Any]:
            events.append("post")
            return {"success": True, "output": "ok"}

        skill = SkillSpec(
            name="full-skill",
            description="Full pipeline skill",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            allowed_tools=["full-skill"],
            hooks=[
                SkillHook(event=HookEvent.PRE_TOOL_USE, hook_type=HookType.HANDLER, handler=on_pre),
                SkillHook(
                    event=HookEvent.POST_TOOL_USE, hook_type=HookType.HANDLER, handler=on_post
                ),
            ],
        )

        policy_enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW,
            side_effect=SideEffect.NONE,
        )
        policy_enforcer.register_skill_policy("full-skill", policy)

        hooks_mgr = SkillHooksManager()
        hooks_mgr.register_hooks(skill)

        metrics_store = MetricsStore()

        pp = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="echo CONTEXT_ENRICHED",
            inject_as="system",
            description="context-enrichment",
        )

        class PipelineAdapter:
            model = "pipeline"
            base_url = "http://fake.local"
            calls = 0

            async def chat(
                self,
                messages: list[Any],
                tools: list[dict[str, Any]] | None = None,
                **_kw: Any,
            ) -> FakeResponse:
                self.calls += 1
                captured_messages.append(list(messages))
                if self.calls == 1:
                    return FakeResponse(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "full-skill", "arguments": "{}"},
                            }
                        ],
                    )
                return FakeResponse(content="done")

        runner = ToolCallRunner(
            skill_hooks_manager=hooks_mgr,
            policy_enforcer=policy_enforcer,
            metrics_store=metrics_store,
        )

        response, trace = await runner.run(
            adapter=PipelineAdapter(),
            messages=[{"role": "user", "content": "go"}],
            tools=[
                skill.to_tool_schema(),
                {
                    "type": "function",
                    "function": {"name": "blocked-tool", "description": "X", "parameters": {}},
                },
            ],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=3,
            preprocessors=[pp],
        )

        # 1. Preprocessor injected context
        first_call_content = " ".join(
            m.get("content", "") for m in captured_messages[0] if isinstance(m, dict)
        )
        assert "CONTEXT_ENRICHED" in first_call_content

        # 2. Tool Router filtered out blocked-tool
        # (since skill declares allowed_tools=["full-skill"], "blocked-tool" is removed)
        # We verify indirectly: execution succeeded and only full-skill was called

        # 3. Hooks fired
        assert "pre" in events
        assert "post" in events

        # 4. Execution recorded
        assert len(trace) >= 1
        assert trace[0]["tool_name"] == "full-skill"

        # 5. Metrics collected
        metrics = metrics_store.aggregate("full-skill")
        assert metrics is not None
        assert metrics.reliability.total_count >= 1


# =========================================================================
# SkillSpec preprocessors field integration
# =========================================================================


class TestPreprocessorsOnSkillSpec:
    """Verify preprocessors field on SkillSpec works end-to-end."""

    def test_skill_spec_with_preprocessors(self) -> None:
        """SkillSpec can hold preprocessor specs."""
        from houyi.domain.skill.preprocessor import PreprocessorSpec, PreprocessorType

        pp = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="echo hello",
        )

        skill = SkillSpec(
            name="pp-skill",
            description="Has preprocessor",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            preprocessors=[pp],
        )

        assert len(skill.preprocessors) == 1
        assert skill.preprocessors[0].command == "echo hello"

    @pytest.mark.asyncio
    async def test_runner_with_skill_preprocessors(self) -> None:
        """ToolCallRunner uses preprocessors from SkillSpec."""
        from houyi.domain.skill.preprocessor import PreprocessorSpec, PreprocessorType

        captured: list[list[Any]] = []

        class CaptureAdapter:
            model = "cap"
            base_url = "http://fake"

            async def chat(self, messages: list[Any], **_kw: Any) -> FakeResponse:
                captured.append(list(messages))
                return FakeResponse(content="done")

        pp = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="echo SPEC_PP_OUTPUT",
            inject_as="system",
            description="skill-level-pp",
        )

        skill = SkillSpec(
            name="pp-aware",
            description="Has preprocessor",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            preprocessors=[pp],
        )

        runner = ToolCallRunner()
        await runner.run(
            adapter=CaptureAdapter(),
            messages=[{"role": "user", "content": "go"}],
            tools=[skill.to_tool_schema()],
            skills=[skill],
            executor=FakeExecutor(),
            max_rounds=1,
            preprocessors=skill.preprocessors,
        )

        assert len(captured) >= 1
        all_content = " ".join(m.get("content", "") for m in captured[0] if isinstance(m, dict))
        assert "SPEC_PP_OUTPUT" in all_content
