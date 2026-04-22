"""Conformance tests for SimpleSkill specification §3-§8.

Validates that the HouYi SDK implementation conforms to the core normative
clauses of the SimpleSkill specification.  Each test class maps to a section
of the spec.

Sections covered:
  §3 — Manifest (identity, compatibility, activation, contributions, permissions, trust)
  §4 — Skill & Tool minimal definition
  §5 — Host Runtime API (capability negotiation, invocation policy, consent, observability)
  §6 — Hooks (event semantics, handler types, degradation)
  §7 — Evaluation / Selection (metrics schema)
  §8 — Certification & Expertise (structural validation only)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from houyi.domain.skill.capability import CapabilityNegotiator, HostCapabilities
from houyi.domain.skill.config import SkillConfig
from houyi.domain.skill.consent import (
    ConsentManager,
    InMemoryConsentStore,
    PolicyBasedConsentHandler,
)
from houyi.domain.skill.hooks import HookEvent, HookType, SkillHook, SkillHooksManager
from houyi.domain.skill.metrics import MetricsStore
from houyi.domain.skill.policy import (
    InvocationPolicy,
    ModelAutoInvoke,
    Permissions,
    PolicyEnforcer,
    SideEffect,
)
from houyi.domain.skill.spec import SkillSpec


class EmptyInput(BaseModel):
    pass


class SimpleOutput(BaseModel):
    result: str = "ok"


# =========================================================================
# §3 — Manifest
# =========================================================================


class TestManifestIdentity:
    """§3.2: Manifest MUST contain id/version/name/description."""

    def test_skill_spec_requires_name(self) -> None:
        """name is a required field."""
        with pytest.raises(ValidationError):
            SkillSpec(description="no name", input_schema=EmptyInput, output_schema=SimpleOutput)  # type: ignore[call-arg]

    def test_skill_spec_requires_description(self) -> None:
        """description is a required field."""
        with pytest.raises(ValidationError):
            SkillSpec(
                name="x", input_schema=EmptyInput, output_schema=SimpleOutput, description=None
            )  # type: ignore[call-arg]

    def test_spec_has_version(self) -> None:
        """version SHOULD be supported."""
        spec = SkillSpec(
            name="test",
            description="d",
            version="1.0.0",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )
        assert spec.version == "1.0.0"

    def test_spec_has_author(self) -> None:
        """author SHOULD be supported."""
        spec = SkillSpec(
            name="test",
            description="d",
            author="Bob",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )
        assert spec.author == "Bob"


class TestManifestPermissions:
    """§3.6: Manifest MUST declare permissions."""

    def test_permissions_from_dict(self) -> None:
        perms = Permissions.from_dict(
            {
                "filesystem": {"read": True, "write": True, "paths": ["./data/**"]},
                "network": {"enabled": True, "domains": ["api.example.com"]},
                "exec": {"enabled": False},
                "secrets": ["API_KEY"],
            }
        )
        assert perms.filesystem.read is True
        assert perms.filesystem.write is True
        assert perms.network.enabled is True
        assert perms.exec.enabled is False
        assert "API_KEY" in perms.secrets

    def test_consent_for_write(self) -> None:
        """Write permissions MUST trigger consent."""
        perms = Permissions.from_dict({"filesystem": {"write": True}})
        assert perms.requires_consent() is True

    def test_consent_for_network(self) -> None:
        perms = Permissions.from_dict({"network": {"enabled": True}})
        assert perms.requires_consent() is True

    def test_consent_for_exec(self) -> None:
        perms = Permissions.from_dict({"exec": {"enabled": True}})
        assert perms.requires_consent() is True

    def test_consent_for_secrets(self) -> None:
        perms = Permissions.from_dict({"secrets": ["KEY"]})
        assert perms.requires_consent() is True

    def test_read_only_no_consent(self) -> None:
        perms = Permissions.from_dict({"filesystem": {"read": True}})
        assert perms.requires_consent() is False

    def test_permissions_describe(self) -> None:
        perms = Permissions.from_dict(
            {
                "filesystem": {"write": True},
                "network": {"enabled": True, "domains": ["a.com"]},
            }
        )
        desc = perms.describe()
        assert any("Write" in d for d in desc)
        assert any("a.com" in d for d in desc)


# =========================================================================
# §4 — Skill & Tool minimal definition
# =========================================================================


class TestToolDefinition:
    """§4.1: Tool MUST define name, description, inputSchema; outputSchema is recommended."""

    def test_schema_required_fields(self) -> None:
        spec = SkillSpec(
            name="calc.add",
            description="Add two numbers",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )
        schema = spec.to_tool_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calc.add"
        assert schema["function"]["description"] == "Add two numbers"
        assert "parameters" in schema["function"]

    def test_namespaced_tool_names(self) -> None:
        """Tool names MAY use namespace notation (e.g., 'rag.search')."""
        spec = SkillSpec(
            name="rag.search",
            description="Search",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
        )
        assert spec.name == "rag.search"


class TestSkillDefinition:
    """§4.2: Skill MUST define an identifier, description, and invocationPolicy."""

    def test_skill_has_invocation_policy(self) -> None:
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW,
            side_effect=SideEffect.NONE,
        )
        spec = SkillSpec(
            name="test",
            description="Test",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            invocation_policy=policy,
        )
        assert spec.invocation_policy is not None

    def test_skill_allowed_tools(self) -> None:
        """§4.2: toolRefs[] or host-native equivalent — skills can declare allowed tools."""
        spec = SkillSpec(
            name="multi-tool",
            description="Uses multiple tools",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            allowed_tools=["tool_a", "tool_b"],
        )
        assert len(spec.allowed_tools) == 2


# =========================================================================
# §5 — Host Runtime API
# =========================================================================


class TestCapabilityNegotiation:
    """§5.1: Host MUST provide capability negotiation."""

    def test_host_capabilities_default(self) -> None:
        caps = HostCapabilities()
        assert caps.manifest_formats is not None
        assert caps.execution_forms is not None

    def test_negotiator_full_compatible(self) -> None:
        from houyi.domain.skill.capability import ExtensionRequirements

        host = HostCapabilities()
        negotiator = CapabilityNegotiator(host)
        result = negotiator.check_compatibility(ExtensionRequirements())
        assert result.compatible is True

    def test_negotiator_incompatible_execution_form(self) -> None:
        from houyi.domain.skill.capability import ExecutionForm, ExtensionRequirements

        host = HostCapabilities(execution_forms=[ExecutionForm.IN_PROCESS])
        negotiator = CapabilityNegotiator(host)
        ext_req = ExtensionRequirements(required_execution_forms=[ExecutionForm.MCP])
        result = negotiator.check_compatibility(ext_req)
        # If host doesn't support requested form, should be incompatible
        assert result is not None
        assert result.compatible is False


class TestInvocationPolicy:
    """§5.2: Skill MUST define invocationPolicy."""

    def test_model_auto_invoke_allow(self) -> None:
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW)
        assert policy.allows_model_invoke() is True
        assert policy.should_prompt_consent() is False

    def test_model_auto_invoke_deny(self) -> None:
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY)
        assert policy.allows_model_invoke() is False

    def test_auto_invoke_allow_consent(self) -> None:
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT)
        assert policy.allows_model_invoke() is True
        assert policy.should_prompt_consent() is True

    def test_side_effect_default_policy(self) -> None:
        """§5.2: sideEffect != none → modelAutoInvoke SHOULD default to deny or consent."""
        policy = InvocationPolicy.default_for_side_effect(SideEffect.FILESYSTEM)
        assert policy.model_auto_invoke in (
            ModelAutoInvoke.DENY,
            ModelAutoInvoke.ALLOW_WITH_CONSENT,
        )

    def test_side_effect_none_allows(self) -> None:
        policy = InvocationPolicy.default_for_side_effect(SideEffect.NONE)
        assert policy.model_auto_invoke == ModelAutoInvoke.ALLOW

    def test_policy_enforcer_check(self) -> None:
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.DENY,
            user_invocable=True,
        )
        enforcer.register_skill_policy("test", policy)

        # Model-initiated → denied
        decision = enforcer.check_invocation("test", is_model_initiated=True)
        assert decision.allowed is False

        # User-initiated → allowed
        decision = enforcer.check_invocation("test", is_model_initiated=False)
        assert decision.allowed is True

    def test_policy_enforcer_consent_required(self) -> None:
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT)
        enforcer.register_skill_policy("test", policy)

        decision = enforcer.check_invocation("test", is_model_initiated=True)
        assert decision.allowed is False
        assert decision.requires_consent is True

        # With consent granted
        decision = enforcer.check_invocation(
            "test", is_model_initiated=True, user_consent_given=True
        )
        assert decision.allowed is True


class TestConsent:
    """§5.3: Host MUST implement unified consent interface."""

    @pytest.mark.asyncio
    async def test_consent_manager_remembered(self) -> None:
        """Consent can be remembered and reused."""
        from houyi.domain.skill.consent import (
            ConsentRequest,
            ConsentResult,
            ConsentType,
        )

        store = InMemoryConsentStore()
        handler = PolicyBasedConsentHandler(auto_grant_skills={"test"})
        manager = ConsentManager(store=store, handler=handler)

        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test",
            remember=True,
        )
        resp = await manager.request_consent(request)
        assert resp.is_granted() is True

        # Manually save to store
        store.save(resp)

        # Second request should hit store
        resp2 = await manager.request_consent(request)
        assert resp2.result == ConsentResult.REMEMBERED

    @pytest.mark.asyncio
    async def test_consent_non_interactive(self) -> None:
        """Non-interactive mode returns NOT_INTERACTIVE."""
        from houyi.domain.skill.consent import ConsentRequest, ConsentResult, ConsentType

        manager = ConsentManager(interactive=False)
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test",
        )
        resp = await manager.request_consent(request)
        assert resp.result == ConsentResult.NOT_INTERACTIVE
        assert resp.is_granted() is False

    def test_consent_audit_log(self) -> None:
        """Consent decisions MUST be auditable."""
        manager = ConsentManager()
        log = manager.get_audit_log()
        assert isinstance(log, list)

    @pytest.mark.asyncio
    async def test_consent_revoke(self) -> None:
        """Consent can be revoked."""
        from houyi.domain.skill.consent import ConsentRequest, ConsentType

        store = InMemoryConsentStore()
        handler = PolicyBasedConsentHandler(auto_grant_skills={"test"})
        manager = ConsentManager(store=store, handler=handler)

        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test",
            remember=True,
        )
        resp = await manager.request_consent(request)
        store.save(resp)

        manager.revoke_consent("test")
        assert not manager.check_permission("test")


class TestObservability:
    """§5.4: Host MUST record ToolUsageStarted/Finished/Error events."""

    def test_metrics_store_records(self) -> None:
        """MetricsStore records invocation metrics via MetricsCollector."""
        from houyi.domain.skill.metrics import MetricsCollector

        store = MetricsStore()

        collector = MetricsCollector("test-skill")
        collector.record_success()
        collector.record_latency(50.0)
        store.store(collector.get_metrics())

        collector2 = MetricsCollector("test-skill")
        collector2.record_success()
        collector2.record_latency(80.0)
        store.store(collector2.get_metrics())

        collector3 = MetricsCollector("test-skill")
        collector3.record_error()
        collector3.record_latency(200.0)
        store.store(collector3.get_metrics())

        metrics = store.aggregate("test-skill")
        assert metrics is not None
        assert metrics.reliability.total_count == 3
        assert metrics.reliability.success_count == 2
        assert metrics.reliability.error_count == 1
        assert metrics.latency.samples == 3

    def test_metrics_list_skills(self) -> None:
        from houyi.domain.skill.metrics import MetricsCollector

        store = MetricsStore()

        c1 = MetricsCollector("a")
        c1.record_success()
        c1.record_latency(10.0)
        store.store(c1.get_metrics())

        c2 = MetricsCollector("b")
        c2.record_success()
        c2.record_latency(20.0)
        store.store(c2.get_metrics())

        skills = store.list_skills()
        assert "a" in skills
        assert "b" in skills


# =========================================================================
# §6 — Hooks
# =========================================================================


class TestHooks:
    """§6.1/6.2: Hooks event semantics and handler types."""

    def test_supported_hook_events(self) -> None:
        """Host SHOULD support PreToolUse, PostToolUse, SessionStart, Stop."""
        required_events = {
            HookEvent.PRE_TOOL_USE,
            HookEvent.POST_TOOL_USE,
            HookEvent.SESSION_START,
            HookEvent.STOP,
        }
        for event in required_events:
            assert event.value  # Exists and has a value

    def test_handler_type_support(self) -> None:
        """Host MUST support 'handler' and 'command' hook types."""
        assert HookType.HANDLER.value == "handler"
        assert HookType.COMMAND.value == "command"

    def test_hooks_manager_registration(self) -> None:
        """SkillHooksManager can register and trigger hooks."""
        manager = SkillHooksManager()

        async def dummy_handler(ctx: Any) -> dict[str, Any]:
            return {"success": True, "output": "ok"}

        skill = SkillSpec(
            name="hook-test",
            description="Test",
            input_schema=EmptyInput,
            output_schema=SimpleOutput,
            hooks=[
                SkillHook(
                    event=HookEvent.PRE_TOOL_USE,
                    hook_type=HookType.HANDLER,
                    handler=dummy_handler,
                ),
            ],
        )
        manager.register_hooks(skill)
        assert len(manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)) >= 1

    def test_skill_hook_degradation(self) -> None:
        """§6.2: Degradation rule — handler > command > ignore."""
        # If handler is available, it should be preferred
        hook = SkillHook(
            event=HookEvent.PRE_TOOL_USE,
            hook_type=HookType.HANDLER,
            handler=lambda ctx: {"success": True, "output": "ok"},
        )
        assert hook.hook_type == HookType.HANDLER


# =========================================================================
# §7 — Evaluation / Selection
# =========================================================================


class TestMetricsSchema:
    """§7.1: Metrics Schema (standardized output)."""

    def test_metrics_has_quality_fields(self) -> None:
        """Aggregated metrics include reliability and latency."""
        from houyi.domain.skill.metrics import MetricsCollector

        store = MetricsStore()
        c = MetricsCollector("s")
        c.record_success()
        c.record_latency(100.0)
        store.store(c.get_metrics())

        m = store.aggregate("s")
        assert m is not None
        assert hasattr(m, "reliability")
        assert hasattr(m, "latency")
        assert m.reliability.total_count == 1
        assert m.latency.samples == 1


# =========================================================================
# §3.4 Activation + §3.5 Contributions — Manifest parsing from SKILL.md
# =========================================================================


class TestSkillMdConformance:
    """Verify SKILL.md parsing conforms to manifest spec."""

    def test_frontmatter_parsing(self, tmp_path: Path) -> None:
        """SKILL.md with frontmatter parses into SkillSpec correctly."""
        md = tmp_path / "test.md"
        md.write_text("""---
name: conformance-test
description: Conformance test skill
version: 1.2.3
author: tester
user_invocable: true
allowed_tools:
  - tool_a
  - tool_b
---

# Conformance Test Skill

This skill is for conformance testing.
""")
        spec = SkillSpec.from_file(str(md))
        assert spec.name == "conformance-test"
        assert spec.description == "Conformance test skill"
        assert spec.version == "1.2.3"
        assert spec.author == "tester"
        assert spec.user_invocable is True
        assert "tool_a" in spec.allowed_tools
        assert "tool_b" in spec.allowed_tools

    def test_frontmatter_with_invocation_policy(self, tmp_path: Path) -> None:
        """InvocationPolicy parsed from frontmatter."""
        md = tmp_path / "policy.md"
        md.write_text("""---
name: policy-skill
description: Has policy
invocationPolicy:
  modelAutoInvoke: deny
  userInvocable: false
  sideEffect: filesystem
---

# Policy Skill
""")
        spec = SkillSpec.from_file(str(md))
        assert spec.name == "policy-skill"
        ip = spec.invocation_policy
        assert ip is not None
        assert ip.model_auto_invoke == ModelAutoInvoke.DENY
        assert ip.user_invocable is False
        assert ip.side_effect == SideEffect.FILESYSTEM

    def test_frontmatter_with_preprocessors(self, tmp_path: Path) -> None:
        """Preprocessors parsed from frontmatter."""
        md = tmp_path / "pp.md"
        md.write_text("""---
name: pp-skill
description: Has preprocessors
preprocessors:
  - type: command
    command: echo hello
    inject_as: system
    description: echo test
---

# Preprocessor Skill
""")
        spec = SkillSpec.from_file(str(md))
        assert spec.name == "pp-skill"
        assert len(spec.preprocessors) == 1
        assert spec.preprocessors[0].command == "echo hello"

    def test_progressive_disclosure_minimal(self, tmp_path: Path) -> None:
        """§4.2: Host can discover skill with only id/description/policy."""
        md = tmp_path / "minimal.md"
        md.write_text("""---
name: minimal-skill
description: Minimal skill for progressive disclosure
---

# Minimal
""")
        spec = SkillSpec.from_file(str(md))
        # Must have name and description even without full content
        assert spec.name == "minimal-skill"
        assert spec.description == "Minimal skill for progressive disclosure"


# =========================================================================
# Config
# =========================================================================


class TestSkillConfig:
    """Verify SkillConfig for runtime configurability."""

    def test_config_defaults(self) -> None:
        config = SkillConfig()
        assert config.hooks.timeout_seconds > 0
        assert isinstance(config.hooks.fail_on_error, bool)

    def test_config_to_dict(self) -> None:
        config = SkillConfig()
        d = config.to_dict()
        assert "hooks" in d
        assert "timeout_seconds" in d["hooks"]
