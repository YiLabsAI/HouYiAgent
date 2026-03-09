"""Skill subsystem tests: policy governance, hooks lifecycle, registry, and service.

These tests stay within the ``houyi.core.skill`` + ``houyi.core.skill_registry``
boundary (no ``ToolCallRunner``).

Use ``pytest -m smoke`` to run only the critical-path smoke tests.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from pydantic import BaseModel

from houyi.domain.skill.hooks import HookEvent, HookType, SkillHook, SkillHooksManager
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
    """Empty input for testing."""

    pass


# =========================================================================
# Smoke Tests: Minimum Viable Path (SKILL.md → parse → register → use)
# =========================================================================


class TestSmokeFullCycle:
    """Smoke test: SKILL.md → parse → register → list → tool_schema → execute.

    These must ALL pass before any delivery.
    Run with ``pytest -m smoke`` for fast feedback.
    """

    @pytest.mark.smoke
    def test_skill_md_to_registry_to_tool_schema(self, tmp_path: Path) -> None:
        """Full cycle: write SKILL.md → load → register → get schema → verify."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent("""\
        ---
        name: smoke-test-skill
        description: A skill for smoke testing the full cycle
        version: "1.0.0"
        user-invocable: true
        allowed-tools:
          - Read
          - Write
        invocationPolicy:
          modelAutoInvoke: allow
          sideEffect: filesystem
        permissions:
          filesystem:
            read: true
            write: true
            paths:
              - "${WORKSPACE}/**"
        hooks:
          PreToolUse:
            - matcher: "Write"
              type: handler
              handler: my_skill.hooks:validate_write
        ---

        # Smoke Test Skill

        ## Description
        This skill demonstrates the full SimpleSkill loading cycle.

        ## Input Schema
        ```json
        {
          "type": "object",
          "properties": {
            "action": {"type": "string", "description": "Action to perform"}
          },
          "required": ["action"]
        }
        ```
        """)
        )

        spec = SkillSpec.from_file(str(skill_dir / "SKILL.md"))
        assert spec.name == "smoke-test-skill"
        assert spec.version == "1.0.0"
        assert isinstance(spec.invocation_policy, InvocationPolicy)
        assert isinstance(spec.permissions, Permissions)
        assert len(spec.hooks) == 1

        registry = SkillRegistry()
        registry.register(spec)
        assert registry.get("smoke-test-skill") is not None

        skill_list = registry.list()
        assert len(skill_list) == 1
        assert skill_list[0].name == "smoke-test-skill"

        schemas = registry.as_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "smoke-test-skill"
        assert "action" in schemas[0]["function"]["parameters"]["properties"]

        assert registry.unregister("smoke-test-skill") is True
        assert registry.get("smoke-test-skill") is None

    @pytest.mark.smoke
    def test_directory_discovery_and_registration(self, tmp_path: Path) -> None:
        """Smoke: discover SKILL.md files in directory tree and register all."""
        for skill_name in ["alpha", "beta", "gamma"]:
            skill_dir = tmp_path / skill_name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                textwrap.dedent(f"""\
            ---
            name: {skill_name}
            description: Skill {skill_name} for directory discovery test
            ---
            # {skill_name}
            """)
            )

        registry = SkillRegistry()
        registered = registry.register_from_directory(tmp_path)

        assert len(registered) == 3
        assert set(registered) == {"alpha", "beta", "gamma"}
        assert len(registry.list()) == 3

    @pytest.mark.smoke
    def test_manifest_loading(self, tmp_path: Path) -> None:
        """Smoke: load and inspect simpleskill.json manifest."""
        manifest = {
            "id": "test-package",
            "version": "1.0.0",
            "name": "Test Package",
            "description": "A test skill package",
            "contributions": {
                "skills": [
                    {
                        "id": "manifest-skill",
                        "description": "Skill from manifest",
                        "invocationPolicy": {
                            "modelAutoInvoke": "allow",
                            "sideEffect": "none",
                        },
                    }
                ],
            },
        }
        manifest_path = tmp_path / "simpleskill.json"
        manifest_path.write_text(json.dumps(manifest))

        registry = SkillRegistry()
        loaded_manifest = registry.get_manifest(manifest_path)

        assert loaded_manifest.name == "Test Package"
        assert loaded_manifest.version == "1.0.0"
        assert loaded_manifest.contributions is not None
        assert len(loaded_manifest.contributions.skills) == 1
        assert loaded_manifest.contributions.skills[0].id == "manifest-skill"


# =========================================================================
# Invocation Policy Edge Cases
# =========================================================================


class TestInvocationPolicyEdgeCases:
    """Edge case tests for invocation policy governance."""

    def test_side_effect_defaults_to_consent_required(self) -> None:
        """Spec: For sideEffect != none, modelAutoInvoke SHOULD default to consent."""
        for se in [SideEffect.FILESYSTEM, SideEffect.NETWORK, SideEffect.EXEC, SideEffect.MIXED]:
            policy = InvocationPolicy.default_for_side_effect(se)
            assert policy.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT

        policy_none = InvocationPolicy.default_for_side_effect(SideEffect.NONE)
        assert policy_none.model_auto_invoke == ModelAutoInvoke.ALLOW

    def test_policy_enforcer_default_fallback(self) -> None:
        """Test that enforcer falls back to default policy for unknown skills."""
        default = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY, user_invocable=False)
        enforcer = PolicyEnforcer(default_policy=default)

        policy = enforcer.get_policy("unknown-skill")
        assert policy.model_auto_invoke == ModelAutoInvoke.DENY
        assert policy.user_invocable is False

    def test_registered_policy_overrides_default(self) -> None:
        default = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY)
        enforcer = PolicyEnforcer(default_policy=default)

        specific = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW)
        enforcer.register_skill_policy("my-skill", specific)

        assert enforcer.get_policy("other-skill").model_auto_invoke == ModelAutoInvoke.DENY
        assert enforcer.get_policy("my-skill").model_auto_invoke == ModelAutoInvoke.ALLOW

    def test_user_invocation_bypasses_model_policy(self) -> None:
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY, user_invocable=True)
        enforcer.register_skill_policy("user-only", policy)

        decision_model = enforcer.check_invocation("user-only", is_model_initiated=True)
        assert decision_model.allowed is False
        assert "denies model auto-invocation" in (decision_model.reason or "")

        decision_user = enforcer.check_invocation("user-only", is_model_initiated=False)
        assert decision_user.allowed is True

    def test_consent_requirement_with_prior_consent(self) -> None:
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
            side_effect=SideEffect.FILESYSTEM,
        )
        enforcer.register_skill_policy("write-skill", policy)

        decision_no = enforcer.check_invocation(
            "write-skill", is_model_initiated=True, user_consent_given=False
        )
        assert decision_no.allowed is False
        assert decision_no.requires_consent is True

        decision_yes = enforcer.check_invocation(
            "write-skill", is_model_initiated=True, user_consent_given=True
        )
        assert decision_yes.allowed is True

    def test_user_not_invocable_skill(self) -> None:
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW, user_invocable=False)
        enforcer.register_skill_policy("internal-tool", policy)

        assert enforcer.check_invocation("internal-tool", is_model_initiated=True).allowed is True

        decision_user = enforcer.check_invocation("internal-tool", is_model_initiated=False)
        assert decision_user.allowed is False
        assert "not user-invocable" in (decision_user.reason or "")

    def test_permissions_require_consent_for_sensitive_ops(self) -> None:
        from houyi.domain.skill.policy import ExecPerm, FilesystemPerm, NetworkPerm

        assert Permissions(filesystem=FilesystemPerm(read=True)).requires_consent() is False
        assert Permissions(filesystem=FilesystemPerm(write=True)).requires_consent() is True
        assert Permissions(filesystem=FilesystemPerm(delete=True)).requires_consent() is True
        assert Permissions(network=NetworkPerm(enabled=True)).requires_consent() is True
        assert Permissions(exec=ExecPerm(enabled=True)).requires_consent() is True
        assert Permissions(secrets=["API_KEY"]).requires_consent() is True

    def test_permissions_description_generation(self) -> None:
        from houyi.domain.skill.policy import ExecPerm, FilesystemPerm, NetworkPerm

        perms = Permissions(
            filesystem=FilesystemPerm(read=True, write=True, paths=["./data/*"]),
            network=NetworkPerm(enabled=True, domains=["api.example.com"]),
            exec=ExecPerm(enabled=True, commands=["git", "npm"]),
            secrets=["OPENAI_API_KEY"],
        )

        descriptions = perms.describe()
        assert len(descriptions) == 5
        assert any("Read files" in d for d in descriptions)
        assert any("Write files" in d for d in descriptions)
        assert any("api.example.com" in d for d in descriptions)
        assert any("git" in d and "npm" in d for d in descriptions)
        assert any("OPENAI_API_KEY" in d for d in descriptions)

    def test_policy_serialization_roundtrip(self) -> None:
        original = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
            user_invocable=False,
            side_effect=SideEffect.MIXED,
        )

        data = original.to_dict()
        assert data["modelAutoInvoke"] == "allow_with_consent"
        assert data["userInvocable"] is False
        assert data["sideEffect"] == "mixed"

        restored = InvocationPolicy.from_dict(data)
        assert restored.model_auto_invoke == original.model_auto_invoke
        assert restored.user_invocable == original.user_invocable
        assert restored.side_effect == original.side_effect

    def test_policy_from_dict_with_snake_case(self) -> None:
        camel = {"modelAutoInvoke": "deny", "userInvocable": False, "sideEffect": "exec"}
        assert InvocationPolicy.from_dict(camel).model_auto_invoke == ModelAutoInvoke.DENY

        snake = {"model_auto_invoke": "deny", "user_invocable": False, "side_effect": "exec"}
        assert InvocationPolicy.from_dict(snake).model_auto_invoke == ModelAutoInvoke.DENY

    @pytest.mark.asyncio
    async def test_consent_with_allow_policy_skips_prompt(self) -> None:
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW,
            side_effect=SideEffect.NONE,
        )
        enforcer.register_skill_policy("safe-skill", policy)

        decision = enforcer.check_invocation(
            "safe-skill", is_model_initiated=True, user_consent_given=False
        )
        assert decision.allowed is True
        assert decision.requires_consent is False


# =========================================================================
# Hooks Lifecycle
# =========================================================================


class TestHooksLifecycle:
    """Integration: hooks registration, execution, and cleanup."""

    def test_hooks_register_and_unregister(self) -> None:
        hooks_manager = SkillHooksManager()
        registry = SkillRegistry(hooks_manager=hooks_manager)

        skill = SkillSpec(
            name="hooked-skill",
            description="Skill with hooks",
            input_schema=EmptyInput,
            output_schema=EmptyInput,
            hooks=[
                SkillHook(
                    event=HookEvent.PRE_TOOL_USE, hook_type=HookType.HANDLER, matcher="Write"
                ),
                SkillHook(event=HookEvent.POST_TOOL_USE, hook_type=HookType.HANDLER),
            ],
        )

        registry.register(skill)

        pre_hooks = hooks_manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)
        assert len(pre_hooks) >= 1

        registry.unregister("hooked-skill")
        pre_hooks_after = hooks_manager.get_registered_hooks(HookEvent.PRE_TOOL_USE)
        assert len(pre_hooks_after) == 0

    def test_hooks_from_skill_md(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text(
            textwrap.dedent("""\
        ---
        name: md-hooked
        description: Hooks from SKILL.md
        hooks:
          PreToolUse:
            - matcher: "Write|Edit"
              type: handler
              handler: module:pre_hook
          PostToolUse:
            - matcher: ".*"
              type: handler
              handler: module:post_hook
          Stop:
            - type: handler
              handler: module:stop_hook
        ---
        # MD Hooked
        """)
        )

        spec = SkillSpec.from_file(str(tmp_path / "SKILL.md"))

        assert len(spec.hooks) == 3
        events = {h.event for h in spec.hooks}
        assert HookEvent.PRE_TOOL_USE in events
        assert HookEvent.POST_TOOL_USE in events
        assert HookEvent.STOP in events


# =========================================================================
# SkillService (Console Server Layer)
# =========================================================================


class TestSkillServiceIntegration:
    """Integration: SkillService for Console UI backend."""

    def test_skill_service_list_and_detail(self) -> None:
        registry = SkillRegistry()

        skill = SkillSpec(
            name="svc-test",
            description="Service test skill",
            input_schema=EmptyInput,
            output_schema=EmptyInput,
            version="1.0.0",
            invocation_policy=InvocationPolicy(
                model_auto_invoke=ModelAutoInvoke.ALLOW,
                side_effect=SideEffect.NONE,
            ),
            permissions=Permissions.from_dict({"filesystem": {"read": True}}),
        )
        registry.register(skill)

        skills = registry.list()
        assert len(skills) == 1

        detail = registry.get("svc-test")
        assert detail is not None
        assert detail.name == "svc-test"
        assert detail.version == "1.0.0"
        assert isinstance(detail.invocation_policy, InvocationPolicy)
        assert isinstance(detail.permissions, Permissions)

        schema = detail.to_tool_schema()
        assert schema["function"]["name"] == "svc-test"

    def test_skill_service_dry_run_validation(self) -> None:
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.DENY, side_effect=SideEffect.EXEC
        )
        enforcer.register_skill_policy("dangerous", policy)

        decision = enforcer.check_invocation("dangerous", is_model_initiated=True)
        assert decision.allowed is False
        assert "denies model auto-invocation" in (decision.reason or "")

        decision_user = enforcer.check_invocation("dangerous", is_model_initiated=False)
        assert decision_user.allowed is True
