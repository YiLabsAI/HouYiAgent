"""Tests for SimpleSkill invocation policy and consent.

Reference: SimpleSkill Specification v0.1 Section 5.2-5.3
"""

from houyi.core.skill.policy import (
    ExecPerm,
    FilesystemPerm,
    InvocationDecision,
    InvocationPolicy,
    ModelAutoInvoke,
    NetworkPerm,
    Permissions,
    PolicyEnforcer,
    ResourceLimits,
    SideEffect,
)


class TestModelAutoInvoke:
    """Test ModelAutoInvoke enum."""

    def test_enum_values(self):
        assert ModelAutoInvoke.ALLOW.value == "allow"
        assert ModelAutoInvoke.DENY.value == "deny"
        assert ModelAutoInvoke.ALLOW_WITH_CONSENT.value == "allow_with_consent"


class TestSideEffect:
    """Test SideEffect enum."""

    def test_enum_values(self):
        assert SideEffect.NONE.value == "none"
        assert SideEffect.FILESYSTEM.value == "filesystem"
        assert SideEffect.NETWORK.value == "network"
        assert SideEffect.EXEC.value == "exec"
        assert SideEffect.MIXED.value == "mixed"


class TestPermissions:
    """Test Permissions dataclass."""

    def test_default_permissions(self):
        perms = Permissions()
        assert not perms.filesystem.read
        assert not perms.filesystem.write
        assert not perms.network.enabled
        assert not perms.exec.enabled
        assert perms.secrets == []

    def test_requires_consent_for_write(self):
        perms = Permissions(filesystem=FilesystemPerm(write=True))
        assert perms.requires_consent()

    def test_requires_consent_for_network(self):
        perms = Permissions(network=NetworkPerm(enabled=True))
        assert perms.requires_consent()

    def test_requires_consent_for_exec(self):
        perms = Permissions(exec=ExecPerm(enabled=True))
        assert perms.requires_consent()

    def test_requires_consent_for_secrets(self):
        perms = Permissions(secrets=["API_KEY"])
        assert perms.requires_consent()

    def test_no_consent_for_read_only(self):
        perms = Permissions(filesystem=FilesystemPerm(read=True))
        assert not perms.requires_consent()

    def test_describe_permissions(self):
        perms = Permissions(
            filesystem=FilesystemPerm(read=True, write=True, paths=["./data"]),
            network=NetworkPerm(enabled=True, domains=["api.example.com"]),
            secrets=["API_KEY"],
        )
        descriptions = perms.describe()
        assert any("Read files" in d for d in descriptions)
        assert any("Write files" in d for d in descriptions)
        assert any("Network access" in d for d in descriptions)
        assert any("Access secrets" in d for d in descriptions)

    def test_to_dict_and_from_dict(self):
        perms = Permissions(
            filesystem=FilesystemPerm(read=True, write=True, paths=["./data"]),
            network=NetworkPerm(enabled=True, domains=["api.example.com"]),
            exec=ExecPerm(enabled=True, commands=["python"]),
            secrets=["API_KEY"],
            resources=ResourceLimits(timeout_ms=5000),
        )
        data = perms.to_dict()
        restored = Permissions.from_dict(data)
        assert restored.filesystem.read == perms.filesystem.read
        assert restored.filesystem.write == perms.filesystem.write
        assert restored.network.enabled == perms.network.enabled
        assert restored.exec.enabled == perms.exec.enabled
        assert restored.secrets == perms.secrets
        assert restored.resources.timeout_ms == perms.resources.timeout_ms


class TestInvocationPolicy:
    """Test InvocationPolicy dataclass."""

    def test_default_policy(self):
        policy = InvocationPolicy()
        assert policy.model_auto_invoke == ModelAutoInvoke.ALLOW
        assert policy.user_invocable is True
        assert policy.side_effect == SideEffect.NONE

    def test_default_for_side_effect_none(self):
        policy = InvocationPolicy.default_for_side_effect(SideEffect.NONE)
        assert policy.model_auto_invoke == ModelAutoInvoke.ALLOW

    def test_default_for_side_effect_filesystem(self):
        policy = InvocationPolicy.default_for_side_effect(SideEffect.FILESYSTEM)
        assert policy.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT

    def test_default_for_side_effect_network(self):
        policy = InvocationPolicy.default_for_side_effect(SideEffect.NETWORK)
        assert policy.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT

    def test_should_prompt_consent(self):
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT)
        assert policy.should_prompt_consent()

        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW)
        assert not policy.should_prompt_consent()

    def test_allows_model_invoke(self):
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW)
        assert policy.allows_model_invoke()

        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT)
        assert policy.allows_model_invoke()

        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY)
        assert not policy.allows_model_invoke()

    def test_to_dict_and_from_dict(self):
        policy = InvocationPolicy(
            model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
            user_invocable=False,
            side_effect=SideEffect.FILESYSTEM,
        )
        data = policy.to_dict()
        assert data["modelAutoInvoke"] == "allow_with_consent"
        assert data["userInvocable"] is False
        assert data["sideEffect"] == "filesystem"

        restored = InvocationPolicy.from_dict(data)
        assert restored.model_auto_invoke == policy.model_auto_invoke
        assert restored.user_invocable == policy.user_invocable
        assert restored.side_effect == policy.side_effect


class TestPolicyEnforcer:
    """Test PolicyEnforcer class."""

    def test_default_policy(self):
        enforcer = PolicyEnforcer()
        policy = enforcer.get_policy("unknown_skill")
        assert policy.model_auto_invoke == ModelAutoInvoke.ALLOW

    def test_register_skill_policy(self):
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY)
        enforcer.register_skill_policy("dangerous_skill", policy)

        retrieved = enforcer.get_policy("dangerous_skill")
        assert retrieved.model_auto_invoke == ModelAutoInvoke.DENY

    def test_check_user_invocation_allowed(self):
        enforcer = PolicyEnforcer()
        decision = enforcer.check_invocation(
            skill_name="test_skill",
            is_model_initiated=False,
            user_consent_given=False,
        )
        assert decision.allowed

    def test_check_user_invocation_denied(self):
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(user_invocable=False)
        enforcer.register_skill_policy("no_user_skill", policy)

        decision = enforcer.check_invocation(
            skill_name="no_user_skill",
            is_model_initiated=False,
            user_consent_given=False,
        )
        assert not decision.allowed
        assert "not user-invocable" in (decision.reason or "")

    def test_check_model_invocation_allow(self):
        enforcer = PolicyEnforcer()
        decision = enforcer.check_invocation(
            skill_name="test_skill",
            is_model_initiated=True,
            user_consent_given=False,
        )
        assert decision.allowed

    def test_check_model_invocation_deny(self):
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY)
        enforcer.register_skill_policy("deny_skill", policy)

        decision = enforcer.check_invocation(
            skill_name="deny_skill",
            is_model_initiated=True,
            user_consent_given=False,
        )
        assert not decision.allowed
        assert not decision.requires_consent
        assert "denies model auto-invocation" in (decision.reason or "")

    def test_check_model_invocation_requires_consent(self):
        enforcer = PolicyEnforcer()
        policy = InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT)
        enforcer.register_skill_policy("consent_skill", policy)

        # Without consent
        decision = enforcer.check_invocation(
            skill_name="consent_skill",
            is_model_initiated=True,
            user_consent_given=False,
        )
        assert not decision.allowed
        assert decision.requires_consent
        assert "requires user consent" in (decision.reason or "")

        # With consent
        decision = enforcer.check_invocation(
            skill_name="consent_skill",
            is_model_initiated=True,
            user_consent_given=True,
        )
        assert decision.allowed


class TestInvocationDecision:
    """Test InvocationDecision dataclass."""

    def test_bool_conversion(self):
        decision = InvocationDecision(allowed=True)
        assert bool(decision)

        decision = InvocationDecision(allowed=False)
        assert not bool(decision)

    def test_requires_consent_flag(self):
        decision = InvocationDecision(allowed=False, requires_consent=True)
        assert decision.requires_consent
        assert not decision.allowed
