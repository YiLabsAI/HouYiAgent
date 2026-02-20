"""Tests for SkillSerializer (SkillSpec → dict)."""

from __future__ import annotations

from houyi_studio.server.skill.serializer import (
    DEFAULT_VERSION,
    POLICY_ALLOW,
    SIDE_EFFECT_EXEC,
    SIDE_EFFECT_NETWORK,
    SIDE_EFFECT_NONE,
    SkillSerializer,
    dominant_side_effect,
    extract_side_effects,
)

# ── Fakes ─────────────────────────────────────────────────────────────


class _PermKind:
    def __init__(self, enabled=False, write=False, delete=False):
        self.enabled = enabled
        self.write = write
        self.delete = delete


class _Perms:
    def __init__(self, exec_=None, network=None, filesystem=None, descs=None):
        self.exec = exec_ or _PermKind()
        self.network = network or _PermKind()
        self.filesystem = filesystem or _PermKind()
        self._descs = descs or []

    def describe(self):
        return self._descs


class _FakePolicy:
    class _MAI:
        def __init__(self, v):
            self.value = v

    def __init__(self, action="allow"):
        self.model_auto_invoke = self._MAI(action)
        self.user_invocable = True

        class _SE:
            value = "none"

        self.side_effect = _SE()


class _Tool:
    def __init__(self, name, desc=""):
        self.name = name
        self.description = desc


class _Skill:
    def __init__(self, **kw):
        self.name = kw.get("name", "s")
        self.display_name = kw.get("display_name", self.name)
        self.description = kw.get("description", "d")
        self.version = kw.get("version", "1.0")
        self.author = kw.get("author")
        self.tools = kw.get("tools", [])
        self.permissions = kw.get("permissions")
        self.invocation_policy = kw.get("policy")
        self.hooks = kw.get("hooks", [])
        self.certification = kw.get("certification", "unverified")
        self.input_schema = kw.get("input_schema")


# ── extract_side_effects / dominant_side_effect ───────────────────────


class TestSideEffects:
    def test_network(self):
        p = _Perms(network=_PermKind(enabled=True))
        assert extract_side_effects(p) == ["network"]

    def test_exec_plus_network(self):
        p = _Perms(exec_=_PermKind(enabled=True), network=_PermKind(enabled=True))
        assert "exec" in extract_side_effects(p)
        assert "network" in extract_side_effects(p)

    def test_filesystem_write(self):
        p = _Perms(filesystem=_PermKind(write=True))
        assert extract_side_effects(p) == ["filesystem"]

    def test_none(self):
        assert extract_side_effects(_Perms()) == []

    def test_dominant_exec_wins(self):
        p = _Perms(exec_=_PermKind(enabled=True), network=_PermKind(enabled=True))
        assert dominant_side_effect(p) == SIDE_EFFECT_EXEC

    def test_dominant_none(self):
        assert dominant_side_effect(_Perms()) == SIDE_EFFECT_NONE


# ── SkillSerializer.to_summary ────────────────────────────────────────


class TestToSummary:
    def setup_method(self):
        self.ser = SkillSerializer()

    def test_basic(self):
        s = _Skill(name="x", description="desc")
        d = self.ser.to_summary(s)
        assert d["name"] == "x"
        assert d["description"] == "desc"
        assert d["policy_action"] == POLICY_ALLOW
        assert d["side_effect"] == SIDE_EFFECT_NONE

    def test_with_tools(self):
        s = _Skill(tools=[_Tool("a"), _Tool("b")])
        d = self.ser.to_summary(s)
        assert d["tools"] == ["a", "b"]

    def test_side_effect_network(self):
        s = _Skill(permissions=_Perms(network=_PermKind(enabled=True)))
        d = self.ser.to_summary(s)
        assert d["side_effect"] == SIDE_EFFECT_NETWORK


# ── SkillSerializer.to_detail ─────────────────────────────────────────


class TestToDetail:
    def setup_method(self):
        self.ser = SkillSerializer()

    def test_includes_version(self):
        s = _Skill(version="2.0")
        d = self.ser.to_detail(s)
        assert d["version"] == "2.0"

    def test_default_version(self):
        s = _Skill(version=None)
        d = self.ser.to_detail(s)
        assert d["version"] == DEFAULT_VERSION

    def test_permissions_serialized(self):
        perms = _Perms(descs=["Network access"])
        s = _Skill(permissions=perms)
        d = self.ser.to_detail(s)
        assert len(d["permissions"]) == 1
        assert d["permissions"][0]["name"] == "Network access"

    def test_policy_serialized(self):
        s = _Skill(policy=_FakePolicy("allow_with_consent"))
        d = self.ser.to_detail(s)
        assert d["policy"]["default_action"] == "allow_with_consent"

    def test_hooks_serialized(self):
        class _H:
            hook_type = "pre_invoke"

        s = _Skill(hooks=[_H()])
        d = self.ser.to_detail(s)
        assert d["hooks"] == ["pre_invoke"]
