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
        self.is_core = kw.get("is_core", False)
        self.extra_frontmatter = kw.get("extra_frontmatter", {})
        self.skill_md_path = kw.get("skill_md_path")
        self.instructions = kw.get("instructions")


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

    def test_source_builtin_for_core(self):
        s = _Skill(name="web_search", is_core=True)
        d = self.ser.to_summary(s)
        assert d["source"] == "builtin"

    def test_source_third_party_for_ext_prefix(self):
        s = _Skill(name="ext__web_search")
        d = self.ser.to_summary(s)
        assert d["source"] == "third_party"
        assert d["is_external_alias"] is True
        assert d["alias_target"] == "web_search"

    def test_source_community_from_skills_directory_path(self):
        s = _Skill(skill_md_path="/tmp/project/skills/some-skill/SKILL.md")
        d = self.ser.to_summary(s)
        assert d["source"] == "community"

    def test_source_trust_source_override(self):
        s = _Skill(extra_frontmatter={"trust": {"source": "community"}})
        d = self.ser.to_summary(s)
        assert d["source"] == "community"

    def test_summary_includes_runtime_binding_and_instruction_length(self):
        s = _Skill(instructions="## steps\nuse tools")
        d = self.ser.to_summary(s)
        assert d["runtime_binding"] == "prompt_instructions"
        assert d["instructions_length"] > 0


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

    def test_version_author_from_frontmatter_when_missing(self, monkeypatch):
        s = _Skill(version=None, author=None)
        monkeypatch.setattr(
            SkillSerializer,
            "_resolve_frontmatter_meta",
            staticmethod(lambda _skill: {"version": "0.9.1", "author": "Team"}),
        )
        d = self.ser.to_detail(s)
        assert d["version"] == "0.9.1"
        assert d["author"] == "Team"

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
        assert d["hooks"] == ["hook:* (pre_invoke)"]

    def test_instructions_and_hook_specs_serialized(self):
        class _H:
            event = "PreToolUse"
            hook_type = "command"
            matcher = "Write|Edit"
            command = "echo hi"
            handler_path = None

        s = _Skill(instructions="body", hooks=[_H()])
        d = self.ser.to_detail(s)
        assert d["instructions"] == "body"
        assert d["hook_specs"][0]["matcher"] == "Write|Edit"
        assert d["hook_specs"][0]["command"] == "echo hi"


# ── capability_tier / runtime_status serialization ─────────────────


class TestCapabilityTierAndRuntimeStatus:
    """Verify serializer emits capability_tier and runtime_status fields."""

    def setup_method(self):
        self.ser = SkillSerializer()

    def test_summary_includes_capability_tier_and_runtime_status(self):
        from pydantic import BaseModel

        from houyi.core.skill.spec import SkillSpec

        class _In(BaseModel):
            q: str

        class _Out(BaseModel):
            r: str

        skill = SkillSpec(
            name="test-skill",
            description="test",
            input_schema=_In,
            output_schema=_Out,
        )
        skill.bind_executor(lambda **kw: kw)
        d = self.ser.to_summary(skill)
        assert "capability_tier" in d
        assert "runtime_status" in d
        assert d["capability_tier"] == "executable"
        assert d["runtime_status"] == "ready"

    def test_detail_includes_capability_tier_and_runtime_status(self):
        from pydantic import BaseModel

        from houyi.core.skill.spec import SkillSpec

        class _In(BaseModel):
            q: str

        class _Out(BaseModel):
            r: str

        skill = SkillSpec(
            name="test-skill",
            description="test",
            input_schema=_In,
            output_schema=_Out,
        )
        d = self.ser.to_detail(skill)
        assert d["capability_tier"] == "schema"
        assert d["runtime_status"] == "degraded"

    def test_metadata_only_skill(self):
        from pydantic import BaseModel

        from houyi.core.skill.spec import SkillSpec

        class _Empty(BaseModel):
            pass

        skill = SkillSpec(
            name="meta-only",
            description="no schema no executor",
            input_schema=_Empty,
            output_schema=_Empty,
        )
        d = self.ser.to_summary(skill)
        assert d["capability_tier"] == "metadata"
        assert d["runtime_status"] == "unavailable"

    def test_fallback_on_fake_skill_without_property(self):
        """Fake skill objects without computed properties get safe defaults."""
        s = _Skill(name="fake")
        d = self.ser.to_summary(s)
        assert d["capability_tier"] == "metadata"
        assert d["runtime_status"] == "unavailable"
