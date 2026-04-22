"""Tests for RuntimeContract parsing and integration level computation.

Covers:
- RuntimeContract model validation (mode, adapter, hooks_root)
- SkillSpec.from_file() parsing of `runtime` frontmatter field
- IntegrationLevel computation (metadata / schema / executable)
- RuntimeStatus computation (ready / degraded / unavailable)
- Degradation path when frontmatter is missing or malformed
- Forward compatibility: unknown runtime sub-fields preserved
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from houyi.domain.skill.runtime_contract import (
    CapabilityTier,
    RuntimeContract,
    RuntimeMode,
    RuntimeStatus,
)
from houyi.domain.skill.spec import SkillSpec

# ── RuntimeContract model tests ──────────────────────────────────────


class TestRuntimeContract:
    """Unit tests for RuntimeContract Pydantic model."""

    def test_valid_tool_mode(self):
        rc = RuntimeContract(mode=RuntimeMode.TOOL, entry="my_tool")
        assert rc.mode == RuntimeMode.TOOL
        assert rc.entry == "my_tool"
        assert rc.adapter is None

    def test_script_mode_with_adapter(self):
        rc = RuntimeContract(
            mode=RuntimeMode.SCRIPT,
            adapter="houyi.skills.planning.adapter:execute",
            hooks_root="houyi/skills/planning",
        )
        assert rc.mode == RuntimeMode.SCRIPT
        assert rc.adapter == "houyi.skills.planning.adapter:execute"
        assert rc.hooks_root == "houyi/skills/planning"

    def test_valid_template_mode(self):
        rc = RuntimeContract(mode=RuntimeMode.TEMPLATE)
        assert rc.mode == RuntimeMode.TEMPLATE

    def test_default_mode_is_tool(self):
        rc = RuntimeContract()
        assert rc.mode == RuntimeMode.TOOL

    def test_from_dict_valid(self):
        data = {
            "mode": "script",
            "adapter": "my.module:func",
            "hooks_root": "skills/my-skill",
        }
        rc = RuntimeContract.from_dict(data)
        assert rc.mode == RuntimeMode.SCRIPT
        assert rc.adapter == "my.module:func"

    def test_from_dict_empty(self):
        rc = RuntimeContract.from_dict({})
        assert rc.mode == RuntimeMode.TOOL

    def test_none_returns_none(self):
        assert RuntimeContract.from_dict(None) is None

    def test_invalid_mode_falls_back(self):
        rc = RuntimeContract.from_dict({"mode": "unknown_mode"})
        assert rc is not None
        assert rc.mode == RuntimeMode.TOOL
        assert "unknown_mode" in rc.extra.get("original_mode", "")

    def test_preserves_unknown_fields(self):
        data = {"mode": "tool", "custom_field": "value123"}
        rc = RuntimeContract.from_dict(data)
        assert rc.extra["custom_field"] == "value123"

    def test_adapter_format_validation(self):
        rc = RuntimeContract(adapter="module.path:function_name")
        assert rc.adapter == "module.path:function_name"

    def test_to_dict_roundtrip(self):
        original = RuntimeContract(
            mode=RuntimeMode.SCRIPT,
            adapter="my.mod:fn",
            hooks_root="skills/x",
        )
        d = original.to_dict()
        restored = RuntimeContract.from_dict(d)
        assert restored.mode == original.mode
        assert restored.adapter == original.adapter
        assert restored.hooks_root == original.hooks_root


# ── CapabilityTier / RuntimeStatus tests ──────────────────────────────


class TestCapabilityTier:
    """Tests for integration level enum values."""

    def test_ordering(self):
        assert CapabilityTier.METADATA.value < CapabilityTier.SCHEMA.value
        assert CapabilityTier.SCHEMA.value < CapabilityTier.EXECUTABLE.value


class TestRuntimeStatus:
    """Tests for runtime status enum values."""

    def test_values(self):
        assert RuntimeStatus.READY == "ready"
        assert RuntimeStatus.DEGRADED == "degraded"
        assert RuntimeStatus.UNAVAILABLE == "unavailable"


# ── SkillSpec integration tests ──────────────────────────────────────


class TestSkillSpecRuntimeParsing:
    """Tests for SkillSpec.from_file() parsing of runtime frontmatter."""

    def _write_skill(self, tmp_path: Path, frontmatter: str, body: str = "# Test") -> str:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir(exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        content = f"---\n{frontmatter}\n---\n{body}"
        skill_file.write_text(content)
        return str(skill_file)

    def test_no_runtime_field(self, tmp_path):
        """Skill without runtime field has runtime_contract=None."""
        path = self._write_skill(tmp_path, "name: test\ndescription: A test")
        skill = SkillSpec.from_file(path)
        assert skill.runtime_contract is None

    def test_runtime_tool_mode(self, tmp_path):
        fm = textwrap.dedent("""\
            name: test
            description: A test
            runtime:
              mode: tool
              entry: my_tool""")
        path = self._write_skill(tmp_path, fm)
        skill = SkillSpec.from_file(path)
        assert skill.runtime_contract is not None
        assert skill.runtime_contract.mode == RuntimeMode.TOOL
        assert skill.runtime_contract.entry == "my_tool"

    def test_parsed_script_with_adapter(self, tmp_path):
        fm = textwrap.dedent("""\
            name: test
            description: A test
            runtime:
              mode: script
              adapter: houyi.skills.test.adapter:execute
              hooks_root: skills/test""")
        path = self._write_skill(tmp_path, fm)
        skill = SkillSpec.from_file(path)
        rc = skill.runtime_contract
        assert rc is not None
        assert rc.mode == RuntimeMode.SCRIPT
        assert rc.adapter == "houyi.skills.test.adapter:execute"
        assert rc.hooks_root == "skills/test"

    def test_invalid_dict_degrades(self, tmp_path):
        fm = textwrap.dedent("""\
            name: test
            description: A test
            runtime:
              mode: nonexistent_mode""")
        path = self._write_skill(tmp_path, fm)
        skill = SkillSpec.from_file(path)
        assert skill.runtime_contract is not None
        assert skill.runtime_contract.mode == RuntimeMode.TOOL

    def test_runtime_string_value_ignored(self, tmp_path):
        """runtime: 'some string' should not crash, just be ignored."""
        fm = "name: test\ndescription: A test\nruntime: some_string"
        path = self._write_skill(tmp_path, fm)
        skill = SkillSpec.from_file(path)
        assert skill.runtime_contract is None

    def test_runtime_not_in_extra(self, tmp_path):
        """runtime dict should NOT leak into extra_frontmatter."""
        fm = textwrap.dedent("""\
            name: test
            description: A test
            runtime:
              mode: tool""")
        path = self._write_skill(tmp_path, fm)
        skill = SkillSpec.from_file(path)
        assert "runtime" not in skill.extra_frontmatter


# ── Integration level computation on SkillSpec ───────────────────────


class TestSkillSpecIntegrationLevel:
    """Tests for SkillSpec.capability_tier property."""

    def test_metadata_only(self):
        """Skill with no schema and no executor is metadata-only."""
        from pydantic import BaseModel

        class EmptyInput(BaseModel):
            pass

        class EmptyOutput(BaseModel):
            pass

        skill = SkillSpec(
            name="t",
            description="d",
            input_schema=EmptyInput,
            output_schema=EmptyOutput,
        )
        assert skill.capability_tier == CapabilityTier.METADATA

    def test_schema_level(self):
        """Skill with real input_schema but no executor is schema-level."""
        from pydantic import BaseModel

        class MyInput(BaseModel):
            query: str

        class MyOutput(BaseModel):
            result: str

        skill = SkillSpec(
            name="t",
            description="d",
            input_schema=MyInput,
            output_schema=MyOutput,
        )
        assert skill.capability_tier == CapabilityTier.SCHEMA

    def test_executable_level(self):
        """Skill with executor bound is executable."""
        from pydantic import BaseModel

        class MyInput(BaseModel):
            query: str

        class MyOutput(BaseModel):
            result: str

        skill = SkillSpec(
            name="t",
            description="d",
            input_schema=MyInput,
            output_schema=MyOutput,
            executor=lambda x: x,
        )
        assert skill.capability_tier == CapabilityTier.EXECUTABLE


class TestSkillSpecRuntimeStatus:
    """Tests for SkillSpec.runtime_status property."""

    def test_ready_with_executor(self):
        from pydantic import BaseModel

        class InputModel(BaseModel):
            q: str

        class OutputModel(BaseModel):
            r: str

        skill = SkillSpec(
            name="t",
            description="d",
            input_schema=InputModel,
            output_schema=OutputModel,
            executor=lambda x: x,
        )
        assert skill.runtime_status == RuntimeStatus.READY

    def test_unavailable_no_executor(self):
        from pydantic import BaseModel

        class E(BaseModel):
            pass

        skill = SkillSpec(name="t", description="d", input_schema=E, output_schema=E)
        assert skill.runtime_status == RuntimeStatus.UNAVAILABLE

    def test_degraded_schema_no_executor(self):
        from pydantic import BaseModel

        class InputModel(BaseModel):
            q: str

        class OutputModel(BaseModel):
            r: str

        skill = SkillSpec(
            name="t", description="d", input_schema=InputModel, output_schema=OutputModel
        )
        assert skill.runtime_status == RuntimeStatus.DEGRADED
