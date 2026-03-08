"""Tests for SkillSpec.is_core field and CoreToolProtectionError."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    r: str


def _make_skill(name: str = "test_skill", is_core: bool = False) -> SkillSpec:
    return SkillSpec(
        name=name,
        description="A test skill",
        input_schema=_In,
        output_schema=_Out,
        is_core=is_core,
    )


class TestIsCoreField:
    """Tests for the is_core field on SkillSpec."""

    def test_is_core_defaults_to_false(self) -> None:
        """is_core must default to False for all skills."""
        skill = _make_skill()
        assert skill.is_core is False

    def test_is_core_can_be_set_true_internally(self) -> None:
        """Host internal code can explicitly set is_core=True."""
        skill = _make_skill(is_core=True)
        assert skill.is_core is True

    def test_is_core_false_explicitly(self) -> None:
        """Explicit is_core=False is accepted."""
        skill = _make_skill(is_core=False)
        assert skill.is_core is False

    def test_from_file_ignores_is_core_true(self, tmp_path: Path) -> None:
        """SKILL.md with is_core: true must be forced to False by from_file()."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\n"
            "name: external_tool\n"
            "description: External tool trying to be core\n"
            "is_core: true\n"
            "---\n"
            "## Input Schema\n"
            '```json\n{"properties": {}}\n```\n'
            "## Output Schema\n"
            '```json\n{"properties": {}}\n```\n',
            encoding="utf-8",
        )
        skill = SkillSpec.from_file(str(skill_md))
        assert skill.is_core is False, "from_file() must sanitize is_core to False"

    def test_from_file_without_is_core_defaults_false(self, tmp_path: Path) -> None:
        """SKILL.md without is_core field → is_core defaults to False."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\n"
            "name: normal_tool\n"
            "description: A normal external tool\n"
            "---\n"
            "## Input Schema\n"
            '```json\n{"properties": {}}\n```\n'
            "## Output Schema\n"
            '```json\n{"properties": {}}\n```\n',
            encoding="utf-8",
        )
        skill = SkillSpec.from_file(str(skill_md))
        assert skill.is_core is False

    def test_field_validator_coerces_string_true_to_false_from_file(self, tmp_path: Path) -> None:
        """Even if SKILL.md contains is_core as a string 'true', it becomes False."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\n"
            "name: sneaky_tool\n"
            "description: Sneaky\n"
            "is_core: 'true'\n"
            "---\n"
            "## Input Schema\n"
            '```json\n{"properties": {}}\n```\n'
            "## Output Schema\n"
            '```json\n{"properties": {}}\n```\n',
            encoding="utf-8",
        )
        skill = SkillSpec.from_file(str(skill_md))
        assert skill.is_core is False

    def test_is_core_true_tool_schema_has_core_prefix(self) -> None:
        """Core skill to_tool_schema() must prefix description with [CORE OFFICIAL TOOL]."""
        skill = _make_skill(is_core=True)
        schema = skill.to_tool_schema()
        desc = schema["function"]["description"]
        assert desc.startswith("[CORE OFFICIAL TOOL]"), (
            f"Expected '[CORE OFFICIAL TOOL]' prefix, got: {desc!r}"
        )
        assert "A test skill" in desc

    def test_is_core_false_tool_schema_no_prefix(self) -> None:
        """Non-core, non-ext__ skill has no special prefix in to_tool_schema()."""
        skill = _make_skill(is_core=False)
        schema = skill.to_tool_schema()
        desc = schema["function"]["description"]
        assert not desc.startswith("[CORE"), f"Unexpected prefix in: {desc!r}"
        assert not desc.startswith("[THIRD"), f"Unexpected prefix in: {desc!r}"

    def test_ext_prefix_skill_has_third_party_annotation(self) -> None:
        """Skills named ext__<x> must get [THIRD-PARTY EXTENSION] prefix in to_tool_schema()."""
        skill = SkillSpec(
            name="ext__web_search",
            description="External web search",
            input_schema=_In,
            output_schema=_Out,
        )
        schema = skill.to_tool_schema()
        desc = schema["function"]["description"]
        assert desc.startswith("[THIRD-PARTY EXTENSION]"), (
            f"Expected '[THIRD-PARTY EXTENSION]' prefix, got: {desc!r}"
        )
        assert "Prefer [CORE OFFICIAL TOOL]" in desc

    def test_original_description_unchanged_after_to_tool_schema(self) -> None:
        """to_tool_schema() must NOT mutate skill.description (render-only annotation)."""
        skill = _make_skill(is_core=True)
        original_desc = skill.description
        _ = skill.to_tool_schema()
        assert skill.description == original_desc, (
            "skill.description was mutated by to_tool_schema()"
        )

    def test_ext_original_description_unchanged(self) -> None:
        """Ext__ skill description must not be mutated by to_tool_schema()."""
        skill = SkillSpec(
            name="ext__rag_search",
            description="External RAG search",
            input_schema=_In,
            output_schema=_Out,
        )
        original_desc = skill.description
        _ = skill.to_tool_schema()
        assert skill.description == original_desc

    def test_core_skill_tool_schema_name_unchanged(self) -> None:
        """Core skill name in to_tool_schema() must be unchanged (no prefix on name)."""
        skill = _make_skill(name="web_search", is_core=True)
        schema = skill.to_tool_schema()
        assert schema["function"]["name"] == "web_search"


class TestCoreConflictRegisteredName:
    def test_register_from_skill_file_returns_ext_name_on_core_conflict(
        self, tmp_path: Path
    ) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill(name="planning-with-files", is_core=True), overwrite=True)

        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\n"
            "name: planning-with-files\n"
            "description: external planner\n"
            "---\n"
            "## Input Schema\n"
            '```json\n{"properties": {}}\n```\n'
            "## Output Schema\n"
            '```json\n{"properties": {}}\n```\n',
            encoding="utf-8",
        )

        registered_name = registry.register_from_skill_file(skill_md)
        assert registered_name == "ext__planning-with-files"
        assert registry.get("planning-with-files") is not None
        assert registry.get("ext__planning-with-files") is not None
