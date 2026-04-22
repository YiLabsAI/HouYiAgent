"""Tests for SkillRegistry core tool protection.

Covers:
- Registration lock: core tools cannot be overwritten
- ext__ prefix renaming for conflicting external tools
- CoreToolProtectionError for illegal overwrite attempts
- as_tool_schemas() ordering: core tools first
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from houyi.domain.skill.registry import CoreToolProtectionError, SkillRegistry
from houyi.domain.skill.spec import SkillSpec


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    r: str


def _skill(name: str, is_core: bool = False, description: str = "desc") -> SkillSpec:
    return SkillSpec(
        name=name,
        description=description,
        input_schema=_In,
        output_schema=_Out,
        is_core=is_core,
    )


class TestCoreToolRegistration:
    """Core tool registration and protection tests."""

    def test_register_core_tool_succeeds(self) -> None:
        """A skill with is_core=True can be registered normally."""
        registry = SkillRegistry()
        skill = _skill("web_search", is_core=True)
        registry.register(skill)
        assert registry.get("web_search") is skill

    def test_register_non_core(self) -> None:
        """A normal (non-core) skill registers without issues."""
        registry = SkillRegistry()
        skill = _skill("helper_tool")
        registry.register(skill)
        assert registry.get("helper_tool") is skill

    def test_conflict_gets_renamed(self) -> None:
        """External tool with same name as core tool is auto-renamed to ext__<name>."""
        registry = SkillRegistry()
        core = _skill("web_search", is_core=True, description="Core search")
        ext = _skill("web_search", is_core=False, description="External search")
        registry.register(core)
        registry.register(ext)

        # Original name still points to core
        assert registry.get("web_search") is core
        # External tool registered under ext__ prefix
        renamed = registry.get("ext__web_search")
        assert renamed is not None
        assert renamed.name == "ext__web_search"
        assert renamed.description == "External search"

    def test_renamed_attribute_updated(self) -> None:
        """After rename, the ext__ skill object has name = 'ext__<original>'."""
        registry = SkillRegistry()
        registry.register(_skill("search", is_core=True))
        ext = _skill("search", description="External search")
        registry.register(ext)
        renamed = registry.get("ext__search")
        assert renamed is not None
        assert renamed.name == "ext__search"

    def test_no_conflict_registered(self) -> None:
        """External tool with no name conflict is registered without ext__ prefix."""
        registry = SkillRegistry()
        registry.register(_skill("web_search", is_core=True))
        ext = _skill("file_reader")  # different name, no conflict
        registry.register(ext)
        assert registry.get("file_reader") is ext
        assert registry.get("ext__file_reader") is None

    def test_overwrite_core_raises(self) -> None:
        """overwrite=True targeting a core tool must raise CoreToolProtectionError."""
        registry = SkillRegistry()
        registry.register(_skill("web_search", is_core=True))
        replacement = _skill("web_search", is_core=False)
        with pytest.raises(CoreToolProtectionError):
            registry.register(replacement, overwrite=True)

    def test_overwrite_non_core(self) -> None:
        """overwrite=True on a non-core tool should succeed as before."""
        registry = SkillRegistry()
        registry.register(_skill("helper"))
        replacement = _skill("helper", description="Updated helper")
        registry.register(replacement, overwrite=True)
        assert registry.get("helper") is replacement

    def test_two_cores_same_name(self) -> None:
        """Two core tools with the same name is a Host code error → ValueError."""
        registry = SkillRegistry()
        registry.register(_skill("web_search", is_core=True))
        with pytest.raises(ValueError):
            registry.register(_skill("web_search", is_core=True))

    def test_schemas_core_tools_first(self) -> None:
        """as_tool_schemas() must return core tools before non-core tools."""
        registry = SkillRegistry()
        registry.register(_skill("zzz_external"))  # non-core, registered first
        registry.register(_skill("aaa_core", is_core=True))
        schemas = registry.as_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        core_idx = names.index("aaa_core")
        ext_idx = names.index("zzz_external")
        assert core_idx < ext_idx, (
            f"Core tool 'aaa_core' (idx {core_idx}) should come before "
            f"non-core 'zzz_external' (idx {ext_idx})"
        )

    def test_ext_schema_third_party(self) -> None:
        """After rename to ext__, the tool schema has [THIRD-PARTY EXTENSION] prefix."""
        registry = SkillRegistry()
        registry.register(_skill("search", is_core=True))
        registry.register(_skill("search", description="External search"))
        schemas = registry.as_tool_schemas()
        ext_schema = next(s for s in schemas if s["function"]["name"] == "ext__search")
        assert ext_schema["function"]["description"].startswith("[THIRD-PARTY EXTENSION]")

    def test_core_schema_core_prefix(self) -> None:
        """Core tool schema has [CORE OFFICIAL TOOL] prefix in description."""
        registry = SkillRegistry()
        registry.register(_skill("search", is_core=True, description="Search the web"))
        schemas = registry.as_tool_schemas()
        core_schema = next(s for s in schemas if s["function"]["name"] == "search")
        assert core_schema["function"]["description"].startswith("[CORE OFFICIAL TOOL]")
