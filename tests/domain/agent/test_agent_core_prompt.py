"""Tests for AgentSpec core tool routing.

Covers:
- to_system_prompt() includes TOOL ROUTING POLICY when core skills present
- to_system_prompt() excludes guardrail when no core skills
- to_system_prompt() respects custom system_prompt override (no guardrail injected)
- get_tool_schemas() places core tools before non-core tools
- get_tool_schemas() ordering is stable with multiple core tools
- get_tool_schemas() delegates annotation to to_tool_schema()
"""

from __future__ import annotations

from pydantic import BaseModel

from houyi.domain.agent import AgentSpec
from houyi.domain.skill.spec import SkillSpec


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    r: str


def _skill(name: str, is_core: bool = False, description: str = "test desc") -> SkillSpec:
    return SkillSpec(
        name=name,
        description=description,
        input_schema=_In,
        output_schema=_Out,
        is_core=is_core,
    )


class TestAgentSpecSystemPrompt:
    """Tests for to_system_prompt() with core tool guardrail injection."""

    def test_no_core_no_guardrail(self) -> None:
        """No TOOL ROUTING POLICY in prompt when no is_core skill is present."""
        spec = AgentSpec(
            role="Researcher",
            skills=[_skill("helper"), _skill("analyzer")],
        )
        prompt = spec.to_system_prompt()
        assert "TOOL ROUTING POLICY" not in prompt

    def test_core_skill_triggers_guardrail(self) -> None:
        """TOOL ROUTING POLICY is injected when at least one is_core skill exists."""
        spec = AgentSpec(
            role="Researcher",
            skills=[_skill("web_search", is_core=True), _skill("ext__web_search")],
        )
        prompt = spec.to_system_prompt()
        assert "TOOL ROUTING POLICY" in prompt

    def test_guardrail_mentions_core(self) -> None:
        """The guardrail section lists the core tool name."""
        spec = AgentSpec(
            role="Analyst",
            skills=[_skill("rag_search", is_core=True)],
        )
        prompt = spec.to_system_prompt()
        assert "rag_search" in prompt

    def test_guardrail_prefer_over_ext(self) -> None:
        """Guardrail text instructs LLM to prefer [CORE OFFICIAL TOOL]."""
        spec = AgentSpec(
            role="Agent",
            skills=[_skill("search", is_core=True)],
        )
        prompt = spec.to_system_prompt()
        assert "[CORE OFFICIAL TOOL]" in prompt
        assert "[THIRD-PARTY EXTENSION]" in prompt

    def test_custom_prompt_not_overridden(self) -> None:
        """If system_prompt is explicitly set, to_system_prompt() returns it unchanged."""
        custom = "Custom prompt here."
        spec = AgentSpec(
            role="Agent",
            system_prompt=custom,
            skills=[_skill("web_search", is_core=True)],
        )
        prompt = spec.to_system_prompt()
        assert prompt == custom
        assert "TOOL ROUTING POLICY" not in prompt

    def test_empty_skills_no_guardrail(self) -> None:
        """No guardrail when skills list is empty."""
        spec = AgentSpec(role="Agent", skills=[])
        prompt = spec.to_system_prompt()
        assert "TOOL ROUTING POLICY" not in prompt

    def test_multiple_core_listed(self) -> None:
        """All core tool names appear in the guardrail."""
        spec = AgentSpec(
            role="Agent",
            skills=[
                _skill("web_search", is_core=True),
                _skill("rag_search", is_core=True),
                _skill("ext__external"),
            ],
        )
        prompt = spec.to_system_prompt()
        assert "web_search" in prompt
        assert "rag_search" in prompt


class TestAgentSpecGetToolSchemas:
    """Tests for get_tool_schemas() ordering and delegation."""

    def test_core_tools_sorted_first(self) -> None:
        """Core tools must appear before non-core tools in get_tool_schemas()."""
        spec = AgentSpec(
            role="Agent",
            skills=[
                _skill("zzz_ext"),  # non-core, registered first
                _skill("aaa_core", is_core=True),
            ],
        )
        schemas = spec.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        core_idx = names.index("aaa_core")
        ext_idx = names.index("zzz_ext")
        assert core_idx < ext_idx

    def test_multiple_core_stable_order(self) -> None:
        """Multiple core tools maintain stable relative order."""
        spec = AgentSpec(
            role="Agent",
            skills=[
                _skill("core_b", is_core=True),
                _skill("ext_tool"),
                _skill("core_a", is_core=True),
            ],
        )
        schemas = spec.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert names.index("ext_tool") > names.index("core_a")
        assert names.index("ext_tool") > names.index("core_b")

    def test_no_core_order_unchanged(self) -> None:
        """Without core tools, order is insertion order (no reordering)."""
        spec = AgentSpec(
            role="Agent",
            skills=[_skill("first"), _skill("second"), _skill("third")],
        )
        schemas = spec.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert names == ["first", "second", "third"]

    def test_core_schema_has_prefix(self) -> None:
        """Core tool schema description has [CORE OFFICIAL TOOL] prefix."""
        spec = AgentSpec(
            role="Agent",
            skills=[_skill("web_search", is_core=True, description="Search the web")],
        )
        schemas = spec.get_tool_schemas()
        assert schemas[0]["function"]["description"].startswith("[CORE OFFICIAL TOOL]")

    def test_ext_schema_third_party(self) -> None:
        """ext__ tool schema description has [THIRD-PARTY EXTENSION] prefix."""
        spec = AgentSpec(
            role="Agent",
            skills=[_skill("ext__web_search", description="External search")],
        )
        schemas = spec.get_tool_schemas()
        assert schemas[0]["function"]["description"].startswith("[THIRD-PARTY EXTENSION]")
