"""Tests for ToolBridge shared tool schema collection."""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from houyi.core.skill.spec import SkillSpec
from houyi.core.skill.tool_bridge import ToolBridge, build_tool_definitions_for_skill
from houyi.core.skill_registry import SkillRegistry


class _InputA(BaseModel):
    query: str


class _InputB(BaseModel):
    path: str


class _Output(BaseModel):
    ok: bool = True


def _make_skill(
    name: str,
    description: str,
    *,
    version: str = "1.0.0",
    is_core: bool = False,
    tags: list[str] | None = None,
) -> SkillSpec:
    metadata = {"tags": tags or []}
    return SkillSpec(
        name=name,
        description=description,
        input_schema=_InputA,
        output_schema=_Output,
        version=version,
        is_core=is_core,
        metadata=metadata,
    )


class TestBuildToolDefinitionsForSkill:
    def test_builds_single_tool_from_skill_schema(self) -> None:
        skill = _make_skill("search_docs", "Search project docs")

        definitions = build_tool_definitions_for_skill(skill)

        assert len(definitions) == 1
        assert definitions[0]["function"]["name"] == "search_docs"
        assert definitions[0]["function"]["description"] == "Search project docs"
        assert definitions[0]["function"]["parameters"]["type"] == "object"

    def test_builds_multiple_tools_from_skill_tools_list(self) -> None:
        tool_a = SimpleNamespace(name="read_file", description="Read file", input_schema=_InputA)
        tool_b = SimpleNamespace(
            name="list_dir", description="List directory", input_schema=_InputB
        )
        skill_like = SimpleNamespace(tools=[tool_a, tool_b])

        definitions = build_tool_definitions_for_skill(skill_like)

        assert [item["function"]["name"] for item in definitions] == ["read_file", "list_dir"]


class TestToolBridgeCollection:
    def test_collect_skills_with_filter_skips_missing_names(self) -> None:
        registry = SkillRegistry()
        alpha = _make_skill("alpha", "Alpha skill")
        registry.register(alpha)
        bridge = ToolBridge(registry)

        selected = bridge.collect_skills(skill_filter=["alpha", "missing"], include_core=False)

        assert selected == [alpha]

    def test_collect_tool_schemas_filters_by_relevance_hint(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill("search_docs", "Search docs", tags=["docs", "search"]))
        registry.register(_make_skill("run_terminal", "Execute terminal command", tags=["shell"]))
        bridge = ToolBridge(registry)

        schemas = bridge.collect_tool_schemas(relevance_hint="need docs search")

        names = [item["function"]["name"] for item in schemas]
        assert names == ["search_docs"]

    def test_collect_tool_schemas_relevance_fallback_returns_all_when_no_match(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill("search_docs", "Search docs"))
        registry.register(_make_skill("run_terminal", "Execute terminal command"))
        bridge = ToolBridge(registry)

        schemas = bridge.collect_tool_schemas(relevance_hint="completely unrelated token")

        assert {item["function"]["name"] for item in schemas} == {"search_docs", "run_terminal"}

    def test_collect_tool_schemas_orders_by_usage_counts(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill("search_docs", "Search docs"))
        registry.register(_make_skill("run_terminal", "Execute terminal command"))
        bridge = ToolBridge(registry)

        schemas = bridge.collect_tool_schemas(
            usage_counts={"run_terminal": 5, "search_docs": 1},
        )

        names = [item["function"]["name"] for item in schemas]
        assert names[0] == "run_terminal"

    def test_collect_tool_schemas_cache_returns_immutable_copies(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill("search_docs", "Search docs"))
        bridge = ToolBridge(registry)

        first = bridge.collect_tool_schemas(skill_filter=["search_docs"])
        first[0]["function"]["description"] = "mutated"

        second = bridge.collect_tool_schemas(skill_filter=["search_docs"])

        assert second[0]["function"]["description"] == "Search docs"
