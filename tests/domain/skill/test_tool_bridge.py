"""Tests for ToolBridge shared tool schema collection."""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from houyi.application.tool_calling.tool_bridge import ToolBridge, build_tool_definitions_for_skill
from houyi.domain.skill.registry import SkillRegistry
from houyi.domain.skill.spec import SkillSpec


class _InputA(BaseModel):
    query: str


class _InputB(BaseModel):
    path: str


class _InputRich(BaseModel):
    path: str
    pattern: str
    recursive: bool = False
    limit: int = 20


class _Output(BaseModel):
    ok: bool = True


def _make_skill(
    name: str,
    description: str,
    *,
    version: str = "1.0.0",
    is_core: bool = False,
    tags: list[str] | None = None,
    input_schema: type[BaseModel] = _InputA,
) -> SkillSpec:
    metadata = {"tags": tags or []}
    return SkillSpec(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=_Output,
        version=version,
        is_core=is_core,
        metadata=metadata,
    )


class TestBuildToolDefinitionsForSkill:
    def test_builds_single_tool(self) -> None:
        skill = _make_skill("search_docs", "Search project docs")

        definitions = build_tool_definitions_for_skill(skill)

        assert len(definitions) == 1
        assert definitions[0]["function"]["name"] == "search_docs"
        assert definitions[0]["function"]["description"] == "Search project docs"
        assert definitions[0]["function"]["parameters"]["type"] == "object"

    def test_builds_multiple_tools(self) -> None:
        tool_a = SimpleNamespace(name="read_file", description="Read file", input_schema=_InputA)
        tool_b = SimpleNamespace(
            name="list_dir", description="List directory", input_schema=_InputB
        )
        skill_like = SimpleNamespace(tools=[tool_a, tool_b])

        definitions = build_tool_definitions_for_skill(skill_like)

        assert [item["function"]["name"] for item in definitions] == ["read_file", "list_dir"]


class TestToolBridgeCollection:
    def test_collect_skills_with_filter(self) -> None:
        registry = SkillRegistry()
        alpha = _make_skill("alpha", "Alpha skill")
        registry.register(alpha)
        bridge = ToolBridge(registry)

        selected = bridge.collect_skills(skill_filter=["alpha", "missing"], include_core=False)

        assert selected == [alpha]

    def test_collect_skills_with_alias(self) -> None:
        registry = SkillRegistry()
        web_search = _make_skill("web_search", "Search the web")
        registry.register(web_search)
        bridge = ToolBridge(registry)

        selected = bridge.collect_skills(
            skill_filter=["houyi_web_search"],
            include_core=False,
        )

        assert selected == [web_search]

    def test_collect_tool_schemas_filters(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill("search_docs", "Search docs", tags=["docs", "search"]))
        registry.register(_make_skill("run_terminal", "Execute terminal command", tags=["shell"]))
        bridge = ToolBridge(registry)

        schemas = bridge.collect_tool_schemas(relevance_hint="need docs search")

        names = [item["function"]["name"] for item in schemas]
        assert names == ["search_docs"]

    def test_collect_tool_schemas_relevance(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill("search_docs", "Search docs"))
        registry.register(_make_skill("run_terminal", "Execute terminal command"))
        bridge = ToolBridge(registry)

        schemas = bridge.collect_tool_schemas(relevance_hint="completely unrelated token")

        assert {item["function"]["name"] for item in schemas} == {"search_docs", "run_terminal"}

    def test_collect_tool_schemas_orders(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill("search_docs", "Search docs"))
        registry.register(_make_skill("run_terminal", "Execute terminal command"))
        bridge = ToolBridge(registry)

        schemas = bridge.collect_tool_schemas(
            usage_counts={"run_terminal": 5, "search_docs": 1},
        )

        names = [item["function"]["name"] for item in schemas]
        assert names[0] == "run_terminal"

    def test_collect_tool_schemas_cache(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_skill("search_docs", "Search docs"))
        bridge = ToolBridge(registry)

        first = bridge.collect_tool_schemas(skill_filter=["search_docs"])
        first[0]["function"]["description"] = "mutated"

        second = bridge.collect_tool_schemas(skill_filter=["search_docs"])

        assert second[0]["function"]["description"] == "Search docs"

    def test_collect_tool_schemas_minimal_exposure(self) -> None:
        registry = SkillRegistry()
        registry.register(
            _make_skill(
                "search_docs",
                "Search docs",
                input_schema=_InputRich,
            )
        )
        bridge = ToolBridge(registry)

        schemas = bridge.collect_tool_schemas(
            skill_filter=["search_docs"],
            schema_exposure="minimal",
        )

        parameters = schemas[0]["function"]["parameters"]
        assert set(parameters["properties"].keys()) <= {"path", "pattern", "recursive", "limit"}
        assert len(parameters["properties"]) <= 3
        assert parameters["required"] == ["path", "pattern"]

    def test_collect_tool_schemas_full_exposure_by_default(self) -> None:
        registry = SkillRegistry()
        registry.register(
            _make_skill(
                "search_docs",
                "Search docs",
                input_schema=_InputRich,
            )
        )
        bridge = ToolBridge(registry)

        schemas = bridge.collect_tool_schemas(skill_filter=["search_docs"])

        parameters = schemas[0]["function"]["parameters"]
        assert set(parameters["properties"].keys()) == {"path", "pattern", "recursive", "limit"}
