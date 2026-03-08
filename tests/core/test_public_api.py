from __future__ import annotations

from houyi import Skill, SkillSpec, __version__, tool


class TestHouyiPublicApi:
    def test_root_exports_version_and_skill_alias(self) -> None:
        assert __version__ == "0.2.0"
        assert Skill is SkillSpec

    def test_tool_decorator_empty_input_schema(self) -> None:
        @tool
        def ping() -> str:
            """Ping tool."""
            return "pong"

        assert ping.name == "ping"
        assert ping.description == "Ping tool."
        assert ping.input_schema.model_fields == {}
        assert ping.input_schema.__name__ == "_EmptyToolInput"
        assert ping.output_schema.__name__ == "PingOutput"
        assert ping._original_func() == "pong"

    def test_tool_decorator_preserves_defaults_and_schema_names(self) -> None:
        @tool
        def search_web(query: str, limit: int = 5) -> list[str]:
            """Search the web."""
            return [query] * limit

        assert search_web.input_schema.__name__ == "SearchWebInput"
        assert search_web.output_schema.__name__ == "SearchWebOutput"
        assert search_web.input_schema.model_fields["query"].is_required()
        assert search_web.input_schema.model_fields["limit"].default == 5
        assert search_web.output_schema.model_fields["result"].annotation == list[str]
        assert search_web._original_func("x", 2) == ["x", "x"]
