"""Unit tests for execution/tool_node_utils.py."""

from pydantic import BaseModel

from houyi.execution.tool_node_utils import (
    build_inputs_from_context_values,
    extract_schema_fields,
    normalize_tool_name,
)


class TestNormalizeToolName:
    def test_normalize_tool_name(self):
        assert normalize_tool_name("  My Tool  ") == "my_tool"


class TestExtractSchemaFields:
    def test_extract_schema_fields_none(self):
        assert extract_schema_fields(None) == set()

    def test_extract_schema_fields_pydantic_v2(self):
        class Input(BaseModel):
            a: int
            b: str

        assert extract_schema_fields(Input) == {"a", "b"}


class TestBuildInputsFromContextValues:
    def test_build_inputs_from_context_values_prefers_direct_context(self):
        schema_fields = {"a", "b"}
        context_values = {"a": 1, "result": {"a": 2, "b": 3}}
        built = build_inputs_from_context_values(
            schema_fields=schema_fields,
            context_values=context_values,
        )
        assert built == {"a": 1, "b": 3}

    def test_build_inputs_from_context_values_parses_json_result(self):
        schema_fields = {"a"}
        context_values = {"result": '{"a": 5}'}
        built = build_inputs_from_context_values(
            schema_fields=schema_fields,
            context_values=context_values,
        )
        assert built == {"a": 5}

    def test_build_inputs_from_context_values_handles_nested_result(self):
        schema_fields = {"a"}
        context_values = {"result": {"result": {"a": 7}}}
        built = build_inputs_from_context_values(
            schema_fields=schema_fields,
            context_values=context_values,
        )
        assert built == {"a": 7}
