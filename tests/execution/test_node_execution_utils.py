"""Unit tests for execution/node_execution_utils.py."""

from houyi.execution.node_execution_utils import (
    extract_output_payload,
    resolve_inputs,
    resolve_value,
)


class TestResolveValue:
    def test_resolves_string_reference(self):
        assert resolve_value("$a", {"a": 1}) == 1
        assert resolve_value("$missing", {}) == "$missing"

    def test_resolves_nested_dict_and_list(self):
        value = {"x": "$a", "y": ["$b", 2], "z": {"k": "$c"}}
        resolved = resolve_value(value, {"a": 1, "b": "two", "c": 3})
        assert resolved == {"x": 1, "y": ["two", 2], "z": {"k": 3}}


class TestResolveInputs:
    def test_resolve_inputs_applies_resolve_value(self):
        inputs = {"a": "$x", "b": 2}
        assert resolve_inputs(inputs, {"x": 1}) == {"a": 1, "b": 2}


class TestExtractOutputPayload:
    def test_extracts_nested_output_result(self):
        outputs = {"output": {"result": {"a": 1}}}
        assert extract_output_payload(outputs) == {"a": 1}

    def test_extracts_output_object(self):
        outputs = {"output": {"a": 1}}
        assert extract_output_payload(outputs) == {"a": 1}

    def test_extracts_result_object(self):
        outputs = {"result": {"a": 1}}
        assert extract_output_payload(outputs) == {"a": 1}

    def test_falls_back_to_outputs(self):
        assert extract_output_payload({"a": 1}) == {"a": 1}
        assert extract_output_payload(None) == {}
