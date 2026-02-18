"""Tests for houyi.execution.tool_result — ToolResultBuilder."""

from __future__ import annotations

from houyi.execution.tool_result import ToolResultBuilder


class TestBuild:
    def test_dict_raw(self):
        result = ToolResultBuilder.build({"ok": True}, call_id="c1")
        assert result["call_id"] == "c1"
        assert result["raw"] == {"ok": True}
        assert result["is_error"] is False

    def test_error_detected(self):
        result = ToolResultBuilder.build({"error": "boom"})
        assert result["is_error"] is True

    def test_non_dict_wraps_as_result(self):
        result = ToolResultBuilder.build("hello")
        assert result["raw"] == {"result": "hello"}


class TestFormat:
    def test_extracts_content_key(self):
        result = {"content": '{"ok": true}'}
        assert ToolResultBuilder.format(result) == '{"ok": true}'

    def test_serializes_if_no_content(self):
        result = {"raw": {"ok": True}}
        assert '"ok"' in ToolResultBuilder.format(result)


class TestIsError:
    def test_is_error_flag(self):
        assert ToolResultBuilder.is_error({"is_error": True}) is True
        assert ToolResultBuilder.is_error({"is_error": False}) is False

    def test_raw_error_key(self):
        assert ToolResultBuilder.is_error({"raw": {"error": "boom"}}) is True
        assert ToolResultBuilder.is_error({"raw": {"ok": True}}) is False

    def test_non_dict(self):
        assert ToolResultBuilder.is_error("string") is False


class TestParseArguments:
    def test_none(self):
        assert ToolResultBuilder.parse_arguments(None) == {}

    def test_dict_passthrough(self):
        assert ToolResultBuilder.parse_arguments({"x": 1}) == {"x": 1}

    def test_json_string(self):
        assert ToolResultBuilder.parse_arguments('{"x": 1}') == {"x": 1}

    def test_bad_json(self):
        assert ToolResultBuilder.parse_arguments("not-json") == {}


class TestCoercePayload:
    def test_dict(self):
        assert ToolResultBuilder.coerce_payload({"a": 1}) == {"a": 1}

    def test_non_dict(self):
        assert ToolResultBuilder.coerce_payload("hello") == {"result": "hello"}
