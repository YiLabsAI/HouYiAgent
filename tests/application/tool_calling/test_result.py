"""Tests for houyi.application.tool_calling.tool_results — ToolResultBuilder."""

from __future__ import annotations

from houyi.application.tool_calling.tool_results import ToolResultBuilder


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


class TestSerialize:
    def test_basic_dict(self):
        out = ToolResultBuilder.serialize({"ok": True})
        assert '"ok": true' in out

    def test_non_ascii_preserved(self):
        """Non-ASCII characters should NOT be escaped to \\uXXXX."""
        out = ToolResultBuilder.serialize({"title": "caf\u00e9"})
        assert "caf\u00e9" in out
        assert "\\u" not in out

    def test_newlines_in_content(self):
        """Newlines should appear as \\n in JSON output (JSON spec), not double-escaped."""
        out = ToolResultBuilder.serialize({"text": "line1\nline2"})
        assert "line1\\nline2" in out

    def test_fallback_for_non_serializable(self):
        out = ToolResultBuilder.serialize(object())
        assert '"result"' in out


class TestCoercePayload:
    def test_dict(self):
        assert ToolResultBuilder.coerce_payload({"a": 1}) == {"a": 1}

    def test_non_dict(self):
        assert ToolResultBuilder.coerce_payload("hello") == {"result": "hello"}
