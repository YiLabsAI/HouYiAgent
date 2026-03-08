"""Tests for houyi.application.tool_calling.placeholder_resolver."""

from __future__ import annotations

from houyi.application.tool_calling.placeholder_resolver import PlaceholderResolver


class TestContains:
    def test_simple_tool_ref(self):
        assert PlaceholderResolver.contains("$tool.get_date") is True

    def test_simple_call_ref(self):
        assert PlaceholderResolver.contains("$call.c1.result") is True

    def test_plain_string(self):
        assert PlaceholderResolver.contains("hello") is False

    def test_nested_dict(self):
        assert PlaceholderResolver.contains({"x": "$tool.t1.value"}) is True
        assert PlaceholderResolver.contains({"x": "plain"}) is False

    def test_nested_list(self):
        assert PlaceholderResolver.contains(["$tool.t1"]) is True
        assert PlaceholderResolver.contains(["plain"]) is False

    def test_non_string(self):
        assert PlaceholderResolver.contains(42) is False
        assert PlaceholderResolver.contains(None) is False


class TestExtract:
    def test_tool_prefix(self):
        assert PlaceholderResolver.extract("$tool.get_date") == ("get_date", [])

    def test_tool_with_path(self):
        assert PlaceholderResolver.extract("$tool.t1.value") == ("t1", ["value"])

    def test_call_prefix(self):
        assert PlaceholderResolver.extract("$call.c1.result.0") == ("c1", ["result", "0"])

    def test_no_prefix(self):
        assert PlaceholderResolver.extract("plain") is None

    def test_empty_path(self):
        assert PlaceholderResolver.extract("$tool.") is None


class TestResolvePath:
    def test_dict_path(self):
        assert PlaceholderResolver.resolve_path({"a": {"b": 42}}, ["a", "b"]) == 42

    def test_list_index(self):
        assert PlaceholderResolver.resolve_path([10, 20, 30], ["1"]) == 20

    def test_missing_key_returns_root(self):
        payload = {"a": 1}
        assert PlaceholderResolver.resolve_path(payload, ["missing"]) is payload

    def test_empty_path(self):
        assert PlaceholderResolver.resolve_path({"a": 1}, []) == {"a": 1}


class TestResolve:
    def test_resolves_tool_ref(self):
        outputs = {"get_date": "2026-01-01"}
        assert PlaceholderResolver.resolve("$tool.get_date", outputs) == "2026-01-01"

    def test_resolves_nested_dict(self):
        outputs = {"t1": {"value": 42}}
        args = {"x": "$tool.t1.value", "y": "literal"}
        result = PlaceholderResolver.resolve(args, outputs)
        assert result == {"x": 42, "y": "literal"}

    def test_unresolvable_returns_original(self):
        assert PlaceholderResolver.resolve("$tool.missing", {}) == "$tool.missing"

    def test_non_placeholder_passthrough(self):
        assert PlaceholderResolver.resolve("hello", {}) == "hello"
        assert PlaceholderResolver.resolve(42, {}) == 42
