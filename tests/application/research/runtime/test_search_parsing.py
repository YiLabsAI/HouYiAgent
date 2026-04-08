from __future__ import annotations

import json

from houyi.application.research.runtime.search_parsing import (
    _canonical_query,
    _parse_query_list,
    _parse_sufficiency,
)


class TestParseQueryList:
    def test_valid_json_array(self):
        assert _parse_query_list('["a", "b"]') == ["a", "b"]

    def test_fenced_code_block(self):
        assert _parse_query_list('```json\n["a"]\n```') == ["a"]

    def test_plain_text_fallback(self):
        assert _parse_query_list("just a query") == ["just a query"]

    def test_numbered_format(self):
        payload = (
            "Thoughts...\n"
            "**Query 1:** first focused query\n"
            "**Query 2:** second focused query\n"
            "**Query 3:** third focused query"
        )
        assert _parse_query_list(payload) == [
            "first focused query",
            "second focused query",
            "third focused query",
        ]

    def test_truncates_long_query(self):
        parsed = _parse_query_list(json.dumps(["x" * 500]))
        assert len(parsed) == 1
        assert len(parsed[0]) == 380


class TestParseSufficiency:
    def test_true(self):
        ok, _ = _parse_sufficiency('{"sufficient": true, "rationale": "ok"}')
        assert ok is True

    def test_false(self):
        ok, _ = _parse_sufficiency('{"sufficient": false, "rationale": "need more"}')
        assert ok is False

    def test_malformed(self):
        ok, _ = _parse_sufficiency("garbage")
        assert ok is False


class TestCanonicalQuery:
    def test_lowercases(self):
        assert _canonical_query("AI Frameworks") == "ai frameworks"

    def test_empty_string(self):
        assert _canonical_query("") == ""

    def test_collapses_whitespace(self):
        assert _canonical_query("  ai   frameworks  ") == "ai frameworks"
