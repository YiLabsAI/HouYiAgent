"""Tests for houyi.utils.json_utils."""

from __future__ import annotations

from json import JSONDecodeError

import pytest

from houyi.utils.json_utils import (
    _extract_balanced_block,
    _extract_top_level_json_block,
    _find_first_json_start,
    _strip_markdown_fence,
    parse_embedded_json,
)


class TestStripMarkdownFence:
    def test_passthrough_no_fence(self) -> None:
        assert _strip_markdown_fence('{"a": 1}') == '{"a": 1}'

    def test_strips_code_block(self) -> None:
        assert _strip_markdown_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


class TestFindJsonStart:
    def test_prefers_earliest(self) -> None:
        assert _find_first_json_start('prefix [1,2,3] and {"a":1}') == 7
        assert _find_first_json_start('prefix {"a":1} and [1,2,3]') == 7

    def test_returns_neg_when_absent(self) -> None:
        assert _find_first_json_start("no json here") == -1


class TestBalancedBlock:
    def test_nested_with_strings(self) -> None:
        content = 'before {"items": [1, {"text": "a } brace"}], "ok": true} after'
        extracted = _extract_balanced_block(content, 7, "{", "}")
        assert extracted == '{"items": [1, {"text": "a } brace"}], "ok": true}'

    def test_escaped_quotes(self) -> None:
        content = 'before {"text": "quote: \\" and slash: \\\\ and } still inside string", "ok": true} after'
        extracted = _extract_balanced_block(content, 7, "{", "}")
        assert extracted == (
            '{"text": "quote: \\" and slash: \\\\ and } still inside string", "ok": true}'
        )

    def test_none_when_unbalanced(self) -> None:
        assert _extract_balanced_block('{"a": 1', 0, "{", "}") is None


class TestTopLevelBlock:
    def test_finds_array(self) -> None:
        assert _extract_top_level_json_block('answer: [1, {"x": 2}] trailing') == '[1, {"x": 2}]'


class TestParseEmbeddedJson:
    def test_direct_json(self) -> None:
        assert parse_embedded_json('{"a": 1}') == {"a": 1}

    def test_markdown_wrapped(self) -> None:
        content = '```json\n{"a": 1, "b": [2, 3]}\n```'
        assert parse_embedded_json(content) == {"a": 1, "b": [2, 3]}

    def test_extracts_from_text(self) -> None:
        content = 'Model output: {"answer": "ok", "count": 2} Thanks.'
        assert parse_embedded_json(content) == {"answer": "ok", "count": 2}

    def test_raises_when_no_json(self) -> None:
        with pytest.raises(JSONDecodeError):
            parse_embedded_json("not json at all")

    def test_picks_valid_fence_block(self) -> None:
        """LLM self-corrects with a second valid code fence."""
        content = (
            '```json\n{"bad": "missing closing\n```\n'
            "Oops, let me try again:\n"
            '```json\n{"good": "value"}\n```'
        )
        assert parse_embedded_json(content) == {"good": "value"}

    def test_truncation_repair(self) -> None:
        content = '{"a": 1} extra garbage that is not json'
        assert parse_embedded_json(content) == {"a": 1}

    def test_repair_degenerate_tail(self) -> None:
        """Valid JSON start followed by LLM degeneration."""
        content = '{"items": [1, 2]} \n\nLet me restart\n\n{"broken'
        assert parse_embedded_json(content) == {"items": [1, 2]}
