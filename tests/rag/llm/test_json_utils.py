"""Tests for JSON helper behavior in RAG LLM utilities."""

from __future__ import annotations

from json import JSONDecodeError

import pytest

from houyi.rag.llm.json_utils import (
    _extract_balanced_block,
    _extract_top_level_json_block,
    _find_first_json_start,
    _strip_markdown_fence,
    parse_embedded_json,
)


def test_strip_markdown_fence_returns_original_without_fence() -> None:
    assert _strip_markdown_fence('{"a": 1}') == '{"a": 1}'


def test_strip_markdown_fence_removes_code_block_wrappers() -> None:
    content = '```json\n{"a": 1}\n```'

    assert _strip_markdown_fence(content) == '{"a": 1}'


def test_find_first_json_start_prefers_earliest_token() -> None:
    assert _find_first_json_start('prefix [1,2,3] and {"a":1}') == 7
    assert _find_first_json_start('prefix {"a":1} and [1,2,3]') == 7
    assert _find_first_json_start("no json here") == -1


def test_extract_balanced_block_handles_nested_json_and_strings() -> None:
    content = 'before {"items": [1, {"text": "a } brace"}], "ok": true} after'

    extracted = _extract_balanced_block(content, 7, "{", "}")

    assert extracted == '{"items": [1, {"text": "a } brace"}], "ok": true}'


def test_extract_balanced_block_handles_escaped_quotes_and_backslashes() -> None:
    content = (
        'before {"text": "quote: \\" and slash: \\\\ and } still inside string", "ok": true} after'
    )

    extracted = _extract_balanced_block(content, 7, "{", "}")

    assert extracted == (
        '{"text": "quote: \\" and slash: \\\\ and } still inside string", "ok": true}'
    )


def test_extract_balanced_block_returns_none_when_unbalanced() -> None:
    assert _extract_balanced_block('{"a": 1', 0, "{", "}") is None


def test_extract_top_level_json_block_finds_array() -> None:
    content = 'answer: [1, {"x": 2}] trailing'

    assert _extract_top_level_json_block(content) == '[1, {"x": 2}]'


def test_parse_embedded_json_parses_direct_json() -> None:
    assert parse_embedded_json('{"a": 1}') == {"a": 1}


def test_parse_embedded_json_parses_markdown_wrapped_json() -> None:
    content = '```json\n{"a": 1, "b": [2, 3]}\n```'

    assert parse_embedded_json(content) == {"a": 1, "b": [2, 3]}


def test_parse_embedded_json_extracts_top_level_block_from_text() -> None:
    content = 'Model output: {"answer": "ok", "count": 2} Thanks.'

    assert parse_embedded_json(content) == {"answer": "ok", "count": 2}


def test_parse_embedded_json_raises_when_no_json_block_exists() -> None:
    with pytest.raises(JSONDecodeError):
        parse_embedded_json("not json at all")
