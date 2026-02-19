"""Tests for NodeExecutionFlow._merge_tool_call_into_context (Issue 1 refactor)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STUDIO_SERVER_ROOT = _REPO_ROOT / "houyi-studio" / "server"
sys.path.insert(0, str(_STUDIO_SERVER_ROOT))

from houyi_studio.server.execution.node_execution_flow import NodeExecutionFlow  # noqa: E402

_merge = NodeExecutionFlow._merge_tool_call_into_context


def test_dict_raw_with_dict_payload_merges_keys() -> None:
    ctx: dict = {}
    call = {
        "tool_name": "web_search",
        "args": {"query": "test"},
        "result": {"raw": {"result": {"url": "http://x", "title": "T"}}},
    }
    _merge(call, ctx)
    assert ctx == {"query": "test", "url": "http://x", "title": "T"}


def test_non_dict_raw_stored_under_tool_name() -> None:
    ctx: dict = {}
    call = {"tool_name": "get_date", "args": {}, "result": {"raw": "2024-01-01"}}
    _merge(call, ctx)
    assert ctx == {"get_date": "2024-01-01", "date": "2024-01-01"}


def test_non_dict_call_is_noop() -> None:
    ctx: dict = {}
    _merge("not a dict", ctx)
    assert ctx == {}


def test_none_raw_only_merges_args() -> None:
    ctx: dict = {}
    call = {"tool_name": "x", "args": {"a": 1}, "result": {"raw": None}}
    _merge(call, ctx)
    assert ctx == {"a": 1}


def test_no_result_key_only_merges_args() -> None:
    ctx: dict = {}
    call = {"tool_name": "x", "args": {"b": 2}}
    _merge(call, ctx)
    assert ctx == {"b": 2}


def test_existing_keys_not_overwritten() -> None:
    ctx: dict = {"query": "original"}
    call = {
        "tool_name": "web_search",
        "args": {"query": "new"},
        "result": {"raw": {"result": {"query": "from_result"}}},
    }
    _merge(call, ctx)
    assert ctx["query"] == "original"


def test_dict_raw_non_dict_payload_stored_under_tool_name() -> None:
    ctx: dict = {}
    call = {
        "tool_name": "summarize",
        "args": {},
        "result": {"raw": {"result": "summary text"}},
    }
    _merge(call, ctx)
    assert ctx == {"summarize": "summary text"}


def test_get_date_special_handling_with_dict_raw() -> None:
    ctx: dict = {}
    call = {
        "tool_name": "get_date",
        "args": {},
        "result": {"raw": {"result": "2025-06-01"}},
    }
    _merge(call, ctx)
    assert ctx["get_date"] == "2025-06-01"
    assert ctx["date"] == "2025-06-01"
