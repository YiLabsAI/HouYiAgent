"""Unit tests for tool call adapter hooks."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pytest

from houyi.core.tool_call_adapter_hooks import (
    _ADAPTER_HOOKS,
    FakeToolCallAdapter,
    ToolCallAdapterContext,
    resolve_tool_call_adapter,
)


def test_resolve_tool_call_adapter_fake() -> None:
    """Resolve should return fake adapter when adapter_name matches."""

    context = ToolCallAdapterContext(
        adapter_name="fake",
        tool_names=[],
        skills=[],
        tool_sequence=["get_date"],
        parallel_tool_calls=None,
        now=datetime.now(timezone.utc),
    )
    adapter = resolve_tool_call_adapter(context)
    assert isinstance(adapter, FakeToolCallAdapter)


@pytest.mark.asyncio
async def test_fake_tool_call_adapter_parallel_calls() -> None:
    """FakeToolCallAdapter should emit parallel tool calls when enabled."""

    adapter = FakeToolCallAdapter(["get_date", "get_weather"], now=datetime(2024, 1, 1))
    response = await adapter.chat([], parallel_tool_calls=True)
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 2
    args = json.loads(response.tool_calls[0]["function"]["arguments"])
    assert args["offset_days"] == "tomorrow"


@pytest.mark.asyncio
async def test_fake_tool_call_adapter_complete_sequence() -> None:
    """FakeToolCallAdapter should return final response after sequence ends."""

    adapter = FakeToolCallAdapter(["get_date"], now=datetime(2024, 1, 1))
    await adapter.chat([])
    response = await adapter.chat([])
    assert response.content == "Done."


def test_resolve_tool_call_adapter_handles_hook_error() -> None:
    """Resolve should ignore hook errors and continue."""

    original_hooks = list(_ADAPTER_HOOKS)

    def _bad_hook(_context: ToolCallAdapterContext):
        raise RuntimeError("boom")

    try:
        _ADAPTER_HOOKS.insert(0, _bad_hook)
        context = ToolCallAdapterContext(
            adapter_name="fake",
            tool_names=[],
            skills=[],
            tool_sequence=["get_date"],
            parallel_tool_calls=None,
            now=datetime.now(timezone.utc),
        )
        adapter = resolve_tool_call_adapter(context)
        assert isinstance(adapter, FakeToolCallAdapter)
    finally:
        _ADAPTER_HOOKS[:] = original_hooks


def test_resolve_tool_call_adapter_openai_compat(monkeypatch) -> None:
    """Resolve should return openai_compat adapter when configured."""

    class _Adapter:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args

    monkeypatch.setitem(
        sys.modules,
        "houyi.llm.openai_compat_adapter",
        type("_mod", (), {"OpenAICompatibleAdapter": _Adapter}),
    )
    context = ToolCallAdapterContext(
        adapter_name="openai_compat",
        tool_names=[],
        skills=[],
        tool_sequence=[],
        parallel_tool_calls=None,
        now=datetime.now(timezone.utc),
    )
    adapter = resolve_tool_call_adapter(context)
    assert isinstance(adapter, _Adapter)


def test_resolve_tool_call_adapter_vertex(monkeypatch) -> None:
    """Resolve should return Vertex adapter via from_env hook."""

    class _Adapter:
        @classmethod
        def from_env(cls):
            return "vertex"

    monkeypatch.setitem(
        sys.modules,
        "houyi.llm.vertex_gemini_adapter",
        type("_mod", (), {"GoogleVertexGeminiAdapter": _Adapter}),
    )
    context = ToolCallAdapterContext(
        adapter_name="vertex_gemini",
        tool_names=[],
        skills=[],
        tool_sequence=[],
        parallel_tool_calls=None,
        now=datetime.now(timezone.utc),
    )
    adapter = resolve_tool_call_adapter(context)
    assert adapter == "vertex"
