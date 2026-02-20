"""Unit tests for tool call adapter hooks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from houyi.core.tool_call_adapter_hooks import (
    _ADAPTER_HOOKS,
    ToolCallAdapterContext,
    resolve_tool_call_adapter,
)
from houyi.llm.models import (
    ADAPTER_FAKE,
    ADAPTER_REAL,
    PROVIDER_OPENAI_COMPAT,
    PROVIDER_SILICONFLOW,
)
from houyi.testkit.fake_adapter import FakeToolCallAdapter


def test_resolve_tool_call_adapter_fake() -> None:
    """Resolve should return fake adapter when adapter_name matches."""

    context = ToolCallAdapterContext(
        adapter_name=ADAPTER_FAKE,
        tool_names=[],
        skills=[],
        tool_sequence=["get_date"],
        parallel_tool_calls=None,
        now=datetime.now(UTC),
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
            adapter_name=ADAPTER_FAKE,
            tool_names=[],
            skills=[],
            tool_sequence=["get_date"],
            parallel_tool_calls=None,
            now=datetime.now(UTC),
        )
        adapter = resolve_tool_call_adapter(context)
        assert isinstance(adapter, FakeToolCallAdapter)
    finally:
        _ADAPTER_HOOKS[:] = original_hooks


def test_resolve_delegates_to_factory_for_known_provider() -> None:
    """Non-fake adapter_name should delegate to LLMAdapterFactory.create()."""

    sentinel = MagicMock(name="FactoryAdapter")
    with patch("houyi.llm.factory.LLMAdapterFactory.create", return_value=sentinel) as mock_create:
        context = ToolCallAdapterContext(
            adapter_name=PROVIDER_SILICONFLOW,
            tool_names=[],
            skills=[],
            tool_sequence=[],
            parallel_tool_calls=None,
            now=datetime.now(UTC),
        )
        adapter = resolve_tool_call_adapter(context)
        mock_create.assert_called_once_with(PROVIDER_SILICONFLOW)
        assert adapter is sentinel


def test_resolve_returns_none_for_real() -> None:
    """adapter_name='real' should return None (caller provides its own adapter)."""

    context = ToolCallAdapterContext(
        adapter_name=ADAPTER_REAL,
        tool_names=[],
        skills=[],
        tool_sequence=[],
        parallel_tool_calls=None,
        now=datetime.now(UTC),
    )
    adapter = resolve_tool_call_adapter(context)
    assert adapter is None


def test_resolve_returns_none_when_factory_fails() -> None:
    """If the factory raises, resolve should return None gracefully."""

    with patch(
        "houyi.llm.factory.LLMAdapterFactory.create",
        side_effect=ValueError("no key"),
    ):
        context = ToolCallAdapterContext(
            adapter_name=PROVIDER_OPENAI_COMPAT,
            tool_names=[],
            skills=[],
            tool_sequence=[],
            parallel_tool_calls=None,
            now=datetime.now(UTC),
        )
        adapter = resolve_tool_call_adapter(context)
        assert adapter is None
