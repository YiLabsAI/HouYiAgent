"""Tests for tool-call adapter registry and normalization."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from houyi.adapters.llm.base import LLMResponse
from houyi.application.tool_calling.tool_call_adapter import (
    normalize_adapter_error,
    normalize_adapter_response,
)
from houyi.application.tool_calling.tool_call_adapter_hooks import (
    _ADAPTER_HOOKS,
    ToolCallAdapterContext,
)
from houyi.application.tool_calling.tool_call_adapter_registry import (
    ToolCallAdapterRegistry,
    ToolCallAdapterRequest,
)


class DummyAdapter:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, _messages, tools=None, **_kwargs):
        self.calls += 1
        return LLMResponse(
            content="ok",
            tool_calls=[],
            finish_reason="stop",
            usage={},
            model="test-model",
        )


def _build_request(adapter_name: str) -> ToolCallAdapterRequest:
    return ToolCallAdapterRequest(
        adapter_name=adapter_name,
        context=ToolCallAdapterContext(
            skills=[],
            tool_sequence=[],
            parallel_tool_calls=None,
            now=datetime.now(UTC),
        ),
    )


@pytest.mark.asyncio
async def test_registry_uses_hook(monkeypatch) -> None:
    """Registry should prefer hook adapters and wrap normalized adapter."""

    adapter = DummyAdapter()

    def hook(_context: ToolCallAdapterContext):
        return adapter

    original_hooks = list(_ADAPTER_HOOKS)
    _ADAPTER_HOOKS.clear()
    _ADAPTER_HOOKS.append(hook)
    try:
        registry = ToolCallAdapterRegistry()
        resolved = registry.resolve(
            _build_request("custom"),
            fallback_factory=lambda: DummyAdapter(),
        )
        response = await resolved.chat(messages=[], tools=[])
        assert isinstance(response, LLMResponse)
        assert resolved.inner is adapter
    finally:
        _ADAPTER_HOOKS.clear()
        _ADAPTER_HOOKS.extend(original_hooks)


@pytest.mark.asyncio
async def test_registry_fallback() -> None:
    """Registry should fall back to factory when hooks return None."""

    adapter = DummyAdapter()
    registry = ToolCallAdapterRegistry()
    resolved = registry.resolve(
        _build_request("missing"),
        fallback_factory=lambda: adapter,
    )
    response = await resolved.chat(messages=[], tools=[])
    assert isinstance(response, LLMResponse)
    assert resolved.inner is adapter


def test_normalize_adapter_error() -> None:
    """Normalize adapter errors to a stable shape."""

    error = normalize_adapter_error(ValueError("boom"))
    assert error.error_type == "ValueError"
    assert error.retryable is False


def test_normalize_adapter_response_openai() -> None:
    """Normalize OpenAI-style responses to LLMResponse."""

    class _Message:
        def __init__(self) -> None:
            self.content = "ok"
            self.tool_calls = []
            self.function_call = None

    class _Choice:
        def __init__(self) -> None:
            self.message = _Message()
            self.finish_reason = "stop"

    class _Usage:
        def __init__(self) -> None:
            self.prompt_tokens = 1
            self.completion_tokens = 1
            self.total_tokens = 2

    class _Response:
        def __init__(self) -> None:
            self.choices = [_Choice()]
            self.usage = _Usage()
            self.model = "test-model"

    result = normalize_adapter_response(_Response())
    assert isinstance(result, LLMResponse)
    assert result.content == "ok"


def test_normalize_adapter_response_anthropic() -> None:
    """Normalize Anthropic-style responses to LLMResponse."""

    class _Block:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _Usage:
        def __init__(self) -> None:
            self.input_tokens = 1
            self.output_tokens = 1

    class _Response:
        def __init__(self) -> None:
            self.content = [_Block("ok")]
            self.stop_reason = "stop"
            self.usage = _Usage()
            self.model = "test-model"

    result = normalize_adapter_response(_Response())
    assert isinstance(result, LLMResponse)
    assert result.content == "ok"


def test_normalize_adapter_response_invalid() -> None:
    """Raise on unsupported adapter response type."""

    try:
        normalize_adapter_response(object())
    except TypeError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("Expected TypeError")
