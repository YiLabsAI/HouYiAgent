"""Covers Anthropic adapter parameter shaping, import fallback, and streaming usage capture."""

from __future__ import annotations

import builtins
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.adapters.llm.anthropic_adapter import AnthropicAdapter


def _build_adapter() -> AnthropicAdapter:
    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=AsyncMock()))
    fake_anthropic_module = types.SimpleNamespace(AsyncAnthropic=lambda **_: client)
    with patch.dict("sys.modules", {"anthropic": fake_anthropic_module}):
        return AnthropicAdapter(api_key="test-key")


class TestAnthropicAdapterHelpers:
    def test_split_keeps_last_system(self):
        adapter = _build_adapter()

        system_message, filtered_messages = adapter._split_system_message(
            [
                {"role": "system", "content": "first"},
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "final system"},
                {"role": "assistant", "content": "hi"},
            ]
        )

        assert system_message == "final system"
        assert filtered_messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_build_params_converts_tools(self):
        adapter = _build_adapter()

        params = adapter._build_anthropic_params(
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                },
                {"type": "noop"},
            ],
            temperature=0.3,
            max_tokens=None,
            extra_kwargs={"top_p": 0.9},
        )

        assert params["model"] == "claude-3-5-sonnet-20241022"
        assert params["messages"] == [{"role": "user", "content": "hello"}]
        assert params["temperature"] == 0.3
        assert params["max_tokens"] == 4096
        assert params["tools"] == [
            {
                "name": "weather",
                "description": "Get weather",
                "input_schema": {"type": "object"},
            }
        ]
        assert params["top_p"] == 0.9

    def test_convert_skips_non_function(self):
        adapter = _build_adapter()

        tools = adapter._convert_tools_to_anthropic(
            [
                {"type": "function", "function": {"name": "search", "parameters": {}}},
                {"type": "other", "function": {"name": "ignored"}},
            ]
        )

        assert tools == [{"name": "search", "description": "", "input_schema": {}}]

    def test_build_params_with_system(self):
        adapter = _build_adapter()

        params = adapter._build_anthropic_params(
            messages=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
            tools=None,
            temperature=0.5,
            max_tokens=32,
            extra_kwargs={},
        )

        assert params["system"] == "be concise"
        assert params["messages"] == [{"role": "user", "content": "hello"}]

    def test_init_without_package(self):
        original_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "anthropic":
                raise ImportError("missing anthropic")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_fake_import):
            with pytest.raises(ImportError, match="Anthropic package not installed"):
                AnthropicAdapter(api_key="test-key")


class TestAnthropicAdapterChat:
    @pytest.mark.asyncio
    async def test_chat_returns_normalized(self):
        adapter = _build_adapter()

        usage = types.SimpleNamespace(input_tokens=3, output_tokens=4)
        response = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="hello")],
            stop_reason="end_turn",
            usage=usage,
            model="claude-test",
        )
        adapter.client.messages.create = AsyncMock(return_value=response)

        result = await adapter.chat(
            [{"role": "system", "content": "be concise"}, {"role": "user", "content": "hi"}],
            max_tokens=20,
            top_p=0.9,
        )

        call_kwargs = adapter.client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "be concise"
        assert call_kwargs["max_tokens"] == 20
        assert call_kwargs["top_p"] == 0.9
        assert result.content == "hello"
        assert result.usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


class TestAnthropicAdapterStreaming:
    @pytest.mark.asyncio
    async def test_stream_yields_text_usage(self):
        adapter = _build_adapter()

        usage = types.SimpleNamespace(input_tokens=4, output_tokens=6)
        final_message = types.SimpleNamespace(stop_reason="end_turn", usage=usage)

        class _StreamContext:
            def __init__(self) -> None:
                self.text_stream = self._iter_text()

            async def _iter_text(self):
                for text in ["hello", " world"]:
                    yield text

            async def get_final_message(self):
                return final_message

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        adapter.client.messages.stream = MagicMock(return_value=_StreamContext())

        chunks = []
        async for chunk in adapter.stream_chat([{"role": "user", "content": "hi"}]):
            chunks.append(chunk.content_delta)

        assert chunks == ["hello", " world"]
        assert adapter.last_finish_reason == "end_turn"
        assert adapter.last_usage == {
            "prompt_tokens": 4,
            "completion_tokens": 6,
            "total_tokens": 10,
        }
