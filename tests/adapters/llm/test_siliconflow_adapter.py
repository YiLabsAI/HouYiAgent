"""Covers SiliconFlow adapter mock fallback, client/httpx paths, retries, and stream parsing."""

from __future__ import annotations

import json
import os
import re
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.adapters.llm.siliconflow_adapter import (
    SiliconFlowAdapter,
    _format_siliconflow_http_error,
)
from houyi.infrastructure.config.env_config import EnvConfig


@pytest.fixture(autouse=True)
def _reset_env_config():
    """Reset EnvConfig singleton before/after each test so env patches take effect."""
    EnvConfig._reset()
    yield
    EnvConfig._reset()


class TestSiliconFlowAdapterMockMode:
    """Test SiliconFlowAdapter when no API key is set (mock mode)."""

    @pytest.mark.asyncio
    async def test_mock_streaming(self):
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                SiliconFlowAdapter._OPENAI_READY = None
                adapter = SiliconFlowAdapter()
                assert adapter.api_key is None

                messages = [{"role": "user", "content": "Hello world"}]
                chunks = []
                async for chunk in adapter.stream_chat(messages):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

                assert len(chunks) > 0
                full_content = "".join(c for c, _ in chunks)
                assert "Mock response" in full_content
                assert all(r is None for _, r in chunks)

    @pytest.mark.asyncio
    async def test_stream_completion(self):
        """stream_completion should delegate to stream_chat."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                SiliconFlowAdapter._OPENAI_READY = None
                adapter = SiliconFlowAdapter()

                chunks = []
                async for chunk in adapter.stream_completion("Test prompt"):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

                assert len(chunks) > 0
                full = "".join(c for c, _ in chunks)
                assert "Mock response" in full

    @pytest.mark.asyncio
    async def test_mock_extracts_user(self):
        """Mock mode should use last user message content."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                SiliconFlowAdapter._OPENAI_READY = None
                adapter = SiliconFlowAdapter()

                messages = [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Tell me about Python"},
                ]
                chunks = []
                async for chunk in adapter.stream_chat(messages):
                    chunks.append(chunk.content_delta)

                full = "".join(chunks)
                assert "Tell me about Python" in full


class TestSiliconFlowAdapterRoutePath:
    """Test SiliconFlowAdapter route selection with mocked openai client."""

    @pytest.mark.asyncio
    async def test_stream_prefers_httpx_route(self):
        """stream_chat should route through httpx even if client support is available."""

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = True
            adapter = SiliconFlowAdapter()

            async def _fake_httpx_stream(*args, **kwargs):
                yield type("Chunk", (), {"content_delta": "Hello", "reasoning_delta": None})()
                yield type("Chunk", (), {"content_delta": " world", "reasoning_delta": None})()
                yield type("Chunk", (), {"content_delta": "", "reasoning_delta": "thinking"})()

            with (
                patch.object(adapter, "_stream_request", AsyncMock()) as direct_stream,
                patch.object(
                    adapter, "_stream_request_httpx", side_effect=_fake_httpx_stream
                ) as httpx_stream,
            ):
                chunks = []
                async for chunk in adapter.stream_chat([{"role": "user", "content": "Hi"}]):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

        assert len(chunks) == 3
        assert chunks[0] == ("Hello", None)
        assert chunks[1] == (" world", None)
        assert chunks[2] == ("", "thinking")
        direct_stream.assert_not_called()
        httpx_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_prefers_httpx(self):
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = True
            adapter = SiliconFlowAdapter()

        async def _fake_httpx_stream(*args, **kwargs):
            yield type("Chunk", (), {"content_delta": "ok", "reasoning_delta": None})()

        with (
            patch.object(adapter, "_stream_request", AsyncMock()) as direct_stream,
            patch.object(
                adapter, "_stream_request_httpx", side_effect=_fake_httpx_stream
            ) as httpx_stream,
        ):
            chunks = []
            async for chunk in adapter.stream_chat([{"role": "user", "content": "Hi"}]):
                chunks.append((chunk.content_delta, chunk.reasoning_delta))

        assert chunks == [("ok", None)]
        direct_stream.assert_not_called()
        httpx_stream.assert_called_once()


class TestSiliconFlowAdapterErrors:
    def test_balance_error(self):
        message = _format_siliconflow_http_error(
            403,
            '{"code":30001,"message":"Sorry, your account balance is insufficient","data":null}',
        )
        assert "insufficient balance or credits" in message


class TestSiliconFlowAdapterHttpxPath:
    """Test SiliconFlowAdapter httpx fallback path."""

    @pytest.mark.asyncio
    async def test_stream_httpx_path(self):
        """Mock the httpx path to verify stream_chat behavior."""

        sse_lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": "Hi"}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": " there"}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"reasoning_content": "let me think"}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
                    "choices": [],
                }
            ),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, *args, **kwargs):
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for chunk in adapter.stream_chat([{"role": "user", "content": "Hi"}]):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

        assert len(chunks) == 3
        assert chunks[0] == ("Hi", None)
        assert chunks[1] == (" there", None)
        assert chunks[2] == ("", "let me think")
        assert adapter.last_usage == {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "total_tokens": 11,
        }


class TestSiliconFlowAdapterReasoning:
    """Test client path with reasoning enabled (covers extra_body and kwargs branches)."""

    @pytest.mark.asyncio
    async def test_stream_reasoning_kwargs(self):
        """stream_chat should pass reasoning kwargs through the httpx path."""

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = True
            adapter = SiliconFlowAdapter()

            async def _fake_httpx_stream(*args, **kwargs):
                yield type("Chunk", (), {"content_delta": "A", "reasoning_delta": None})()

            with (
                patch.object(adapter, "_stream_request", AsyncMock()) as direct_stream,
                patch.object(
                    adapter, "_stream_request_httpx", side_effect=_fake_httpx_stream
                ) as httpx_stream,
            ):
                chunks = []
                async for chunk in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}],
                    model="deepseek-chat",
                    enable_thinking=True,
                    thinking_budget=1024,
                    temperature=0.7,
                ):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

        assert chunks == [("A", None)]
        direct_stream.assert_not_called()
        httpx_stream.assert_called_once()
        call_args = httpx_stream.call_args
        request = call_args.args[0]
        assert request.model == "deepseek-chat"
        assert request.enable_thinking is True
        assert request.thinking_budget == 1024
        assert request.temperature == 0.7


class TestSiliconFlowHttpxEdgeCases:
    """Test httpx path edge cases: empty lines, bad JSON, missing delta."""

    @pytest.mark.asyncio
    async def test_httpx_edge_cases(self):
        sse_lines = [
            "",  # empty line
            "event: ping",  # non-data line
            "data: ",  # empty data
            "data: not-json",  # bad JSON
            "data: " + json.dumps({"choices": []}),  # empty choices
            "data: " + json.dumps({"choices": [{"delta": None}]}),  # null delta
            "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"reasoning_content": "think"}}],
                }
            ),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, *args, **kwargs):
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for chunk in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}],
                    enable_thinking=True,
                    thinking_budget=512,
                ):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

        assert len(chunks) == 2
        assert chunks[0] == ("OK", None)
        assert chunks[1] == ("", "think")


class TestSiliconFlowChatRequestSanitization:
    """Test non-stream chat payload sanitation for strict OpenAI-compatible providers."""

    @pytest.mark.asyncio
    async def test_chat_sanitizes_messages(self):
        captured: dict[str, object] = {}

        class MockResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "model": "test-model",
                    "choices": [
                        {"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            text = ""

        class MockHttpxClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["body"] = json
                captured["headers"] = headers
                return MockResponse()

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            messages = [
                {"role": "user", "content": [{"type": "text", "text": "search file skill.md"}]},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "houyi_grep",
                                "arguments": {"query": "skill.md"},
                            },
                        }
                    ],
                },
            ]

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                await adapter.chat(messages=messages, model="test-model")

        body = captured["body"]
        assert isinstance(body, dict)
        payload_messages = body["messages"]
        assert isinstance(payload_messages, list)
        assert payload_messages[0]["content"] == "search file skill.md"
        args = payload_messages[1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert "skill.md" in args
        assert payload_messages[1]["reasoning_content"] == ""

    @pytest.mark.asyncio
    async def test_chat_preserves_extra_body_for_reasoning(self):
        captured: dict[str, object] = {}

        class MockResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "model": "deepseek-reasoner",
                    "choices": [
                        {"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            text = ""

        class MockHttpxClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["body"] = json
                captured["headers"] = headers
                return MockResponse()

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                await adapter.chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="deepseek-reasoner",
                    enable_thinking=True,
                    thinking_budget=256,
                )

        body = captured["body"]
        assert isinstance(body, dict)
        assert body["extra_body"] == {"thinking_budget": 256}
        assert "thinking_budget" not in body

    @pytest.mark.asyncio
    async def test_chat_keeps_sampling(self):
        captured: dict[str, object] = {}

        class MockResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "model": "test-model",
                    "choices": [
                        {"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            text = ""

        class MockHttpxClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, json=None, headers=None):
                captured["body"] = json
                return MockResponse()

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                await adapter.chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="test-model",
                    top_k=16,
                    frequency_penalty=0.5,
                )

        body = captured["body"]
        assert isinstance(body, dict)
        assert body["top_k"] == 16
        assert body["frequency_penalty"] == 0.5


class TestSiliconFlowStreamingRequestSanitization:
    """Test stream_chat request payload sanitation for strict providers."""

    @pytest.mark.asyncio
    async def test_stream_sanitizes_messages(self):
        captured_payload: dict[str, object] = {}

        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, *args, **kwargs):
                captured_payload["json"] = kwargs.get("json")
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            messages = [
                {"role": "user", "content": [{"type": "text", "text": "search file skill.md"}]},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "houyi_grep",
                                "arguments": {"query": "skill.md"},
                            },
                        }
                    ],
                },
            ]

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for chunk in adapter.stream_chat(messages=messages, model="test-model"):
                    chunks.append(chunk.content_delta)

        assert "OK" in "".join(chunks)
        payload = captured_payload["json"]
        assert isinstance(payload, dict)
        payload_messages = payload["messages"]
        assert payload_messages[0]["content"] == "search file skill.md"
        args = payload_messages[1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert "skill.md" in args
        assert payload_messages[1]["reasoning_content"] == ""


class TestSiliconFlowAdapterHelpers:
    def test_build_request(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        request = adapter._build_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": {"q": "hi"}},
                        }
                    ],
                }
            ],
            tools=[{"type": "function", "function": {"name": "search"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=True,
            kwargs={
                "model": "deepseek-reasoner",
                "tool_choice": "required",
                "top_p": 0.8,
                "thinking_budget": 256,
            },
        )

        assert request.model == "deepseek-reasoner"
        assert request.enable_streaming is True
        assert request.top_p == 0.8
        assert request.tool_choice == "required"
        assert request.thinking_budget == 256
        assert request.messages[0]["content"] == "hello"
        assert request.messages[0]["reasoning_content"] == ""
        assert isinstance(request.messages[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_resolve_transport(self, monkeypatch):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        SiliconFlowAdapter._OPENAI_READY = True

        chat_request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            temperature=0.2,
            max_tokens=None,
            enable_streaming=False,
            kwargs={},
        )
        stream_request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            temperature=0.2,
            max_tokens=None,
            enable_streaming=True,
            kwargs={},
        )

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "auto")
        assert adapter._resolve_transport(chat_request) == "sdk"
        assert adapter._resolve_transport(stream_request) == "httpx"

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "httpx")
        assert adapter._resolve_transport(chat_request) == "httpx"
        assert adapter._resolve_transport(stream_request) == "httpx"

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "sdk")
        assert adapter._resolve_transport(chat_request) == "sdk"
        assert adapter._resolve_transport(stream_request) == "sdk"

    def test_encode_chat_request_for_httpx(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-chat", "tool_choice": "required"},
        )
        body = adapter._encode_chat_request_for_httpx(request)

        assert body == {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "stream": False,
            "max_tokens": 64,
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "tool_choice": "required",
        }

    def test_encode_stream_request(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            temperature=0.2,
            max_tokens=None,
            enable_streaming=True,
            kwargs={
                "model": "deepseek-chat",
                "enable_thinking": True,
                "thinking_budget": 256,
                "tool_choice": "required",
                "top_p": 0.8,
            },
        )
        kwargs = adapter._encode_stream_request(request)

        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "search"}}]
        assert kwargs["tool_choice"] == "required"
        assert kwargs["extra_body"] == {"thinking_budget": 256}
        assert kwargs["top_p"] == 0.8

    def test_encode_stream_request_for_httpx(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            temperature=0.2,
            max_tokens=None,
            enable_streaming=True,
            kwargs={
                "model": "deepseek-chat",
                "enable_thinking": False,
                "thinking_budget": None,
                "top_p": 0.8,
                "presence_penalty": None,
            },
        )
        payload = adapter._encode_stream_request_for_httpx(request)

        assert payload["top_p"] == 0.8
        assert "presence_penalty" not in payload

    def test_encode_chat_request(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={
                "model": "deepseek-chat",
                "enable_thinking": True,
                "thinking_budget": 128,
                "tool_choice": "required",
                "top_p": 0.8,
            },
        )
        kwargs = adapter._encode_chat_request(request)

        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 64
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "search"}}]
        assert kwargs["tool_choice"] == "required"
        assert kwargs["top_p"] == 0.8
        assert kwargs["extra_body"] == {"thinking_budget": 128}

    def test_extract_tool_calls(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        function = type("Func", (), {"name": "search", "arguments": '{"query":"hi"}'})()
        tool_delta = type("ToolDelta", (), {"index": 0, "id": "call_1", "function": function})()
        delta = type(
            "Delta",
            (),
            {"content": None, "reasoning_content": None, "tool_calls": [tool_delta]},
        )()
        choice = type("Choice", (), {"delta": delta, "finish_reason": "tool_calls"})()

        chunk, content_inc, reasoning_inc = adapter._build_stream_chunk(choice=choice)

        assert chunk is not None
        assert content_inc == 0
        assert reasoning_inc == 0
        assert chunk.tool_calls_delta == [
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "search", "arguments": '{"query":"hi"}'},
            }
        ]
        assert adapter.last_finish_reason == "tool_calls"

    def test_parse_httpx_sse_line(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        event, done = adapter._parse_httpx_sse_line("data: [DONE]")
        assert event is None
        assert done is True

        event, done = adapter._parse_httpx_sse_line("data: not-json")
        assert event is None
        assert done is False

    def test_build_httpx_stream_chunk(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        chunk, content_inc, reasoning_inc = adapter._build_httpx_stream_chunk(
            {
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                "choices": [{"delta": {"reasoning_content": "thinking"}}],
            }
        )

        assert chunk is not None
        assert chunk.content_delta == ""
        assert chunk.reasoning_delta == "thinking"
        assert content_inc == 0
        assert reasoning_inc == 1
        assert adapter.last_usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


class TestSiliconFlowAdapterHttpxChatRetry:
    @pytest.mark.asyncio
    async def test_chat_httpx_retries(self):
        class MockResponse:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {
                    "model": "deepseek-chat",
                    "choices": [
                        {"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }

        attempts = 0

        class ConnectBoom(Exception):
            pass

        class MockHttpxClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectBoom("connect failed")
                return MockResponse()

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch("httpx.TransportError", ConnectBoom),
                patch("houyi.adapters.llm.siliconflow_adapter.asyncio.sleep", new=AsyncMock()),
                patch("houyi.infrastructure.net.proxy.detect_proxy", return_value=None),
            ):
                result = await adapter.chat(
                    [{"role": "user", "content": "hi"}], model="deepseek-chat"
                )

        assert attempts == 2
        assert result.content == "ok"
        assert adapter.last_usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

    @pytest.mark.asyncio
    async def test_chat_httpx_retry_exhausts(self):
        class MockResponse:
            status_code = 500
            text = "server error"
            headers = {}

            @staticmethod
            def json():
                return {}

        attempts = 0

        class MockHttpxClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                nonlocal attempts
                attempts += 1
                return MockResponse()

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch("houyi.adapters.llm.siliconflow_adapter.asyncio.sleep", new=AsyncMock()),
                patch("houyi.infrastructure.net.proxy.detect_proxy", return_value=None),
                pytest.raises(
                    RuntimeError,
                    match=re.escape(
                        "SiliconFlow is temporarily unavailable. Please retry in a moment."
                    ),
                ),
            ):
                await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert attempts == 4


class TestSiliconFlowAdapterChatPath:
    @pytest.mark.asyncio
    async def test_chat_returns_mock(self):
        with patch.dict(os.environ, {}, clear=True):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

        result = await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert result.content == "Mock response (no API key)"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_uses_client(self):
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = True
            adapter = SiliconFlowAdapter()

        expected = MagicMock(content="client")
        with patch.object(adapter, "_chat_request", AsyncMock(return_value=expected)) as mocked:
            result = await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert result is expected
        mocked.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_updates_usage(self):
        tool_call = MagicMock()
        tool_call.model_dump.return_value = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"},
        }
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "ok"
        response.choices[0].message.tool_calls = [tool_call]
        response.choices[0].finish_reason = "tool_calls"
        response.usage.prompt_tokens = 2
        response.usage.completion_tokens = 3
        response.usage.total_tokens = 5
        response.model = "deepseek-chat"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        mock_client.close = AsyncMock()

        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = True
            adapter = SiliconFlowAdapter()

            with patch.dict(sys.modules, {"openai": fake_openai}):
                request = adapter._build_request(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[{"type": "function", "function": {"name": "search"}}],
                    temperature=0.2,
                    max_tokens=64,
                    enable_streaming=False,
                    kwargs={"model": "deepseek-chat", "tool_choice": "required", "top_p": 0.8},
                )
                result = await adapter._chat_request(request)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == [{"type": "function", "function": {"name": "search"}}]
        assert call_kwargs["tool_choice"] == "required"
        assert call_kwargs["top_p"] == 0.8
        assert result.content == "ok"
        assert result.tool_calls[0]["function"]["name"] == "search"
        assert adapter.last_usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        mock_client.close.assert_awaited_once()


class TestSiliconFlowAdapterHttpxStreamHelpers:
    @pytest.mark.asyncio
    async def test_stream_error_retries(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        response = MagicMock(status_code=500, headers={})
        response.aread = AsyncMock(return_value=b"server error")
        retry_controller = MagicMock()
        retry_controller.policy.status_forcelist = {500}
        retry_controller.retries_used = 1
        retry_controller.policy.total_retries = 3
        retry_controller.on_status_code.return_value = type(
            "Decision", (), {"retry": True, "bucket": "status", "delay_seconds": 0.0}
        )()

        with patch("houyi.adapters.llm.siliconflow_adapter.asyncio.sleep", new=AsyncMock()):
            assert (
                await adapter._stream_status(
                    response=response,
                    retry_controller=retry_controller,
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_stream_error_raises(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        response = MagicMock(status_code=400, headers={})
        response.aread = AsyncMock(return_value=b"bad request")
        response.raise_for_status.side_effect = RuntimeError("400 bad request")
        retry_controller = MagicMock()
        retry_controller.policy.status_forcelist = {500}

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "SiliconFlow rejected the request as invalid. Please retry or adjust the request payload."
            ),
        ):
            await adapter._stream_status(
                response=response,
                retry_controller=retry_controller,
            )

    @pytest.mark.asyncio
    async def test_stream_httpx_retry(self):
        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: [DONE]",
        ]
        attempts = 0

        class ConnectBoom(Exception):
            pass

        class MockResponse:
            status_code = 200

            async def aread(self):
                return b""

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class MockHttpxClient:
            def stream(self, method, url, json=None, headers=None):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectBoom("connect fail")
                return MockResponse()

            async def aclose(self):
                return None

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

        with (
            patch("httpx.AsyncClient", return_value=MockHttpxClient()),
            patch("httpx.TransportError", ConnectBoom),
            patch("houyi.adapters.llm.siliconflow_adapter.asyncio.sleep", new=AsyncMock()),
            patch("houyi.infrastructure.net.proxy.detect_proxy", return_value=None),
        ):
            chunks = []
            request = adapter._build_request(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                temperature=0.7,
                max_tokens=None,
                enable_streaming=True,
                kwargs={"model": "deepseek-chat"},
            )
            async for chunk in adapter._stream_request_httpx(request):
                chunks.append(chunk.content_delta)

        assert attempts == 2
        assert chunks == ["OK"]

    def test_parse_httpx_sse_non_dict(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        event, done = adapter._parse_httpx_sse_line("data: [1, 2]")
        assert event is None
        assert done is False

    def test_build_httpx_stream_empty(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        chunk, content_inc, reasoning_inc = adapter._build_httpx_stream_chunk(
            {"choices": [{"delta": {"content": "", "reasoning_content": ""}}]}
        )

        assert chunk is None
        assert content_inc == 0
        assert reasoning_inc == 0


class TestSiliconFlowAdapterAdditionalCoverage:
    def test_parse_httpx_chat_response_raises(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        response = type(
            "Resp", (), {"status_code": 500, "text": "boom", "json": staticmethod(lambda: {})}
        )()

        with pytest.raises(
            RuntimeError,
            match=re.escape("SiliconFlow is temporarily unavailable. Please retry in a moment."),
        ):
            adapter._parse_httpx_chat_response(response)

    def test_build_stream_chunk_returns_content(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        delta = type(
            "Delta", (), {"content": "hello", "reasoning_content": None, "tool_calls": None}
        )()
        choice = type("Choice", (), {"delta": delta, "finish_reason": "stop"})()

        chunk, content_inc, reasoning_inc = adapter._build_stream_chunk(choice=choice)

        assert chunk is not None
        assert chunk.content_delta == "hello"
        assert chunk.reasoning_delta is None
        assert content_inc == 1
        assert reasoning_inc == 0
        assert adapter.last_finish_reason == "stop"

    def test_build_stream_chunk_returns_none_when_empty(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        delta = type("Delta", (), {"content": "", "reasoning_content": "", "tool_calls": None})()
        choice = type("Choice", (), {"delta": delta, "finish_reason": None})()

        chunk, content_inc, reasoning_inc = adapter._build_stream_chunk(choice=choice)

        assert chunk is None
        assert content_inc == 0
        assert reasoning_inc == 0

    def test_build_httpx_stream_chunk_uses_message_fallbacks(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        chunk, content_inc, reasoning_inc = adapter._build_httpx_stream_chunk(
            {
                "choices": [
                    {
                        "message": {"content": "fallback", "reasoning_content": "reason"},
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

        assert chunk is not None
        assert chunk.content_delta == "fallback"
        assert chunk.reasoning_delta == "reason"
        assert content_inc == 1
        assert reasoning_inc == 1

    def test_stream_fills_tool_call(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        request = adapter._build_request(
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Find README"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "houyi_grep", "arguments": {"query": "README"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"matches":["README.md"]}',
                    "tool_call_id": "call_1",
                    "name": "houyi_grep",
                },
            ],
            tools=None,
            temperature=0.2,
            max_tokens=None,
            enable_streaming=True,
            kwargs={"model": "deepseek-chat"},
        )

        payload = adapter._encode_stream_request_for_httpx(request)

        assert payload["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert payload["messages"][1] == {"role": "user", "content": "Find README"}
        assert payload["messages"][2]["role"] == "assistant"
        assert payload["messages"][2]["content"] == "[tool call]"
        assert payload["messages"][2]["tool_calls"][0]["function"]["name"] == "houyi_grep"
        assert payload["messages"][3]["role"] == "tool"
        assert payload["messages"][3]["tool_call_id"] == "call_1"

    def test_chat_keeps_tool_calls(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        request = adapter._build_request(
            messages=[
                {"role": "user", "content": "Find README"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "houyi_grep", "arguments": {"query": "README"}},
                        }
                    ],
                },
            ],
            tools=[{"type": "function", "function": {"name": "houyi_grep"}}],
            temperature=0.2,
            max_tokens=None,
            enable_streaming=False,
            kwargs={"model": "deepseek-chat"},
        )

        payload = adapter._encode_chat_request_for_httpx(request)

        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][1]["content"] == ""
        assert payload["messages"][1]["tool_calls"][0]["function"]["name"] == "houyi_grep"

    @pytest.mark.asyncio
    async def test_chat_transport_false(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        retry_controller = MagicMock()
        retry_controller.on_transport_exception.return_value = type(
            "Decision", (), {"retry": False, "bucket": "connect", "delay_seconds": 0.0}
        )()

        should_retry, delay = adapter._chat_transport(
            exc=RuntimeError("boom"),
            retry_controller=retry_controller,
        )

        assert should_retry is False
        assert delay == 0.0

    @pytest.mark.asyncio
    async def test_chat_status_none(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        retry_controller = MagicMock()
        retry_controller.policy.status_forcelist = {500}
        retry_controller.on_status_code.return_value = type(
            "Decision", (), {"retry": False, "bucket": "status", "delay_seconds": 0.0}
        )()
        response = type("Resp", (), {"status_code": 500, "headers": {}, "text": "server error"})()

        should_retry, error = await adapter._chat_status(
            response=response,
            retry_controller=retry_controller,
        )

        assert should_retry is False
        assert error is None
