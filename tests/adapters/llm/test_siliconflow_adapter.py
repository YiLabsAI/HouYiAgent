"""Covers SiliconFlow adapter mock fallback, client/httpx paths, retries, and stream parsing."""

from __future__ import annotations

import json
import os
import re
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.adapters.llm import siliconflow_adapter as siliconflow_module
from houyi.adapters.llm.request_models import OpenAICompatRequest
from houyi.adapters.llm.siliconflow_adapter import (
    SiliconFlowAdapter,
    _format_siliconflow_http_error,
)
from houyi.infrastructure.config.env_config import EnvConfig
from houyi.infrastructure.net import proxy as proxy_module


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
    async def test_stream_prefers_httpx(self):
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

    def test_prepare_v32_replay(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3.2",
        )
        request = OpenAICompatRequest(
            model="deepseek-ai/DeepSeek-V3.2",
            messages=[
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "houyi_find_files",
                                "arguments": {"pattern": "*.md"},
                            },
                        }
                    ],
                    "content": None,
                    "reasoning_content": "hidden",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "houyi_find_files",
                    "content": {"matches": ["README.md"]},
                    "metadata": {"ignored": True},
                },
            ],
            temperature=0.2,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_find_files",
                        "description": "find files",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            extra_kwargs={"parallel_tool_calls": False, "prompt_cache_key": "abc"},
            tool_choice="required",
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.messages == [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": 'Tool calls requested:\n1. houyi_find_files {"pattern": "*.md"}',
            },
            {
                "role": "user",
                "content": 'Tool result for houyi_find_files (call_1):\n{"matches": ["README.md"]}',
            },
        ]
        assert prepared.extra_kwargs == {}
        assert prepared.tool_choice is None

    def test_prepare_v3_tool_turn(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3",
        )
        request = OpenAICompatRequest(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "find search skills"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "houyi_find_files",
                                "arguments": {"pattern": "*search*"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "houyi_find_files",
                    "content": {"matches": ["houyi/skills/web_search/SKILL.md"]},
                },
            ],
            temperature=0.2,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_find_files",
                        "description": "find files",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            extra_kwargs={"parallel_tool_calls": True},
            tool_choice="required",
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.messages == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "find search skills"},
            {
                "role": "assistant",
                "content": 'Tool calls requested:\n1. houyi_find_files {"pattern": "*search*"}',
            },
            {
                "role": "user",
                "content": 'Tool result for houyi_find_files (call_1):\n{"matches": ["houyi/skills/web_search/SKILL.md"]}',
            },
        ]
        assert prepared.extra_kwargs == {}
        assert prepared.tool_choice is None

    def test_prepare_r1_tool_result(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )
        request = OpenAICompatRequest(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "houyi_find_files", "arguments": {"pattern": "*"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "houyi_find_files",
                    "metadata": {"ignored": True},
                    "content": json.dumps(
                        {
                            "data": {"matches": ["/tmp/a", "/tmp/b"], "root_path": "/tmp"},
                            "metadata": {"latency_ms": 10},
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.2,
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.messages == [
            {
                "role": "assistant",
                "content": 'Tool calls requested:\n1. houyi_find_files {"pattern": "*"}',
            },
            {
                "role": "user",
                "content": 'Tool result for houyi_find_files (call_1):\n{"matches": ["/tmp/a", "/tmp/b"], "root_path": "/tmp"}',
            },
        ]

    def test_prepare_r1_empty_tool_message(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )
        request = OpenAICompatRequest(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "houyi_list_dir", "arguments": {}},
                        }
                    ],
                }
            ],
            temperature=0.2,
            tool_choice="required",
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.messages == [
            {
                "role": "assistant",
                "content": "Tool calls requested:\n1. houyi_list_dir {}",
            }
        ]
        assert prepared.tool_choice is None


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
    """Test chat payload sanitation for strict OpenAI-compatible providers."""

    @pytest.mark.asyncio
    async def test_chat_sanitizes_messages(self):
        captured: dict[str, object] = {}

        sse_body = (
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n'
            "data: [DONE]\n"
        )

        class MockStreamResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                for line in sse_body.strip().splitlines():
                    yield line

        class MockHttpxClient:
            async def aclose(self):
                pass

            def stream(self, method, url, json=None, headers=None):
                captured["url"] = url
                captured["body"] = json
                captured["headers"] = headers

                class _Ctx:
                    async def __aenter__(inner_self):
                        return MockStreamResponse()

                    async def __aexit__(inner_self, *args):
                        return False

                return _Ctx()

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
        assert "reasoning_content" not in payload_messages[1]

    def test_request_deepseek_tools(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )

        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_web_search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                        "strict": True,
                    },
                    "server_only": True,
                }
            ],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-R1"},
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.tools == [
            {
                "type": "function",
                "function": {
                    "name": "houyi_web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]

    def test_prepare_r1_multi_tool_turn(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )

        request = OpenAICompatRequest(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "I will check two sources in parallel first",
                    "reasoning_content": "Think first, then call tools",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "houyi_find_files",
                                "arguments": {"pattern": "*.md"},
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": {"query": "infoq"}},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"matches": ["README.md"]}'},
                {"role": "tool", "tool_call_id": "call_2", "content": '{"results": ["infoq"]}'},
            ],
            temperature=0.2,
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.messages == [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": (
                    "Tool calls requested:\n"
                    '1. houyi_find_files {"pattern": "*.md"}\n'
                    '2. web_search {"query": "infoq"}'
                ),
            },
            {
                "role": "user",
                "content": 'Tool result for call_1:\n{"matches": ["README.md"]}',
            },
            {
                "role": "user",
                "content": 'Tool result for call_2:\n{"results": ["infoq"]}',
            },
        ]

    def test_prepare_r1_reasoning_turn(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )

        request = OpenAICompatRequest(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "Let me analyze this first",
                    "reasoning_content": "This is the full reasoning flow",
                },
            ],
            temperature=0.2,
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.messages == [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Let me analyze this first",
            },
        ]

    def test_request_v32_tools(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3.2",
        )

        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "demo",
                        "description": "Demo",
                        "parameters": {},
                        "strict": True,
                    },
                    "x-provider": "ignored",
                }
            ],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-V3.2"},
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.tools == [
            {
                "type": "function",
                "function": {
                    "name": "demo",
                    "description": "Demo",
                    "parameters": {},
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_preserves_extra_body(self):
        captured: dict[str, object] = {}

        sse_body = (
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n'
            "data: [DONE]\n"
        )

        class MockStreamResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                for line in sse_body.strip().splitlines():
                    yield line

        class MockHttpxClient:
            async def aclose(self):
                pass

            def stream(self, method, url, json=None, headers=None):
                captured["url"] = url
                captured["body"] = json
                captured["headers"] = headers

                class _Ctx:
                    async def __aenter__(inner_self):
                        return MockStreamResponse()

                    async def __aexit__(inner_self, *args):
                        return False

                return _Ctx()

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

        sse_body = (
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n'
            "data: [DONE]\n"
        )

        class MockStreamResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                for line in sse_body.strip().splitlines():
                    yield line

        class MockHttpxClient:
            async def aclose(self):
                pass

            def stream(self, method, url, json=None, headers=None):
                captured["body"] = json

                class _Ctx:
                    async def __aenter__(inner_self):
                        return MockStreamResponse()

                    async def __aexit__(inner_self, *args):
                        return False

                return _Ctx()

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
        assert "reasoning_content" not in payload_messages[1]


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
        assert "reasoning_content" not in request.messages[0]
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
        assert adapter._resolve_transport(chat_request) == "client"
        assert adapter._resolve_transport(stream_request) == "httpx"

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "httpx")
        assert adapter._resolve_transport(chat_request) == "httpx"
        assert adapter._resolve_transport(stream_request) == "httpx"

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "client")
        assert adapter._resolve_transport(chat_request) == "client"
        assert adapter._resolve_transport(stream_request) == "client"

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "sdk")
        assert adapter._resolve_transport(chat_request) == "client"
        assert adapter._resolve_transport(stream_request) == "client"

    def test_resolve_transport_tool_model(self, monkeypatch):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )
        SiliconFlowAdapter._OPENAI_READY = True

        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_grep",
                        "description": "grep",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-R1"},
        )

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "auto")
        assert adapter._resolve_transport(request) == "client"

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "sdk")
        assert adapter._resolve_transport(request) == "client"

        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "client")
        assert adapter._resolve_transport(request) == "client"

        SiliconFlowAdapter._OPENAI_READY = False
        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "auto")
        assert adapter._resolve_transport(request) == "httpx"

    def test_prepare_request_logs(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )
        request = adapter._build_request(
            messages=[
                {"role": "assistant", "content": "", "tool_calls": []},
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "houyi_read_file",
                    "metadata": {"round": 1},
                    "content": '{"ok":true}',
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_read_file",
                        "description": "read file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={
                "model": "deepseek-ai/DeepSeek-R1",
                "tool_choice": "required",
                "parallel_tool_calls": True,
            },
        )

        with patch.object(siliconflow_module.logger, "debug") as mock_debug:
            prepared = adapter._prepare_request_for_provider(request)

        assert prepared.tool_choice is None
        assert "parallel_tool_calls" not in prepared.extra_kwargs
        mock_debug.assert_called()
        assert "SiliconFlow DeepSeek prepared payload summary" in mock_debug.call_args.args[0]

    def test_prepare_request_drops_required_for_kimi(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="moonshotai/Kimi-K2.5",
        )
        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_web_search",
                        "description": "search",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={
                "model": "moonshotai/Kimi-K2.5",
                "tool_choice": "required",
            },
        )

        prepared = adapter._prepare_request_for_provider(request)

        assert prepared.tool_choice is None

    @pytest.mark.asyncio
    async def test_sdk_create_logs_deepseek_final_kwargs_summary(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )
        request = adapter._build_request(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "houyi_read_file", "arguments": {"path": "a"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "houyi_read_file",
                    "metadata": {"round": 1},
                    "content": '{"ok":true}',
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_read_file",
                        "description": "read file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={
                "model": "deepseek-ai/DeepSeek-R1",
                "tool_choice": "required",
                "parallel_tool_calls": True,
            },
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=object())

        with patch.object(siliconflow_module.logger, "debug") as mock_debug:
            await adapter._create_chat_response(request=request, client=mock_client)

        sent_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in sent_kwargs
        assert "parallel_tool_calls" not in sent_kwargs
        assert sent_kwargs["messages"][0] == {
            "role": "assistant",
            "content": 'Tool calls requested:\n1. houyi_read_file {"path": "a"}',
        }
        assert sent_kwargs["messages"][1] == {
            "role": "user",
            "content": 'Tool result for houyi_read_file (call_1):\n{"ok": true}',
        }
        mock_debug.assert_called()
        assert "SiliconFlow DeepSeek client create kwargs summary" in mock_debug.call_args.args[0]

    def test_encode_request_for_httpx(self):
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

    def test_encode_request_for_deepseek(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )

        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "houyi_web_search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                        "strict": True,
                    },
                    "x-provider": "ignored",
                }
            ],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={
                "model": "deepseek-ai/DeepSeek-R1",
                "transport": "httpx",
                "parallel_tool_calls": False,
            },
        )

        body = adapter._encode_chat_request_for_httpx(request)

        assert body["model"] == "deepseek-ai/DeepSeek-R1"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "houyi_web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body

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
    def test_chat_timeout(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        timeout = adapter._get_httpx_chat_timeout()

        assert timeout.connect == 10.0
        assert timeout.read == 30.0

    def test_retry_policy_scopes(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        chat_retry = adapter._get_httpx_retry_controller().policy
        stream_retry = adapter._stream_retry().policy

        assert chat_retry.total_retries == 1
        assert chat_retry.status_retries == 1
        assert stream_retry.total_retries == 3
        assert stream_retry.status_retries == 3

    def test_chat_proxy_disabled(self, monkeypatch):
        monkeypatch.delenv("HOUYI_PROXY_ENABLED", raising=False)
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        with patch.object(proxy_module, "detect_proxy") as mock_detect:
            assert adapter._get_httpx_proxy() is None

        mock_detect.assert_not_called()

    def test_chat_proxy_enabled(self, monkeypatch):
        monkeypatch.setenv("HOUYI_PROXY_ENABLED", "true")
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        with patch.object(proxy_module, "detect_proxy", return_value="http://proxy:7890"):
            assert adapter._get_httpx_proxy() == "http://proxy:7890"

    def test_client_proxy_disabled(self, monkeypatch):
        monkeypatch.delenv("HOUYI_PROXY_ENABLED", raising=False)
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        fake_client = MagicMock()
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock(return_value=fake_client)

        with (
            patch.dict(sys.modules, {"openai": fake_openai}),
            patch("httpx.AsyncClient", return_value=MagicMock()) as mock_httpx_client,
            patch.object(proxy_module, "detect_proxy") as mock_detect,
        ):
            created = adapter._new_client()

        assert created is fake_client
        mock_detect.assert_not_called()
        async_client_kwargs = mock_httpx_client.call_args.kwargs
        assert async_client_kwargs["proxy"] is None
        assert async_client_kwargs["trust_env"] is False
        assert (
            fake_openai.AsyncOpenAI.call_args.kwargs["http_client"]
            is mock_httpx_client.return_value
        )

    def test_client_proxy_enabled(self, monkeypatch):
        monkeypatch.setenv("HOUYI_PROXY_ENABLED", "true")
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )
        fake_client = MagicMock()
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock(return_value=fake_client)

        with (
            patch.dict(sys.modules, {"openai": fake_openai}),
            patch("httpx.AsyncClient", return_value=MagicMock()) as mock_httpx_client,
            patch.object(proxy_module, "detect_proxy", return_value="http://proxy:7890"),
        ):
            created = adapter._new_client()

        assert created is fake_client
        async_client_kwargs = mock_httpx_client.call_args.kwargs
        assert async_client_kwargs["proxy"] == "http://proxy:7890"
        assert async_client_kwargs["trust_env"] is False
        assert (
            fake_openai.AsyncOpenAI.call_args.kwargs["http_client"]
            is mock_httpx_client.return_value
        )

    @pytest.mark.asyncio
    async def test_chat_httpx_retries(self):
        attempts = 0

        class ConnectBoom(Exception):
            pass

        sse_body = (
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n'
            "data: [DONE]\n"
        )

        class MockStreamResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                for line in sse_body.strip().splitlines():
                    yield line

        class MockHttpxClient:
            async def aclose(self):
                pass

            def stream(self, method, url, json=None, headers=None):
                nonlocal attempts
                attempts += 1

                class _Ctx:
                    async def __aenter__(inner_self):
                        if attempts == 1:
                            raise ConnectBoom("connect failed")
                        return MockStreamResponse()

                    async def __aexit__(inner_self, *args):
                        return False

                return _Ctx()

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch("httpx.TransportError", ConnectBoom),
                patch.object(siliconflow_module.asyncio, "sleep", new=AsyncMock()),
                patch.object(proxy_module, "detect_proxy", return_value=None),
            ):
                result = await adapter.chat(
                    [{"role": "user", "content": "hi"}], model="deepseek-chat"
                )

        assert attempts == 2
        assert result.content == "ok"
        assert adapter.last_usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

    @pytest.mark.asyncio
    async def test_chat_httpx_retry_exhausts(self):
        attempts = 0

        class MockStreamResponse:
            status_code = 500
            text = "server error"
            headers: dict = {}

            def raise_for_status(self):
                import httpx

                raise httpx.HTTPStatusError(
                    "500", request=httpx.Request("POST", "http://x"), response=self
                )

            async def aread(self):
                return b"server error"

            async def aiter_lines(self):
                return
                yield  # pragma: no cover

        class MockHttpxClient:
            async def aclose(self):
                pass

            def stream(self, method, url, json=None, headers=None):
                nonlocal attempts
                attempts += 1

                class _Ctx:
                    async def __aenter__(inner_self):
                        return MockStreamResponse()

                    async def __aexit__(inner_self, *args):
                        return False

                return _Ctx()

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch.object(siliconflow_module.asyncio, "sleep", new=AsyncMock()),
                patch.object(proxy_module, "detect_proxy", return_value=None),
                pytest.raises(RuntimeError),
            ):
                await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert attempts >= 2


class TestSiliconFlowAdapterChatPath:
    @pytest.mark.asyncio
    async def test_chat_returns_mock(self):
        with patch.dict(os.environ, {}, clear=True):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = False
            adapter = SiliconFlowAdapter()

        result = await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert "Mock response from" in result.content
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_uses_client(self):
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._OPENAI_READY = True
            adapter = SiliconFlowAdapter()

        from houyi.adapters.llm.base import StreamChunk

        async def _fake_stream_chat(*args, **kwargs):
            yield StreamChunk(content_delta="client")

        with patch.object(adapter, "stream_chat", _fake_stream_chat):
            result = await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert result.content == "client"

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

        with patch.object(siliconflow_module.asyncio, "sleep", new=AsyncMock()):
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
            patch.object(siliconflow_module.asyncio, "sleep", new=AsyncMock()),
            patch.object(proxy_module, "detect_proxy", return_value=None),
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
            adapter._parse_httpx_response(response)

    @pytest.mark.asyncio
    async def test_chat_request_httpx_parses_token_wrapped_toolcall(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3.2",
        )
        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "functions.houyi_list_dir"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-V3.2", "transport": "httpx"},
        )
        response = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "text": "",
                "json": staticmethod(
                    lambda: {
                        "model": "deepseek-ai/DeepSeek-V3.2",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": (
                                        "<|tool_calls_section_begin|><|tool_call_begin|>functions.houyi_list_dir:20"
                                        '<|tool_call_argument_begin|>{"path": ".houyi/skills", "max_results": 50}'
                                        "<|tool_call_end|><|tool_calls_section_end|>"
                                    )
                                },
                            }
                        ],
                    }
                ),
            },
        )()

        with patch.object(adapter, "_execute_chat_httpx", AsyncMock(return_value=response)):
            result = await adapter._chat_request_httpx(request)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "functions.houyi_list_dir"
        assert result.tool_calls[0]["function"]["arguments"] == {
            "path": ".houyi/skills",
            "max_results": 50,
        }

    @pytest.mark.asyncio
    async def test_chat_request_httpx_parses_multiple_xml_toolcalls(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3.2",
        )
        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "houyi_read_file"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-V3.2", "transport": "httpx"},
        )
        response = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "text": "",
                "json": staticmethod(
                    lambda: {
                        "model": "deepseek-ai/DeepSeek-V3.2",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": (
                                        "houyi_read_file"
                                        "<arg_key>path</arg_key><arg_value>./houyi/domain/skill/schema.py</arg_value>"
                                        "houyi_read_file"
                                        "<arg_key>path</arg_key><arg_value>./houyi/domain/skill/spec.py</arg_value>"
                                    )
                                },
                            }
                        ],
                    }
                ),
            },
        )()

        with patch.object(adapter, "_execute_chat_httpx", AsyncMock(return_value=response)):
            result = await adapter._chat_request_httpx(request)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["function"]["arguments"] == {
            "path": "./houyi/domain/skill/schema.py"
        }
        assert result.tool_calls[1]["function"]["arguments"] == {
            "path": "./houyi/domain/skill/spec.py"
        }

    @pytest.mark.asyncio
    async def test_chat_request_httpx_parses_bracket_toolcall(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3.2",
        )
        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "houyi_shell_exec"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-V3.2", "transport": "httpx"},
        )
        response = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "text": "",
                "json": staticmethod(
                    lambda: {
                        "model": "deepseek-ai/DeepSeek-V3.2",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": (
                                        "[tool:houyi_shell_exec] "
                                        '{"command": "find /Users/von/workspace/HouYiAgent -name \\"*.md\\"", '
                                        '"cwd": "/Users/von/workspace/HouYiAgent", "timeout_seconds": 30}'
                                    )
                                },
                            }
                        ],
                    }
                ),
            },
        )()

        with patch.object(adapter, "_execute_chat_httpx", AsyncMock(return_value=response)):
            result = await adapter._chat_request_httpx(request)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "houyi_shell_exec"
        assert result.tool_calls[0]["function"]["arguments"] == {
            "command": 'find /Users/von/workspace/HouYiAgent -name "*.md"',
            "cwd": "/Users/von/workspace/HouYiAgent",
            "timeout_seconds": 30,
        }

    @pytest.mark.asyncio
    async def test_chat_request_httpx_parses_bare_bracket_toolcall(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3.2",
        )
        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "houyi_shell_exec"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-V3.2", "transport": "httpx"},
        )
        response = type(
            "Resp",
            (),
            {"status_code": 200, "text": "", "json": staticmethod(lambda: {})},
        )()

        with patch.object(adapter, "_execute_chat_httpx", AsyncMock(return_value=response)):
            result = await adapter._chat_request_httpx(request)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "houyi_shell_exec"
        assert result.tool_calls[0]["function"]["arguments"] == {}

    @pytest.mark.asyncio
    async def test_chat_request_httpx_ignores_bracket_tool_result_envelope(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3.2",
        )
        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "houyi_shell_exec"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-V3.2", "transport": "httpx"},
        )
        response = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "text": "",
                "json": staticmethod(
                    lambda: {
                        "model": "deepseek-ai/DeepSeek-V3.2",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": (
                                        "[tool:houyi_shell_exec] "
                                        '{"data": {"command": "find /Users/von/workspace/HouYiAgent -name \\"readme.md\\""}, '
                                        '"message": "", "success": true}'
                                    )
                                },
                            }
                        ],
                    }
                ),
            },
        )()

        with patch.object(adapter, "_execute_chat_httpx", AsyncMock(return_value=response)):
            result = await adapter._chat_request_httpx(request)

        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_chat_request_for_v32(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-V3.2",
        )
        request = adapter._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            temperature=0.2,
            max_tokens=64,
            enable_streaming=False,
            kwargs={"model": "deepseek-ai/DeepSeek-V3.2", "transport": "httpx"},
        )
        response = type(
            "Resp",
            (),
            {"status_code": 400, "text": "bad request", "json": staticmethod(lambda: {})},
        )()

        with (
            patch.object(adapter, "_execute_chat_httpx", AsyncMock(return_value=response)),
            patch.object(siliconflow_module.logger, "error") as mock_error,
        ):
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    "SiliconFlow rejected the request as invalid. Please retry or adjust the request payload."
                ),
            ):
                await adapter._chat_request_httpx(request)

        mock_error.assert_called_once()
        assert "SiliconFlow DeepSeek httpx 400 payload summary" in mock_error.call_args.args[0]

    def test_stream_chunk_returns_content(self):
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

    def test_stream_chunk_returns_none(self):
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

    def test_httpx_stream_uses_message_fallbacks(self):
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

    def test_stream_fills_toolcall(self):
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
        assert payload["messages"][2]["content"] == ""
        assert payload["messages"][2]["tool_calls"][0]["function"]["name"] == "houyi_grep"
        assert payload["messages"][3]["role"] == "tool"
        assert payload["messages"][3]["tool_call_id"] == "call_1"

    def test_chat_keeps_toolcalls(self):
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

        should_retry, delay = adapter._handle_httpx_transport_error(
            exc=RuntimeError("boom"),
            retry_controller=retry_controller,
        )

        assert should_retry is False
        assert delay == 0.0

    def test_chat_transport_logs_context(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-ai/DeepSeek-R1",
        )
        retry_controller = MagicMock()
        retry_controller.retries_used = 1
        retry_controller.policy.total_retries = 3
        retry_controller.on_transport_exception.return_value = type(
            "Decision", (), {"retry": True, "bucket": "read", "delay_seconds": 0.5}
        )()
        request = OpenAICompatRequest(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": "find skill.md"}],
            tools=[{"type": "function", "function": {"name": "houyi_find_files"}}],
            temperature=0.2,
            max_tokens=64,
        )
        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [{"role": "user", "content": "find skill.md"}],
            "tools": [{"type": "function", "function": {"name": "houyi_find_files"}}],
        }

        with patch.object(siliconflow_module.logger, "warning") as mock_warning:
            should_retry, delay = adapter._handle_httpx_transport_error(
                exc=RuntimeError("Server disconnected without sending a response."),
                retry_controller=retry_controller,
                request=request,
                payload=payload,
            )

        assert should_retry is True
        assert delay == 0.5
        assert mock_warning.called
        assert "SiliconFlow chat transport error" in mock_warning.call_args.args[0]
        assert mock_warning.call_args.args[1] == "deepseek-ai/DeepSeek-R1"
        assert mock_warning.call_args.args[6] > 0
        assert mock_warning.call_args.args[7][0]["role"] == "user"
        assert mock_warning.call_args.args[8] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_httpx_status_none(self):
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

        should_retry, error = await adapter._handle_httpx_status(
            response=response,
            retry_controller=retry_controller,
        )

        assert should_retry is False
        assert error is None
