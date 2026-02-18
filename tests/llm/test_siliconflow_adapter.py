"""Unit tests for houyi.llm.siliconflow_adapter — SiliconFlowAdapter.

Tests cover:
- Mock mode (no API key)
- SDK streaming path (mocked openai client)
- httpx fallback streaming path
- Reasoning (enable_reasoning + thinking_budget)
- httpx edge cases (empty lines, bad JSON, missing delta)
"""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.config.env_config import EnvConfig
from houyi.llm.siliconflow_adapter import SiliconFlowAdapter


@pytest.fixture(autouse=True)
def _reset_env_config():
    """Reset EnvConfig singleton before/after each test so env patches take effect."""
    EnvConfig._reset()
    yield
    EnvConfig._reset()


class TestSiliconFlowAdapterMockMode:
    """Test SiliconFlowAdapter when no API key is set (mock mode)."""

    @pytest.mark.asyncio
    async def test_mock_streaming_no_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                SiliconFlowAdapter._SDK_AVAILABLE = None
                adapter = SiliconFlowAdapter()
                assert adapter.api_key is None

                messages = [{"role": "user", "content": "Hello world"}]
                chunks = []
                async for content, reasoning in adapter.stream_chat(messages):
                    chunks.append((content, reasoning))

                assert len(chunks) > 0
                full_content = "".join(c for c, _ in chunks)
                assert "Mock response" in full_content
                assert all(r is None for _, r in chunks)

    @pytest.mark.asyncio
    async def test_stream_completion_delegates_to_stream_chat(self):
        """stream_completion should delegate to stream_chat."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                SiliconFlowAdapter._SDK_AVAILABLE = None
                adapter = SiliconFlowAdapter()

                chunks = []
                async for chunk in adapter.stream_completion("Test prompt"):
                    chunks.append(chunk)

                assert len(chunks) > 0
                full = "".join(c for c, _ in chunks)
                assert "Mock response" in full

    @pytest.mark.asyncio
    async def test_mock_extracts_last_user_content(self):
        """Mock mode should use last user message content."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                SiliconFlowAdapter._SDK_AVAILABLE = None
                adapter = SiliconFlowAdapter()

                messages = [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Tell me about Python"},
                ]
                chunks = []
                async for content, _ in adapter.stream_chat(messages):
                    chunks.append(content)

                full = "".join(chunks)
                assert "Tell me about Python" in full


class TestSiliconFlowAdapterSDKPath:
    """Test SiliconFlowAdapter SDK path with mocked openai client."""

    @pytest.mark.asyncio
    async def test_sdk_stream_chat(self):
        """Mock the openai SDK path to verify stream_chat behavior."""

        class MockDelta:
            def __init__(self, content=None, reasoning_content=None):
                self.content = content
                self.reasoning_content = reasoning_content
                self.tool_calls = None

        class MockChoice:
            def __init__(self, delta, finish_reason=None):
                self.delta = delta
                self.finish_reason = finish_reason

        class MockChunk:
            def __init__(self, choices=None, usage=None):
                self.choices = choices or []
                self.usage = usage

        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(MockDelta(content="Hello"))])
            yield MockChunk(choices=[MockChoice(MockDelta(content=" world"))])
            yield MockChunk(choices=[MockChoice(MockDelta(reasoning_content="thinking"))])
            yield MockChunk(choices=[], usage=MockUsage())

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_client.close = AsyncMock()

        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._SDK_AVAILABLE = True
            adapter = SiliconFlowAdapter()

            with patch.dict(sys.modules, {"openai": fake_openai}):
                chunks = []
                async for content, reasoning in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}]
                ):
                    chunks.append((content, reasoning))

        assert len(chunks) == 3
        assert chunks[0] == ("Hello", None)
        assert chunks[1] == (" world", None)
        assert chunks[2] == ("", "thinking")
        assert adapter.last_usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


class TestSiliconFlowAdapterHttpxPath:
    """Test SiliconFlowAdapter httpx fallback path."""

    @pytest.mark.asyncio
    async def test_httpx_stream_chat(self):
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
            SiliconFlowAdapter._SDK_AVAILABLE = False
            adapter = SiliconFlowAdapter()

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for content, reasoning in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}]
                ):
                    chunks.append((content, reasoning))

        assert len(chunks) == 3
        assert chunks[0] == ("Hi", None)
        assert chunks[1] == (" there", None)
        assert chunks[2] == ("", "let me think")
        assert adapter.last_usage == {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "total_tokens": 11,
        }


class TestSiliconFlowAdapterSDKReasoning:
    """Test SDK path with reasoning enabled (covers extra_body and kwargs branches)."""

    @pytest.mark.asyncio
    async def test_sdk_with_reasoning_and_kwargs(self):
        """Cover enable_reasoning + thinking_budget + extra kwargs."""

        class MockDelta:
            def __init__(self, content=None, reasoning_content=None):
                self.content = content
                self.reasoning_content = reasoning_content
                self.tool_calls = None

        class MockChoice:
            def __init__(self, delta, finish_reason=None):
                self.delta = delta
                self.finish_reason = finish_reason

        class MockChunk:
            def __init__(self, choices=None, usage=None):
                self.choices = choices or []
                self.usage = usage

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(MockDelta(content="A"))])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_client.close = AsyncMock()

        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._SDK_AVAILABLE = True
            adapter = SiliconFlowAdapter()

            with patch.dict(sys.modules, {"openai": fake_openai}):
                chunks = []
                async for c, r in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}],
                    enable_reasoning=True,
                    thinking_budget=1024,
                    temperature=0.7,
                ):
                    chunks.append((c, r))

        assert chunks == [("A", None)]
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"] == {"thinking_budget": 1024}


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
            SiliconFlowAdapter._SDK_AVAILABLE = False
            adapter = SiliconFlowAdapter()

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for c, r in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}],
                    enable_reasoning=True,
                    thinking_budget=512,
                ):
                    chunks.append((c, r))

        assert len(chunks) == 2
        assert chunks[0] == ("OK", None)
        assert chunks[1] == ("", "think")
