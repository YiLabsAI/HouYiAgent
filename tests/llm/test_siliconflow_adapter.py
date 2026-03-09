"""Covers SiliconFlow adapter mock fallback, SDK/httpx paths, retries, and stream parsing."""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter
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
                async for chunk in adapter.stream_chat(messages):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

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
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

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
                async for chunk in adapter.stream_chat(messages):
                    chunks.append(chunk.content_delta)

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
                async for chunk in adapter.stream_chat([{"role": "user", "content": "Hi"}]):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

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
                async for chunk in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}],
                    enable_reasoning=True,
                    thinking_budget=1024,
                    temperature=0.7,
                ):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

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
                async for chunk in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}],
                    enable_reasoning=True,
                    thinking_budget=512,
                ):
                    chunks.append((chunk.content_delta, chunk.reasoning_delta))

        assert len(chunks) == 2
        assert chunks[0] == ("OK", None)
        assert chunks[1] == ("", "think")


class TestSiliconFlowChatRequestSanitization:
    """Test non-stream chat payload sanitation for strict OpenAI-compatible providers."""

    @pytest.mark.asyncio
    async def test_chat_sanitizes_message_content_and_tool_arguments(self):
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
            SiliconFlowAdapter._SDK_AVAILABLE = False
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


class TestSiliconFlowStreamingRequestSanitization:
    """Test stream_chat request payload sanitation for strict providers."""

    @pytest.mark.asyncio
    async def test_stream_chat_sanitizes_message_content_and_tool_arguments(self):
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
            SiliconFlowAdapter._SDK_AVAILABLE = False
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
    def test_build_httpx_chat_body_includes_tools_and_tool_choice(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        body = adapter._build_httpx_chat_body(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-chat",
            tools=[{"type": "function", "function": {"name": "search"}}],
            temperature=0.2,
            max_tokens=64,
            extra_kwargs={"tool_choice": "required"},
        )

        assert body == {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "stream": False,
            "max_tokens": 64,
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "tool_choice": "required",
        }

    def test_build_sdk_stream_kwargs_includes_tool_choice_and_extra_body(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        kwargs = adapter._build_sdk_stream_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-chat",
            enable_reasoning=True,
            thinking_budget=256,
            extra_kwargs={
                "tools": [{"type": "function", "function": {"name": "search"}}],
                "tool_choice": "required",
                "top_p": 0.8,
            },
        )

        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "search"}}]
        assert kwargs["tool_choice"] == "required"
        assert kwargs["extra_body"] == {"thinking_budget": 256}
        assert kwargs["top_p"] == 0.8

    def test_extract_sdk_tool_calls_delta_and_build_chunk(self):
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

        chunk, content_inc, reasoning_inc = adapter._build_sdk_stream_chunk(choice=choice)

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

    def test_parse_httpx_sse_line_handles_done_and_invalid_payload(self):
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

    def test_build_httpx_stream_chunk_updates_usage_and_yields_reasoning(self):
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
    async def test_chat_httpx_retries_on_transport_error(self):
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
            SiliconFlowAdapter._SDK_AVAILABLE = False
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
    async def test_chat_httpx_returns_last_retry_error_when_status_retries_exhausted(self):
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
            SiliconFlowAdapter._SDK_AVAILABLE = False
            adapter = SiliconFlowAdapter()

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch("houyi.adapters.llm.siliconflow_adapter.asyncio.sleep", new=AsyncMock()),
                patch("houyi.infrastructure.net.proxy.detect_proxy", return_value=None),
                pytest.raises(RuntimeError, match="SiliconFlow HTTP 500: server error"),
            ):
                await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert attempts == 4


class TestSiliconFlowAdapterSdkChat:
    @pytest.mark.asyncio
    async def test_chat_returns_mock_response_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            EnvConfig._reset()
            SiliconFlowAdapter._SDK_AVAILABLE = False
            adapter = SiliconFlowAdapter()

        result = await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert result.content == "Mock response (no API key)"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_uses_sdk_path_when_available(self):
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._SDK_AVAILABLE = True
            adapter = SiliconFlowAdapter()

        expected = MagicMock(content="sdk")
        with patch.object(adapter, "_chat_via_sdk", AsyncMock(return_value=expected)) as mocked:
            result = await adapter.chat([{"role": "user", "content": "hi"}], model="deepseek-chat")

        assert result is expected
        mocked.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_via_sdk_passes_tools_and_updates_usage(self):
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
            SiliconFlowAdapter._SDK_AVAILABLE = True
            adapter = SiliconFlowAdapter()

            with patch.dict(sys.modules, {"openai": fake_openai}):
                result = await adapter._chat_via_sdk(
                    [{"role": "user", "content": "hi"}],
                    "deepseek-chat",
                    tools=[{"type": "function", "function": {"name": "search"}}],
                    temperature=0.2,
                    max_tokens=64,
                    tool_choice="required",
                    top_p=0.8,
                )

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
    async def test_handle_httpx_stream_error_response_retries_on_retryable_status(self):
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
                await adapter._handle_httpx_stream_error_response(
                    response=response,
                    retry_controller=retry_controller,
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_handle_httpx_stream_error_response_raises_on_non_retryable_status(self):
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

        with pytest.raises(RuntimeError, match="400 bad request"):
            await adapter._handle_httpx_stream_error_response(
                response=response,
                retry_controller=retry_controller,
            )

    @pytest.mark.asyncio
    async def test_stream_via_httpx_retries_on_transport_error_then_succeeds(self):
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
            SiliconFlowAdapter._SDK_AVAILABLE = False
            adapter = SiliconFlowAdapter()

        with (
            patch("httpx.AsyncClient", return_value=MockHttpxClient()),
            patch("httpx.TransportError", ConnectBoom),
            patch("houyi.adapters.llm.siliconflow_adapter.asyncio.sleep", new=AsyncMock()),
            patch("houyi.infrastructure.net.proxy.detect_proxy", return_value=None),
        ):
            chunks = []
            async for chunk in adapter._stream_via_httpx(
                [{"role": "user", "content": "hi"}],
                "deepseek-chat",
            ):
                chunks.append(chunk.content_delta)

        assert attempts == 2
        assert chunks == ["OK"]

    def test_build_httpx_stream_payload_keeps_non_none_extra_kwargs(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        payload = adapter._build_httpx_stream_payload(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-chat",
            enable_reasoning=False,
            thinking_budget=None,
            extra_kwargs={"top_p": 0.8, "presence_penalty": None},
        )

        assert payload["top_p"] == 0.8
        assert "presence_penalty" not in payload

    def test_parse_httpx_sse_line_returns_none_for_non_dict_json(self):
        adapter = SiliconFlowAdapter(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            default_model="deepseek-chat",
        )

        event, done = adapter._parse_httpx_sse_line("data: [1, 2]")
        assert event is None
        assert done is False

    def test_build_httpx_stream_chunk_returns_none_without_content_or_reasoning(self):
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
