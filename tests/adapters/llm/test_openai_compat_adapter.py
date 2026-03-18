"""Unit tests for OpenAICompatibleAdapter."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, patch

import pytest

from houyi.adapters.llm.base import LLMResponse
from houyi.adapters.llm.openai_compat_adapter import OpenAICompatibleAdapter
from houyi.adapters.llm.openai_compat_base import OpenAICompatAdapterBase


class _FakeChatCompletions:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeStream:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def _iter():
            for chunk in self._chunks:
                yield chunk

        return _iter()


class _FakeChat:
    def __init__(self, response: object) -> None:
        self.completions = _FakeChatCompletions(response)


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.chat = _FakeChat(response)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = []
        self.function_call = None


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 1
        self.completion_tokens = 1
        self.total_tokens = 2
        self.completion_tokens_details = types.SimpleNamespace(reasoning_tokens=7)
        self.prompt_tokens_details = types.SimpleNamespace(cached_tokens=3)
        self.prompt_cache_hit_tokens = 3
        self.prompt_cache_miss_tokens = 1


class _FakeOpenAIResponse:
    def __init__(self, content: str, model: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()
        self.model = model


def _build_openai_module(response: object) -> types.SimpleNamespace:
    class _AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self._client = _FakeClient(response)
            self.chat = self._client.chat

    return types.SimpleNamespace(AsyncOpenAI=_AsyncOpenAI)


class _FakeStreamUsage:
    def __init__(self, prompt_tokens: int = 1, completion_tokens: int = 2, total_tokens: int = 3):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeStreamDelta:
    def __init__(self, content: str | None = None, reasoning_content: str | None = None) -> None:
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeStreamingChoice:
    def __init__(
        self,
        *,
        content: str | None = None,
        reasoning_content: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.delta = _FakeStreamDelta(content=content, reasoning_content=reasoning_content)
        self.finish_reason = finish_reason


class _FakeStreamingChunk:
    def __init__(self, choices: list[object], usage: object | None = None) -> None:
        self.choices = choices
        self.usage = usage


class _ReasoningOpenAICompatAdapter(OpenAICompatAdapterBase):
    def __init__(self) -> None:
        self.api_key = "test-key"
        self.base_url = "https://example.test"
        self.default_headers = {"X-Test": "1"}
        self.model = "reasoning-model"
        self.last_usage = None
        self.last_finish_reason = None
        self.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=AsyncMock()))
        )

    def _build_reasoning_extra_body(self, request) -> dict[str, object] | None:
        return {"reasoning": "on"}


def test_build_request(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model", strict_message_string_contract=True)
    request = adapter._build_request(
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": {"x": 1}},
                    }
                ],
            }
        ],
        tools=[{"type": "function", "function": {"name": "tool", "parameters": {}}}],
        temperature=0.2,
        max_tokens=32,
        enable_streaming=True,
        kwargs={
            "model": "alt-model",
            "top_p": 0.7,
            "top_k": 20,
            "frequency_penalty": 0.4,
            "tool_choice": "required",
        },
    )

    assert request.model == "alt-model"
    assert request.enable_streaming is True
    assert request.max_tokens == 32
    assert request.top_p == 0.7
    assert request.top_k == 20
    assert request.frequency_penalty == 0.4
    assert request.tool_choice == "required"
    assert request.messages[0]["content"] == "hello"
    assert isinstance(request.messages[0]["tool_calls"][0]["function"]["arguments"], str)


def test_resolve_transport(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.1,
        max_tokens=None,
        enable_streaming=False,
        kwargs={},
    )

    monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "httpx")
    assert adapter._resolve_transport(request) == "httpx"
    monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "sdk")
    assert adapter._resolve_transport(request) == "sdk"
    monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "auto")
    assert adapter._resolve_transport(request) == "sdk"


def test_encode_request(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "tool", "parameters": {}}}],
        temperature=0.1,
        max_tokens=10,
        enable_streaming=False,
        kwargs={
            "model": "alt-model",
            "top_p": 0.7,
            "top_k": 12,
            "frequency_penalty": 0.6,
            "tool_choice": "required",
        },
    )

    payload = adapter._encode_chat_request(request)

    assert payload["model"] == "alt-model"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 10
    assert payload["tools"] == [
        {"type": "function", "function": {"name": "tool", "parameters": {}}}
    ]
    assert payload["tool_choice"] == "required"
    assert payload["top_p"] == 0.7
    assert payload["top_k"] == 12
    assert payload["frequency_penalty"] == 0.6
    assert "stream" not in payload


def test_encode_stream_request(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.1,
        max_tokens=None,
        enable_streaming=True,
        kwargs={"model": "alt-model", "top_p": 0.7},
    )

    payload = adapter._encode_stream_request(request)

    assert payload["model"] == "alt-model"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["top_p"] == 0.7


def test_encode_request_for_httpx(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.3,
        max_tokens=12,
        enable_streaming=False,
        kwargs={"top_p": 0.4, "top_k": 8, "frequency_penalty": 0.2},
    )

    payload = adapter._encode_chat_request_for_httpx(request)

    assert payload["stream"] is False
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 12
    assert payload["top_p"] == 0.4
    assert payload["top_k"] == 8
    assert payload["frequency_penalty"] == 0.2


def test_encode_stream_request_for_httpx(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.3,
        max_tokens=None,
        enable_streaming=True,
        kwargs={},
    )

    payload = adapter._encode_stream_request_for_httpx(request)

    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_encode_request_without_usage(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.3,
        max_tokens=None,
        enable_streaming=True,
        kwargs={"include_stream_usage": False},
    )

    payload = adapter._encode_stream_request_for_httpx(request)

    assert payload["stream"] is True
    assert "stream_options" not in payload


def test_parse_sse_event(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")

    assert adapter._parse_httpx_sse_event("event: ping") is None
    assert adapter._parse_httpx_sse_event("data: ") is None
    assert adapter._parse_httpx_sse_event("data: [DONE]") is None
    assert adapter._parse_httpx_sse_event('data: {"choices":[{"delta":{"content":"ok"}}]}') == {
        "choices": [{"delta": {"content": "ok"}}]
    }


def test_build_stream_chunk(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")

    assert adapter._build_stream_chunk_from_httpx_event({"choices": []}) is None
    assert adapter._build_stream_chunk_from_httpx_event({"choices": ["bad"]}) is None

    reasoning_chunk = adapter._build_stream_chunk_from_httpx_event(
        {
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            "choices": [{"finish_reason": "stop", "delta": {"reasoning_content": "thinking"}}],
        }
    )

    assert reasoning_chunk is not None
    assert reasoning_chunk.content_delta == ""
    assert reasoning_chunk.reasoning_delta == "thinking"
    assert adapter.last_usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert adapter.last_finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_keeps_usage(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    result = await adapter.chat([{"role": "user", "content": "hi"}])

    assert result.usage["completion_tokens_details"]["reasoning_tokens"] == 7
    assert result.usage["prompt_tokens_details"]["cached_tokens"] == 3
    assert result.usage["prompt_cache_hit_tokens"] == 3
    assert result.usage["prompt_cache_miss_tokens"] == 1


@pytest.mark.asyncio
async def test_dispatches_chat_by_transport(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.1,
        max_tokens=None,
        enable_streaming=False,
        kwargs={},
    )

    with (
        patch.object(
            adapter,
            "_chat_request",
            AsyncMock(
                return_value=LLMResponse(content="sdk", finish_reason="stop", model="test-model")
            ),
        ) as chat,
        patch.object(
            adapter,
            "_chat_request_httpx",
            AsyncMock(
                return_value=LLMResponse(content="httpx", finish_reason="stop", model="test-model")
            ),
        ) as httpx_chat,
    ):
        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "sdk")
        result = await adapter._chat(request)
        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "httpx")
        httpx_result = await adapter._chat(request)

    assert result.content == "sdk"
    assert httpx_result.content == "httpx"
    chat.assert_called_once_with(request)
    httpx_chat.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_dispatches_stream(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.1,
        max_tokens=None,
        enable_streaming=True,
        kwargs={},
    )

    async def _direct_stream(_request):
        yield type("Chunk", (), {"content_delta": "sdk", "reasoning_delta": None})()

    async def _httpx_stream(_request):
        yield type("Chunk", (), {"content_delta": "httpx", "reasoning_delta": None})()

    with (
        patch.object(adapter, "_stream_request", side_effect=_direct_stream) as direct_stream,
        patch.object(adapter, "_stream_request_httpx", side_effect=_httpx_stream) as httpx_stream,
    ):
        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "sdk")
        direct_chunks = [chunk async for chunk in adapter._stream_chat(request)]
        monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "httpx")
        httpx_chunks = [chunk async for chunk in adapter._stream_chat(request)]

    assert [chunk.content_delta for chunk in direct_chunks] == ["sdk"]
    assert [chunk.content_delta for chunk in httpx_chunks] == ["httpx"]
    direct_stream.assert_called_once_with(request)
    httpx_stream.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_stream_request_updates_usage(monkeypatch) -> None:
    stream = _FakeStream(
        [
            _FakeStreamingChunk(
                [_FakeStreamingChoice(reasoning_content="thinking", finish_reason="stop")],
                usage=_FakeStreamUsage(4, 5, 9),
            ),
        ]
    )
    fake_openai = _build_openai_module(stream)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.1,
        max_tokens=None,
        enable_streaming=True,
        kwargs={},
    )

    chunks = [chunk async for chunk in adapter._stream_request(request)]

    assert len(chunks) == 1
    assert chunks[0].content_delta == ""
    assert chunks[0].reasoning_delta == "thinking"
    assert adapter.last_usage == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
    assert adapter.last_finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_request_skips_empty(monkeypatch) -> None:
    stream = _FakeStream([_FakeStreamingChunk([])])
    fake_openai = _build_openai_module(stream)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.1,
        max_tokens=None,
        enable_streaming=True,
        kwargs={},
    )

    chunks = [chunk async for chunk in adapter._stream_request(request)]

    assert chunks == []


@pytest.mark.asyncio
async def test_stream_request_httpx_reads_sse(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(_FakeOpenAIResponse("ok", "test-model"))
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    class _HttpxStreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            yield "event: ping"
            yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
            yield 'data: {"choices":[{"finish_reason":"stop","delta":{"reasoning_content":"think"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'
            yield "data: [DONE]"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _HttpxClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, json=None, headers=None):
            return _HttpxStreamResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: _HttpxClient())

    adapter = OpenAICompatibleAdapter(model="test-model")
    request = adapter._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.1,
        max_tokens=None,
        enable_streaming=True,
        kwargs={},
    )

    chunks = [chunk async for chunk in adapter._stream_request_httpx(request)]

    assert [chunk.content_delta for chunk in chunks] == ["hello", ""]
    assert [chunk.reasoning_delta for chunk in chunks] == [None, "think"]
    assert adapter.last_usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert adapter.last_finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_compat_adapter_chat(monkeypatch) -> None:
    """OpenAICompatibleAdapter should call OpenAI client and normalize response."""

    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model")
    result = await adapter.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "tool", "parameters": {}}}],
        max_tokens=10,
        temperature=0.1,
    )

    assert isinstance(result, LLMResponse)
    assert result.content == "ok"
    assert adapter.client.chat.completions.calls[0]["max_tokens"] == 10


@pytest.mark.asyncio
async def test_chat_sanitizes_messages(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    adapter = OpenAICompatibleAdapter(model="test-model", strict_message_string_contract=True)
    await adapter.chat(
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "tool", "arguments": {"x": 1}},
                    }
                ],
            }
        ]
    )

    sent_messages = adapter.client.chat.completions.calls[0]["messages"]
    assert sent_messages[0]["content"] == "hello"
    assert isinstance(sent_messages[0]["tool_calls"][0]["function"]["arguments"], str)


@pytest.mark.asyncio
async def test_openai_compat_adapter(monkeypatch) -> None:
    """OpenAICompatibleAdapter should stream content chunks."""

    class _Delta:
        def __init__(self, content: str) -> None:
            self.content = content

    class _StreamChoice:
        def __init__(self, content: str) -> None:
            self.delta = _Delta(content)

    class _StreamChunk:
        def __init__(self, content: str) -> None:
            self.choices = [_StreamChoice(content)]

    stream = _FakeStream([_StreamChunk("hello"), _StreamChunk("world")])
    fake_openai = _build_openai_module(stream)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    adapter = OpenAICompatibleAdapter(model="test-model")
    chunks = []
    async for chunk in adapter.stream_chat([{"role": "user", "content": "hi"}]):
        chunks.append(chunk.content_delta)

    assert chunks == ["hello", "world"]


@pytest.mark.asyncio
async def test_stream_sanitizes_messages(monkeypatch) -> None:
    class _Delta:
        def __init__(self, content: str) -> None:
            self.content = content

    class _StreamChoice:
        def __init__(self, content: str) -> None:
            self.delta = _Delta(content)

    class _StreamChunk:
        def __init__(self, content: str) -> None:
            self.choices = [_StreamChoice(content)]

    stream = _FakeStream([_StreamChunk("hello")])
    fake_openai = _build_openai_module(stream)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    adapter = OpenAICompatibleAdapter(model="test-model", strict_message_string_contract=True)
    chunks = []
    async for chunk in adapter.stream_chat([{"role": "assistant", "content": {"k": "v"}}]):
        chunks.append(chunk.content_delta)

    assert chunks == ["hello"]
    sent_messages = adapter.client.chat.completions.calls[0]["messages"]
    assert isinstance(sent_messages[0]["content"], str)


@pytest.mark.asyncio
async def test_chat_httpx(monkeypatch) -> None:
    response = _FakeOpenAIResponse("ok", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    monkeypatch.setenv("HOUYI_OPENAI_COMPAT_TRANSPORT", "httpx")
    fake_openai = _build_openai_module(response)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    captured: dict[str, object] = {}

    class _HttpxResponse:
        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def json():
            return {
                "model": "test-model",
                "choices": [
                    {"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    class _HttpxClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _HttpxResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: _HttpxClient())

    adapter = OpenAICompatibleAdapter(model="test-model")
    result = await adapter.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.1,
    )

    assert result.content == "ok"
    assert captured["url"] == "https://example.test/chat/completions"
    assert captured["json"]["stream"] is False
    assert captured["json"]["max_tokens"] == 10


def test_requires_key(monkeypatch) -> None:
    """Adapter should raise when API key is missing."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatibleAdapter()


def test_requires_openai_package(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delitem(__import__("sys").modules, "openai", raising=False)

    original_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("missing openai")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        with pytest.raises(ImportError):
            OpenAICompatibleAdapter()
