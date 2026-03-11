"""Unit tests for OpenAICompatibleAdapter."""

from __future__ import annotations

import types

import pytest

from houyi.adapters.llm.base import LLMResponse
from houyi.adapters.llm.openai_compat_adapter import OpenAICompatibleAdapter


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
async def test_chat_sanitizes_messages_with_strict_contract(monkeypatch) -> None:
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
async def test_openai_compat_adapter_stream(monkeypatch) -> None:
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
async def test_stream_sanitizes_messages_with_strict_contract(monkeypatch) -> None:
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


def test_openai_compat_adapter_requires_key(monkeypatch) -> None:
    """Adapter should raise when API key is missing."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatibleAdapter()
