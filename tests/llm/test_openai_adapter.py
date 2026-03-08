"""Unit tests for houyi.adapters.llm.openai_adapter retry behavior."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.adapters.llm import openai_adapter as openai_adapter_module
from houyi.adapters.llm.openai_adapter import OpenAIAdapter


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.headers = headers or {}


class _FakeAPIError(Exception):
    def __init__(
        self, message: str, status_code: int | None = None, headers: dict[str, str] | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeResponse(status_code, headers) if status_code is not None else None


def _make_openai_like_response(content: str = "ok") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = 2
    response.usage.completion_tokens = 1
    response.usage.total_tokens = 3
    response.model = "gpt-4"
    return response


def _make_stream_chunk(content: str) -> MagicMock:
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = None
    chunk.usage = None
    return chunk


def _build_adapter_with_client(create_mock: AsyncMock) -> OpenAIAdapter:
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=create_mock,
            )
        )
    )
    fake_openai_module = types.SimpleNamespace(AsyncOpenAI=lambda **_: client)
    with patch.dict("sys.modules", {"openai": fake_openai_module}):
        return OpenAIAdapter(api_key="test-key")


class TestOpenAIAdapterRetry:
    @pytest.mark.asyncio
    async def test_chat_retries_on_retryable_status(self):
        create_mock = AsyncMock(
            side_effect=[
                _FakeAPIError("rate limited", status_code=429, headers={"Retry-After": "0"}),
                _make_openai_like_response("done"),
            ]
        )
        adapter = _build_adapter_with_client(create_mock)

        with patch.object(openai_adapter_module.asyncio, "sleep", new_callable=AsyncMock):
            response = await adapter.chat([{"role": "user", "content": "hi"}])

        assert response.content == "done"
        assert create_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_chat_does_not_retry_non_retryable_status(self):
        create_mock = AsyncMock(side_effect=_FakeAPIError("bad request", status_code=400))
        adapter = _build_adapter_with_client(create_mock)

        with pytest.raises(_FakeAPIError):
            await adapter.chat([{"role": "user", "content": "hi"}])

        assert create_mock.await_count == 1

    @pytest.mark.asyncio
    async def test_stream_retries_before_first_chunk(self):
        async def good_stream():
            yield _make_stream_chunk("ok")

        create_mock = AsyncMock(side_effect=[_FakeAPIError("connect fail"), good_stream()])
        adapter = _build_adapter_with_client(create_mock)

        with patch.object(openai_adapter_module.asyncio, "sleep", new_callable=AsyncMock):
            chunks = []
            async for chunk in adapter.stream_chat([{"role": "user", "content": "hi"}]):
                chunks.append(chunk.content_delta)

        assert chunks == ["ok"]
        assert create_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_stream_does_not_retry_after_emitting_chunks(self):
        async def bad_stream():
            yield _make_stream_chunk("part")
            raise _FakeAPIError("stream interrupted")

        create_mock = AsyncMock(return_value=bad_stream())
        adapter = _build_adapter_with_client(create_mock)

        with pytest.raises(_FakeAPIError):
            chunks = []
            async for chunk in adapter.stream_chat([{"role": "user", "content": "hi"}]):
                chunks.append(chunk.content_delta)

        assert create_mock.await_count == 1
