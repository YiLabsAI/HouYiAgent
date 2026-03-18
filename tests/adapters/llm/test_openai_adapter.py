"""Covers OpenAI adapter retry decisions, streaming deltas, and usage tracking."""

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


class TestOpenAIAdapterHelpers:
    def test_extract_status_code_prefers_direct_status_code(self):
        exc = _FakeAPIError("rate limited", status_code=429)
        assert OpenAIAdapter._extract_status_code(exc) == 429

    def test_extract_status_code_falls_back_to_response_status(self):
        exc = Exception("transport")
        exc.response = _FakeResponse(503)  # type: ignore[attr-defined]
        assert OpenAIAdapter._extract_status_code(exc) == 503

    def test_extract_status_code_returns_none_without_status(self):
        assert OpenAIAdapter._extract_status_code(Exception("boom")) is None

    def test_build_stream_chunk_with_tool_call_delta_sets_finish_reason(self):
        adapter = _build_adapter_with_client(AsyncMock())

        function = types.SimpleNamespace(name="search", arguments='{"query":"hi"}')
        tool_delta = types.SimpleNamespace(index=0, id="call_1", function=function)
        delta = types.SimpleNamespace(content=None, tool_calls=[tool_delta])
        choice = types.SimpleNamespace(delta=delta, finish_reason="tool_calls")

        chunk = adapter._build_stream_chunk(choice)

        assert chunk is not None
        assert chunk.content_delta == ""
        assert chunk.tool_calls_delta == [
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "search", "arguments": '{"query":"hi"}'},
            }
        ]
        assert adapter.last_finish_reason == "tool_calls"

    def test_build_stream_chunk_returns_none_when_delta_has_no_content_or_tool_calls(self):
        adapter = _build_adapter_with_client(AsyncMock())
        delta = types.SimpleNamespace(content=None, tool_calls=None)
        choice = types.SimpleNamespace(delta=delta, finish_reason=None)

        assert adapter._build_stream_chunk(choice) is None

    def test_update_stream_usage_records_prompt_completion_and_total(self):
        adapter = _build_adapter_with_client(AsyncMock())
        usage = types.SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=7,
            total_tokens=12,
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=4),
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=2),
            prompt_cache_hit_tokens=2,
        )
        chunk = types.SimpleNamespace(usage=usage)

        adapter._update_stream_usage(chunk)

        assert adapter.last_usage == {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 4},
            "prompt_tokens_details": {"cached_tokens": 2},
            "prompt_cache_hit_tokens": 2,
        }

    def test_build_chat_params_includes_stream_options_tools_and_extra_kwargs(self):
        adapter = _build_adapter_with_client(AsyncMock())

        params = adapter._build_chat_params(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=123,
            tools=[{"type": "function", "function": {"name": "search"}}],
            stream=True,
            extra_kwargs={"tool_choice": "required"},
        )

        assert params["model"] == "gpt-4"
        assert params["messages"] == [{"role": "user", "content": "hi"}]
        assert params["temperature"] == 0.2
        assert params["max_tokens"] == 123
        assert params["stream"] is True
        assert params["stream_options"] == {"include_usage": True}
        assert params["tools"] == [{"type": "function", "function": {"name": "search"}}]
        assert params["tool_choice"] == "required"
