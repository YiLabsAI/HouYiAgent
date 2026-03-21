"""Tests for StreamResponse accumulation and adapter stream_chat StreamChunk contract.

Covers:
- StreamResponse accumulates content and reasoning from yielded StreamChunk objects
- StreamResponse base-layer tool_calls delta accumulation
- StreamResponse.finalize() pulls usage/finish_reason from adapter + parses tool_calls
- StreamResponse.to_response() produces correct LLMResponse
- stream_completion() yields StreamChunk objects
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from houyi.adapters.llm.base import (
    DEFAULT_TEMPERATURE,
    LLMAdapter,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    StreamResponse,
)

# ---------------------------------------------------------------------------
# Fake adapter that simulates the stream_chat StreamChunk contract
# ---------------------------------------------------------------------------


class _FakeStreamAdapter(LLMAdapter):
    """Adapter that yields predetermined StreamChunk objects."""

    def __init__(
        self,
        chunks: list[StreamChunk],
        usage: dict[str, int] | None = None,
        finish_reason: str | None = "stop",
    ) -> None:
        self._chunks = chunks
        self.last_usage: dict[str, int] = usage or {}
        self.last_finish_reason: str | None = finish_reason
        self.model = "fake-model"

    async def chat(self, messages: Any, **kw: Any) -> LLMResponse:
        return LLMResponse(content="", tool_calls=[], finish_reason="stop", usage={}, model="fake")

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self._chunks:
            yield chunk


# ---------------------------------------------------------------------------
# StreamResponse accumulation tests
# ---------------------------------------------------------------------------


class TestStreamResponseAccumulation:
    """StreamResponse should accumulate content/reasoning from yielded chunks."""

    @pytest.mark.asyncio
    async def test_accumulates_content(self) -> None:
        chunks: list[StreamChunk] = [
            StreamChunk(content_delta="Hello"),
            StreamChunk(content_delta=" world"),
            StreamChunk(content_delta="!"),
        ]
        adapter = _FakeStreamAdapter(chunks)
        stream = StreamResponse(adapter.stream_chat([]))
        collected: list[str] = []
        async for chunk in stream:
            collected.append(chunk.content_delta)
        assert collected == ["Hello", " world", "!"]
        assert stream.accumulated_content == "Hello world!"

    @pytest.mark.asyncio
    async def test_accumulates_reasoning(self) -> None:
        chunks: list[StreamChunk] = [
            StreamChunk(content_delta="c1", reasoning_delta="r1"),
            StreamChunk(content_delta="c2", reasoning_delta="r2"),
            StreamChunk(content_delta="c3"),
        ]
        adapter = _FakeStreamAdapter(chunks)
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        assert stream.accumulated_content == "c1c2c3"
        assert stream.accumulated_reasoning == "r1r2"

    @pytest.mark.asyncio
    async def test_empty_stream(self) -> None:
        adapter = _FakeStreamAdapter([])
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        assert stream.accumulated_content == ""
        assert stream.accumulated_reasoning == ""


# ---------------------------------------------------------------------------
# StreamResponse.finalize() pulls metadata from adapter
# ---------------------------------------------------------------------------


class TestStreamResponseToolCallsAccumulation:
    """StreamResponse base-layer tool_calls delta accumulation."""

    @pytest.mark.asyncio
    async def test_accumulates_single_toolcall(self) -> None:
        """Single tool call across multiple delta chunks."""
        tc_chunks: list[StreamChunk] = [
            StreamChunk(
                tool_calls_delta=[{"index": 0, "id": "call_1", "function": {"name": "search"}}]
            ),
            StreamChunk(tool_calls_delta=[{"index": 0, "function": {"arguments": '{"q":'}}]),
            StreamChunk(tool_calls_delta=[{"index": 0, "function": {"arguments": '"hello"}'}}]),
        ]
        adapter = _FakeStreamAdapter(
            tc_chunks,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="tool_calls",
        )
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)

        assert len(stream.tool_calls) == 1
        tc = stream.tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == {"q": "hello"}
        assert stream.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert stream.finish_reason == "tool_calls"
        assert stream.model == "fake-model"

    @pytest.mark.asyncio
    async def test_accumulates_parallel_toolcalls(self) -> None:
        """Multiple tool calls (parallel) across delta chunks."""
        tc_chunks: list[StreamChunk] = [
            StreamChunk(
                tool_calls_delta=[
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": "search", "arguments": '{"q":"a"}'},
                    },
                    {
                        "index": 1,
                        "id": "call_b",
                        "function": {"name": "read", "arguments": '{"f":"b"}'},
                    },
                ]
            ),
        ]
        adapter = _FakeStreamAdapter(tc_chunks, finish_reason="tool_calls")
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)

        assert len(stream.tool_calls) == 2
        assert stream.tool_calls[0]["function"]["name"] == "search"
        assert stream.tool_calls[1]["function"]["name"] == "read"

    @pytest.mark.asyncio
    async def test_no_toolcalls(self) -> None:
        adapter = _FakeStreamAdapter(
            [StreamChunk(content_delta="done")],
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            finish_reason="stop",
        )
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)

        assert stream.tool_calls == []
        assert stream.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_finalize_missing_attributes(self) -> None:
        """finalize() should not crash if adapter lacks last_* attributes."""

        class _BareAdapter(LLMAdapter):
            model = "bare"

            async def chat(self, messages: Any, **kw: Any) -> LLMResponse:
                return LLMResponse(
                    content="", tool_calls=[], finish_reason="stop", usage={}, model="bare"
                )

            async def stream_chat(self, messages: Any, **kw: Any) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(content_delta="x")

        adapter = _BareAdapter()
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)

        assert stream.tool_calls == []
        assert stream.usage == {}
        assert stream.finish_reason is None


# ---------------------------------------------------------------------------
# StreamResponse.to_response()
# ---------------------------------------------------------------------------


class TestStreamResponseToResponse:
    """to_response() should produce a correct LLMResponse from accumulated data."""

    @pytest.mark.asyncio
    async def test_response_after_finalize(self) -> None:
        chunks: list[StreamChunk] = [
            StreamChunk(content_delta="result: "),
            StreamChunk(
                content_delta="42",
                tool_calls_delta=[
                    {"index": 0, "id": "tc1", "function": {"name": "echo", "arguments": '{"x":1}'}}
                ],
            ),
        ]
        adapter = _FakeStreamAdapter(
            chunks,
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            finish_reason="tool_calls",
        )
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)

        resp = stream.to_response()
        assert isinstance(resp, LLMResponse)
        assert resp.content == "result: 42"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["function"]["arguments"] == {"x": 1}
        assert resp.finish_reason == "tool_calls"
        assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        assert resp.model == "fake-model"

    @pytest.mark.asyncio
    async def test_response_without_finalize(self) -> None:
        """to_response() should work even without finalize, using defaults."""
        adapter = _FakeStreamAdapter([StreamChunk(content_delta="hello")])
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        # No finalize call
        resp = stream.to_response()
        assert resp.content == "hello"
        assert resp.tool_calls == []
        assert resp.finish_reason == "stop"
        assert resp.usage == {}
        assert resp.model == "unknown"

    @pytest.mark.asyncio
    async def test_response_keeps_reasoning_metadata(self) -> None:
        adapter = _FakeStreamAdapter(
            [StreamChunk(content_delta="answer", reasoning_delta="step 1")]
        )
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)

        resp = stream.to_response()

        assert resp.content == "answer"
        assert resp.metadata["reasoning_content"] == "step 1"

    @pytest.mark.asyncio
    async def test_response_reasoning_only_metadata(self) -> None:
        adapter = _FakeStreamAdapter([StreamChunk(reasoning_delta="step only")])
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)

        resp = stream.to_response()

        assert resp.content == ""
        assert resp.metadata["reasoning_content"] == "step only"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestStreamResponseEdgeCases:
    """Edge-case coverage for StreamResponse."""

    @pytest.mark.asyncio
    async def test_empty_content_chunks(self) -> None:
        """Empty-string content should not pollute accumulated_content."""
        chunks: list[StreamChunk] = [
            StreamChunk(),
            StreamChunk(content_delta="hi"),
            StreamChunk(reasoning_delta="r"),
            StreamChunk(),
        ]
        adapter = _FakeStreamAdapter(chunks)
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        assert stream.accumulated_content == "hi"
        assert stream.accumulated_reasoning == "r"

    @pytest.mark.asyncio
    async def test_finalize_twice_idempotent(self) -> None:
        """finalize called twice does not duplicate tool_calls."""
        chunks: list[StreamChunk] = [
            StreamChunk(
                tool_calls_delta=[
                    {"index": 0, "id": "t1", "function": {"name": "a", "arguments": "{}"}}
                ]
            ),
        ]
        adapter = _FakeStreamAdapter(
            chunks,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason="stop",
        )
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)
        stream.finalize(adapter)
        assert len(stream.tool_calls) == 1
        assert stream.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_finalize_without_adapter(self) -> None:
        """finalize() without adapter still parses accumulated tool_calls."""
        chunks: list[StreamChunk] = [
            StreamChunk(
                tool_calls_delta=[
                    {"index": 0, "id": "t1", "function": {"name": "a", "arguments": '{"x":1}'}}
                ]
            ),
        ]
        adapter = _FakeStreamAdapter(chunks)
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize()  # no adapter
        assert len(stream.tool_calls) == 1
        assert stream.tool_calls[0]["function"]["arguments"] == {"x": 1}
        assert stream.usage == {}  # not pulled from adapter

    @pytest.mark.asyncio
    async def test_stream_completion(self) -> None:
        """LLMAdapter.stream_completion() should yield StreamChunk objects."""
        adapter = _FakeStreamAdapter([StreamChunk(content_delta="prompt reply")])
        collected = []
        async for chunk in adapter.stream_completion("hello"):
            collected.append((chunk.content_delta, chunk.reasoning_delta))
        assert collected == [("prompt reply", None)]

    @pytest.mark.asyncio
    async def test_toolcall_args(self) -> None:
        """If arguments is not valid JSON, finalize keeps it as a string."""
        chunks: list[StreamChunk] = [
            StreamChunk(
                tool_calls_delta=[
                    {"index": 0, "id": "t1", "function": {"name": "x", "arguments": "not-json"}}
                ]
            ),
        ]
        adapter = _FakeStreamAdapter(chunks)
        stream = StreamResponse(adapter.stream_chat([]))
        async for _chunk in stream:
            pass
        stream.finalize(adapter)
        assert stream.tool_calls[0]["function"]["arguments"] == "not-json"
