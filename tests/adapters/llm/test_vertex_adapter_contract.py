from __future__ import annotations

import pytest

from houyi.adapters.llm.base import StreamChunk, StreamResponse


class _FakeAdapter:
    def __init__(
        self, *, chunks: list[StreamChunk], finish_reason: str, model: str = "vertex-test"
    ) -> None:
        self._chunks = chunks
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self.last_finish_reason = finish_reason
        self.model = model

    async def stream_chat(self, _messages):
        for chunk in self._chunks:
            yield chunk


async def _collect_response(chunks: list[StreamChunk], finish_reason: str) -> StreamResponse:
    adapter = _FakeAdapter(chunks=chunks, finish_reason=finish_reason)
    stream = StreamResponse(adapter.stream_chat([]))
    async for _chunk in stream:
        pass
    stream.finalize(adapter)
    return stream


class TestVertexAdapterStreamContract:
    @pytest.mark.asyncio
    async def test_stream_accumulates_tools(self) -> None:
        stream = await _collect_response(
            [
                StreamChunk(
                    tool_calls_delta=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search"},
                        }
                    ]
                ),
                StreamChunk(
                    tool_calls_delta=[
                        {
                            "index": 0,
                            "function": {"arguments": '{"q":"docs"}'},
                        }
                    ]
                ),
            ],
            finish_reason="tool_calls",
        )

        assert stream.finish_reason == "tool_calls"
        assert stream.tool_calls == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": {"q": "docs"}},
            }
        ]

    @pytest.mark.asyncio
    async def test_stream_keeps_reasoning(self) -> None:
        stream = await _collect_response(
            [
                StreamChunk(reasoning_delta="think"),
                StreamChunk(content_delta="done"),
            ],
            finish_reason="stop",
        )

        assert stream.accumulated_reasoning == "think"
        assert stream.accumulated_content == "done"
        assert stream.finish_reason == "stop"
