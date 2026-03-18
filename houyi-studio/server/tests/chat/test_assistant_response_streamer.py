from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from houyi_studio.server.chat.assistant_response_streamer import AssistantResponseStreamer

from houyi.adapters.llm.base import StreamChunk
from houyi.infrastructure.observability import Span


@contextmanager
def _stage_span(*args, **kwargs):
    yield args[0] if args else None


def _parse_sse(raw_chunks: list[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for chunk in raw_chunks:
        item: dict[str, object] = {}
        for line in chunk.strip().splitlines():
            if line.startswith("id: "):
                item["id"] = line.removeprefix("id: ")
            if line.startswith("event: "):
                item["event"] = line.removeprefix("event: ")
            if line.startswith("data: "):
                item["data"] = json.loads(line.removeprefix("data: "))
        events.append(item)
    return events


class TestAssistantResponseStreamer:
    @pytest.mark.asyncio
    async def test_replay_events(self):
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: "stop",
            finalize_stream_result=lambda **kwargs: ({}, "stop", {}),
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: value,
            stage_span=_stage_span,
        )

        chunks, content_parts, reasoning_parts = await streamer.stream_replay_chunks(
            replay_response=SimpleNamespace(
                content=(
                    "hello world from replay streaming with a longer final answer that should "
                    "arrive across several delta chunks instead of one abrupt block"
                ),
                metadata={
                    "reasoning_content": (
                        "think through replay path carefully and surface the final answer in "
                        "several reasoning deltas"
                    )
                },
            ),
            assistant_message_id="msg-1",
            model="model-1",
            context_usage={"used_tokens": 10},
            finish_reason="stop",
        )

        parsed = _parse_sse(chunks)
        assert parsed[0]["event"] == "context.usage"
        deltas = [item for item in parsed if item["event"] == "message.delta"]
        assert len(deltas) >= 2
        assert "".join(str(item["data"].get("content", "")) for item in deltas) == (
            "hello world from replay streaming with a longer final answer that should arrive across several delta chunks instead of one abrupt block"
        )
        assert "".join(str(item["data"].get("reasoning_content", "")) for item in deltas) == (
            "think through replay path carefully and surface the final answer in several reasoning deltas"
        )
        assert parsed[-1]["event"] == "message.finish"
        assert content_parts == [
            "hello world from replay streaming with a longer final answer that should arrive across several delta chunks instead of one abrupt block"
        ]
        assert reasoning_parts == [
            "think through replay path carefully and surface the final answer in several reasoning deltas"
        ]

    @pytest.mark.asyncio
    async def test_replay_sanitizes_tool_markers(self):
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: "stop",
            finalize_stream_result=lambda **kwargs: ({}, "stop", {}),
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: value,
            stage_span=_stage_span,
        )

        chunks, content_parts, reasoning_parts = await streamer.stream_replay_chunks(
            replay_response=SimpleNamespace(
                content="[tool call]<tool_call>demo</tool_call> final answer",
                metadata={"reasoning_content": "<think>step</think>"},
            ),
            assistant_message_id="msg-1",
            model="model-1",
            context_usage={"used_tokens": 10},
            finish_reason="stop",
        )

        parsed = _parse_sse(chunks)
        deltas = [item for item in parsed if item["event"] == "message.delta"]
        assert "".join(str(item["data"].get("content", "")) for item in deltas) == "final answer"
        assert "".join(str(item["data"].get("reasoning_content", "")) for item in deltas) == "step"
        assert content_parts == ["final answer"]
        assert reasoning_parts == ["step"]

    @pytest.mark.asyncio
    async def test_empty_fallback(self):
        async def fake_stream_chat(**kwargs):
            yield StreamChunk(content_delta="", reasoning_delta=None)

        finalize = MagicMock(return_value=({"prompt_tokens": 3}, "stop", {"metric": 1}))
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: "stop",
            finalize_stream_result=finalize,
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: value,
            stage_span=_stage_span,
        )
        adapter = SimpleNamespace(
            stream_chat=fake_stream_chat,
            last_usage={"prompt_tokens": 3},
            last_finish_reason="stop",
        )

        (
            chunks,
            content_parts,
            reasoning_parts,
            usage_payload,
            finish_reason,
            metadata,
        ) = await streamer.stream_final_response(
            llm_adapter=adapter,
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"temperature": 0.1},
            model="model-1",
            conversation_id="conv-1",
            assistant_message_id="msg-1",
            context_usage={"used_tokens": 10},
            chat_span=Span(name="chat.request"),
        )

        parsed = _parse_sse(chunks)
        deltas = [item for item in parsed if item["event"] == "message.delta"]
        assert deltas[-1]["data"]["content"] == "empty fallback"
        assert content_parts == ["empty fallback"]
        assert reasoning_parts == []
        assert usage_payload == {"prompt_tokens": 3}
        assert finish_reason == "stop"
        assert metadata == {
            "metric": 1,
            "final_stream_message_count": 1,
            "final_stream_input_chars": 2,
            "final_stream_chunk_count": 3,
        }
        finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_fallsback(self):
        async def fake_stream_chat(**kwargs):
            yield StreamChunk(
                content_delta="[tool call]<tool_call>demo</tool_call>",
                reasoning_delta=None,
            )

        finalize = MagicMock(return_value=({"prompt_tokens": 3}, "stop", {"metric": 1}))
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: "stop",
            finalize_stream_result=finalize,
            json_safe=lambda value: value,
            normalize_usage_payload=lambda value: value,
            stage_span=_stage_span,
        )
        adapter = SimpleNamespace(
            stream_chat=fake_stream_chat,
            last_usage={"prompt_tokens": 3},
            last_finish_reason="stop",
        )

        (
            chunks,
            content_parts,
            reasoning_parts,
            usage_payload,
            finish_reason,
            metadata,
        ) = await streamer.stream_final_response(
            llm_adapter=adapter,
            llm_messages=[{"role": "user", "content": "hi"}],
            llm_kwargs={"temperature": 0.1},
            model="model-1",
            conversation_id="conv-1",
            assistant_message_id="msg-1",
            context_usage={"used_tokens": 10},
            chat_span=Span(name="chat.request"),
        )

        parsed = _parse_sse(chunks)
        deltas = [item for item in parsed if item["event"] == "message.delta"]
        assert deltas[-1]["data"]["content"] == "empty fallback"
        assert content_parts == ["empty fallback"]
        assert reasoning_parts == []
        assert usage_payload == {"prompt_tokens": 3}
        assert finish_reason == "stop"
        assert metadata == {
            "metric": 1,
            "final_stream_message_count": 1,
            "final_stream_input_chars": 2,
            "final_stream_chunk_count": 3,
        }
        finalize.assert_called_once()
