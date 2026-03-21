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
            normalize_chat_error=lambda exc: SimpleNamespace(
                error_code="provider_request_failed", category="unknown", retryable=True
            ),
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
    async def test_replay_logs_shape(self, caplog: pytest.LogCaptureFixture):
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: "stop",
            finalize_stream_result=lambda **kwargs: ({}, "stop", {}),
            json_safe=lambda value: value,
            normalize_chat_error=lambda exc: SimpleNamespace(
                error_code="provider_request_failed", category="unknown", retryable=True
            ),
            normalize_usage_payload=lambda value: value,
            stage_span=_stage_span,
        )

        with caplog.at_level("INFO"):
            await streamer.stream_replay_chunks(
                replay_response=SimpleNamespace(
                    content="replay answer",
                    metadata={"reasoning_content": "replay reasoning"},
                ),
                assistant_message_id="msg-1",
                model="model-1",
                context_usage={"used_tokens": 10},
                finish_reason="stop",
            )

        assert any(
            "Chat replay stream:" in record.message
            and "message=msg-1" in record.message
            and "content_len=13" in record.message
            and "reasoning_len=16" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_replay_sanitizes_tool_markers(self):
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: "stop",
            finalize_stream_result=lambda **kwargs: ({}, "stop", {}),
            json_safe=lambda value: value,
            normalize_chat_error=lambda exc: SimpleNamespace(
                error_code="provider_request_failed", category="unknown", retryable=True
            ),
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
            normalize_chat_error=lambda exc: SimpleNamespace(
                error_code="provider_request_failed", category="unknown", retryable=True
            ),
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
            "final_stream_status": "empty_visible_output",
            "final_stream_empty_visible_output": True,
            "final_stream_message_count": 1,
            "final_stream_input_chars": 2,
            "final_stream_chunk_count": 3,
        }
        finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_name_payload_in_reasoning(self):
        async def fake_stream_chat(**kwargs):
            yield StreamChunk(
                content_delta="",
                reasoning_delta='I\'ll help you search first.\ntool:houyi_find_files&args:{"path":"/Users/von","pattern":"skill.md"}',
            )

        finalize = MagicMock(return_value=({"prompt_tokens": 3}, "stop", {"metric": 1}))
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: "stop",
            finalize_stream_result=finalize,
            json_safe=lambda value: value,
            normalize_chat_error=lambda exc: SimpleNamespace(
                error_code="provider_request_failed", category="unknown", retryable=True
            ),
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
            "final_stream_status": "empty_visible_output",
            "final_stream_empty_visible_output": True,
            "final_stream_message_count": 1,
            "final_stream_input_chars": 2,
            "final_stream_chunk_count": 3,
        }
        finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_final_stream_logs_completion(self, caplog: pytest.LogCaptureFixture):
        async def fake_stream_chat(**kwargs):
            _ = kwargs
            yield StreamChunk(reasoning_delta="step")
            yield StreamChunk(content_delta="answer")

        finalize = MagicMock(return_value=({"prompt_tokens": 3}, "stop", {"metric": 1}))
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: "stop",
            finalize_stream_result=finalize,
            json_safe=lambda value: value,
            normalize_chat_error=lambda exc: SimpleNamespace(
                error_code="provider_request_failed", category="unknown", retryable=True
            ),
            normalize_usage_payload=lambda value: value,
            stage_span=_stage_span,
        )
        adapter = SimpleNamespace(
            stream_chat=fake_stream_chat,
            last_usage={"prompt_tokens": 3},
            last_finish_reason="stop",
        )

        with caplog.at_level("INFO"):
            await streamer.stream_final_response(
                llm_adapter=adapter,
                llm_messages=[{"role": "user", "content": "hi"}],
                llm_kwargs={"temperature": 0.1},
                model="model-1",
                conversation_id="conv-1",
                assistant_message_id="msg-1",
                context_usage={"used_tokens": 10},
                chat_span=Span(name="chat.request"),
            )

        assert any(
            "Chat final stream complete:" in record.message
            and "conversation=conv-1" in record.message
            and "message=msg-1" in record.message
            and "chunks=4" in record.message
            and "errored=False" in record.message
            for record in caplog.records
        )

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
            normalize_chat_error=lambda exc: SimpleNamespace(
                error_code="provider_request_failed", category="unknown", retryable=True
            ),
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
            "final_stream_status": "empty_visible_output",
            "final_stream_empty_visible_output": True,
            "final_stream_message_count": 1,
            "final_stream_input_chars": 2,
            "final_stream_chunk_count": 3,
        }
        finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_final_stream_timeout_metadata(self):
        async def fake_stream_chat(**kwargs):
            _ = kwargs
            raise TimeoutError("request timed out")
            yield

        finalize = MagicMock(return_value=(None, None, {"metric": 1}))
        streamer = AssistantResponseStreamer(
            build_stream_error_content=lambda exc: f"visible: {exc}",
            build_public_stream_error_message=lambda exc: str(exc),
            build_empty_stream_content=lambda: "empty fallback",
            extract_finish_reason=lambda *args: None,
            finalize_stream_result=finalize,
            json_safe=lambda value: value,
            normalize_chat_error=lambda exc: SimpleNamespace(
                error_code="provider_timeout", category="timeout", retryable=True
            ),
            normalize_usage_payload=lambda value: value,
            stage_span=_stage_span,
        )
        adapter = SimpleNamespace(
            stream_chat=fake_stream_chat,
            last_usage=None,
            last_finish_reason=None,
        )

        (
            _chunks,
            content_parts,
            _reasoning_parts,
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

        assert content_parts == ["visible: request timed out"]
        assert usage_payload is None
        assert finish_reason == "error"
        assert metadata["metric"] == 1
        assert metadata["final_stream_status"] == "error"
        assert metadata["final_stream_error_code"] == "provider_timeout"
        assert metadata["final_stream_error_category"] == "timeout"
        assert metadata["final_stream_error_retryable"] is True
        assert metadata["final_stream_empty_visible_output"] is False
