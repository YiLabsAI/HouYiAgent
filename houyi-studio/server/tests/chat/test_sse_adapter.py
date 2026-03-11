"""Unit tests for houyi_studio.server.chat.sse_adapter."""

from __future__ import annotations

import asyncio
import json

import pytest
from houyi_studio.server.chat.sse_adapter import SSEEvent, stream_chat_sse


class TestSSEEvent:
    """Test SSEEvent encoding."""

    def test_encode_basic(self):
        evt = SSEEvent(event="message.delta", data={"content": "hello"})
        encoded = evt.encode()
        lines = encoded.strip().split("\n")
        assert lines[0] == "event: message.delta"
        assert lines[1].startswith("data: ")
        parsed = json.loads(lines[1][6:])
        assert parsed["content"] == "hello"

    def test_encode_with_id(self):
        evt = SSEEvent(event="message.delta", data={"seq": 1}, event_id="msg-1")
        encoded = evt.encode()
        assert "id: msg-1" in encoded
        assert "event: message.delta" in encoded

    def test_encode_unicode(self):
        evt = SSEEvent(event="message.delta", data={"content": "hello world, special chars: @#$%"})
        encoded = evt.encode()
        parsed = json.loads(encoded.split("data: ")[1].split("\n")[0])
        assert parsed["content"] == "hello world, special chars: @#$%"

    def test_encode_empty_data(self):
        evt = SSEEvent(event="test", data={})
        encoded = evt.encode()
        assert "data: {}" in encoded


class TestStreamChatSSE:
    """Test stream_chat_sse async generator."""

    @pytest.mark.asyncio
    async def test_normal_stream(self):
        """Normal flow: deltas → finish."""

        async def mock_llm():
            yield ("Hello", None)
            yield (" world", None)
            yield ("!", None)

        events = []
        async for chunk in stream_chat_sse(mock_llm(), message_id="msg001", model="test"):
            events.append(chunk)

        # Parse events
        parsed = _parse_sse_events(events)
        types = [e["event"] for e in parsed]

        assert types.count("message.delta") == 3
        assert types[-1] == "message.finish"

        # Verify seq increments
        deltas = [e for e in parsed if e["event"] == "message.delta"]
        seqs = [e["data"]["seq"] for e in deltas]
        assert seqs == [1, 2, 3]

        # Verify content in deltas
        assert deltas[0]["data"]["content"] == "Hello"
        assert deltas[1]["data"]["content"] == " world"
        assert deltas[2]["data"]["content"] == "!"

        # Verify finish event
        finish = parsed[-1]
        assert finish["data"]["model"] == "test"
        assert finish["data"]["finish_reason"] == "stop"
        assert finish["data"]["total_chunks"] == 3

    @pytest.mark.asyncio
    async def test_stream_with_reasoning(self):
        """Stream with both content and reasoning."""

        async def mock_llm():
            yield ("Hi", "thinking...")
            yield (" there", " more thinking")

        events = []
        async for chunk in stream_chat_sse(mock_llm(), message_id="msg002"):
            events.append(chunk)

        parsed = _parse_sse_events(events)
        deltas = [e for e in parsed if e["event"] == "message.delta"]
        assert deltas[0]["data"]["content"] == "Hi"
        assert deltas[0]["data"]["reasoning_content"] == "thinking..."
        assert deltas[1]["data"]["reasoning_content"] == " more thinking"

        finish = next(e for e in parsed if e["event"] == "message.finish")
        assert finish["data"]["reasoning_length"] > 0

    @pytest.mark.asyncio
    async def test_stream_with_context_usage(self):
        """Context usage event sent before deltas."""

        async def mock_llm():
            yield ("ok", None)

        usage = {"model": "test", "used_tokens": 100, "max_context_tokens": 1000}
        events = []
        async for chunk in stream_chat_sse(mock_llm(), message_id="msg003", context_usage=usage):
            events.append(chunk)

        parsed = _parse_sse_events(events)
        assert parsed[0]["event"] == "context.usage"
        assert parsed[0]["data"]["usage"]["used_tokens"] == 100

    @pytest.mark.asyncio
    async def test_stream_error(self):
        """LLM raises exception → message.error event."""

        async def mock_llm():
            yield ("partial", None)
            raise RuntimeError("LLM connection failed")

        events = []
        async for chunk in stream_chat_sse(mock_llm(), message_id="msg004"):
            events.append(chunk)

        parsed = _parse_sse_events(events)
        types = [e["event"] for e in parsed]
        assert "message.delta" in types
        assert "message.error" in types

        error_evt = next(e for e in parsed if e["event"] == "message.error")
        assert "LLM connection failed" in error_evt["data"]["error"]
        assert error_evt["data"]["error_type"] == "RuntimeError"
        assert error_evt["data"]["chunks_sent"] == 1

    @pytest.mark.asyncio
    async def test_stream_error_can_emit_visible_fallback_text(self):
        async def mock_llm():
            raise RuntimeError("LLM connection failed")
            yield

        events = []
        async for chunk in stream_chat_sse(
            mock_llm(),
            message_id="msg004b",
            error_message_builder=lambda exc: f"visible: {exc}",
        ):
            events.append(chunk)

        parsed = _parse_sse_events(events)
        assert parsed[0]["event"] == "message.delta"
        assert parsed[0]["data"]["content"] == "visible: LLM connection failed"
        assert parsed[1]["event"] == "message.error"

    @pytest.mark.asyncio
    async def test_stream_abort(self):
        """CancelledError → message.aborted event."""

        async def mock_llm():
            yield ("partial", None)
            raise asyncio.CancelledError()

        events = []
        with pytest.raises(asyncio.CancelledError):
            async for chunk in stream_chat_sse(mock_llm(), message_id="msg005"):
                events.append(chunk)

        parsed = _parse_sse_events(events)
        types = [e["event"] for e in parsed]
        assert "message.delta" in types
        assert "message.aborted" in types

        aborted = next(e for e in parsed if e["event"] == "message.aborted")
        assert aborted["data"]["reason"] == "user_abort"
        assert aborted["data"]["chunks_sent"] == 1

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Empty LLM stream → just finish event."""

        async def mock_llm():
            return
            yield  # make it an async generator

        events = []
        async for chunk in stream_chat_sse(mock_llm(), message_id="msg006"):
            events.append(chunk)

        parsed = _parse_sse_events(events)
        assert len(parsed) == 1
        assert parsed[0]["event"] == "message.finish"
        assert parsed[0]["data"]["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_finish_reason_can_be_provided_explicitly(self):
        """Finish event uses explicit provider finish reason when supplied."""

        async def mock_llm():
            yield ("cut", None)

        events = []
        async for chunk in stream_chat_sse(
            mock_llm(),
            message_id="msg007",
            finish_reason="length",
        ):
            events.append(chunk)

        parsed = _parse_sse_events(events)
        finish = next(e for e in parsed if e["event"] == "message.finish")
        assert finish["data"]["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_finish_reason_can_be_resolved_lazily(self):
        """Finish event resolves finish reason after the stream completes."""

        state = {"finish_reason": None}

        async def mock_llm():
            yield ("partial", None)
            state["finish_reason"] = "tool_calls"

        events = []
        async for chunk in stream_chat_sse(
            mock_llm(),
            message_id="msg008",
            finish_reason=lambda: state["finish_reason"],
        ):
            events.append(chunk)

        parsed = _parse_sse_events(events)
        finish = next(e for e in parsed if e["event"] == "message.finish")
        assert finish["data"]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_event_ids_include_message_id(self):
        """Event IDs follow {message_id}-{seq} pattern."""

        async def mock_llm():
            yield ("a", None)
            yield ("b", None)

        events = []
        async for chunk in stream_chat_sse(mock_llm(), message_id="m1"):
            events.append(chunk)

        parsed = _parse_sse_events(events)
        deltas = [e for e in parsed if e["event"] == "message.delta"]
        assert deltas[0].get("id") == "m1-1"
        assert deltas[1].get("id") == "m1-2"


# --- Helper ---


def _parse_sse_events(raw_chunks: list[str]) -> list[dict]:
    """Parse raw SSE chunks into structured events."""
    events = []
    for chunk in raw_chunks:
        lines = chunk.strip().split("\n")
        evt: dict = {}
        for line in lines:
            if line.startswith("id: "):
                evt["id"] = line[4:]
            elif line.startswith("event: "):
                evt["event"] = line[7:]
            elif line.startswith("data: "):
                evt["data"] = json.loads(line[6:])
        if "event" in evt:
            events.append(evt)
    return events


class TestStreamingOutputEventMetadata:
    """Test StreamingOutputEvent.metadata field (SSE extension)."""

    def test_metadata_field_accepted(self):
        from houyi_studio.server.gateway.events import StreamingOutputEvent

        event = StreamingOutputEvent(
            event_id="e1",
            session_id="s1",
            execution_id="x1",
            node_id="n1",
            chunk="",
            is_final=True,
            metadata={"trace_id": "t1", "usage": {"total_tokens": 100}},
        )
        assert event.metadata is not None
        assert event.metadata["trace_id"] == "t1"
        assert event.metadata["usage"]["total_tokens"] == 100

    def test_metadata_defaults_to_none(self):
        from houyi_studio.server.gateway.events import StreamingOutputEvent

        event = StreamingOutputEvent(
            event_id="e1",
            session_id="s1",
            execution_id="x1",
            node_id="n1",
            chunk="hello",
            is_final=False,
        )
        assert event.metadata is None
