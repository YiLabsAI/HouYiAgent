import pytest

from houyi.adapters.llm.base import StreamChunk
from houyi.rag.generation.streaming import (
    StreamEvent,
    StreamEventType,
    StreamingAnswerGenerator,
    build_non_streaming_events,
    stream_sse,
)
from houyi.rag.generation.streaming_helpers import (
    RAG_ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    build_stream_sources,
    estimate_stream_confidence,
    format_stream_context,
)
from houyi.rag.types import SearchResult, Source


class TestStreamEvent:
    """Tests for StreamEvent."""

    def test_stream_event_to_sse_string(self) -> None:
        """Test SSE formatting with string data."""
        event = StreamEvent(
            event_type=StreamEventType.CHUNK,
            data="Hello world",
        )
        sse = event.to_sse()
        assert "event: chunk" in sse
        assert "data: Hello world" in sse

    def test_stream_event_to_sse_dict(self) -> None:
        """Test SSE formatting with dict data."""
        event = StreamEvent(
            event_type=StreamEventType.START,
            data={"sources": ["file1.md", "file2.md"]},
        )
        sse = event.to_sse()
        assert "event: start" in sse
        assert "sources" in sse
        assert "file1.md" in sse

    def test_stream_event_to_dict(self) -> None:
        """Test to_dict conversion."""
        event = StreamEvent(
            event_type=StreamEventType.END,
            data={"confidence": 0.8},
            metadata={"test": "value"},
        )
        result = event.to_dict()
        assert result["event"] == "end"
        assert result["data"]["confidence"] == 0.8
        assert result["metadata"]["test"] == "value"

    def test_stream_event_types(self) -> None:
        """Test all event types."""
        assert StreamEventType.START.value == "start"
        assert StreamEventType.CHUNK.value == "chunk"
        assert StreamEventType.SOURCE.value == "source"
        assert StreamEventType.METADATA.value == "metadata"
        assert StreamEventType.ERROR.value == "error"
        assert StreamEventType.END.value == "end"


class TestStreamingAnswerGenerator:
    """Tests for StreamingAnswerGenerator."""

    def test_build_stream_sources(self) -> None:
        """Test stream source payload construction."""
        results = [
            SearchResult(
                chunk_id="c1",
                content="Test content about the topic.",
                score=0.9,
                source=Source(file_path="/docs/test.md"),
            ),
        ]

        sources = build_stream_sources(results)

        assert len(sources) == 1
        assert sources[0]["index"] == 1
        assert sources[0]["file_path"] == "/docs/test.md"
        assert "Test content" in str(sources[0]["snippet"])

    def test_format_stream_context(self) -> None:
        """Test stream context formatting."""
        results = [
            SearchResult(
                chunk_id="c1",
                content="First result",
                score=0.9,
                source=Source(file_path="/docs/test.md"),
            ),
        ]

        context = format_stream_context(results)

        assert "[1]" in context
        assert "/docs/test.md" in context
        assert "First result" in context

    def test_build_answer_prompt(self) -> None:
        """Test shared answer prompt formatting."""
        results = [
            SearchResult(
                chunk_id="c1",
                content="First result",
                score=0.9,
                source=Source(file_path="/docs/test.md"),
            ),
        ]

        prompt = build_answer_prompt("What is the topic?", results)

        assert prompt.startswith("Context:\n[1]")
        assert "Question: What is the topic?" in prompt
        assert "/docs/test.md" in prompt

    def test_shared_answer_system_prompt(self) -> None:
        """Test shared answer system prompt content."""
        assert "Answer ONLY based on the provided context" in RAG_ANSWER_SYSTEM_PROMPT
        assert "Cite sources using [1], [2], etc. markers" in RAG_ANSWER_SYSTEM_PROMPT

    def test_build_non_streaming_events(self) -> None:
        """Test fallback stream event construction from a retrieval result."""
        from houyi.rag.types import RetrievalResult

        result = RetrievalResult(
            answer="Answer text",
            sources=[Source(file_path="/docs/test.md")],
            confidence=0.8,
        )

        events = build_non_streaming_events(result)

        assert [event.event_type for event in events] == [
            StreamEventType.START,
            StreamEventType.CHUNK,
            StreamEventType.END,
        ]
        assert events[0].data == {"sources": ["/docs/test.md"]}
        assert events[1].data == "Answer text"
        assert events[2].data == {"confidence": 0.8}

    def test_estimate_stream_confidence(self) -> None:
        """Test stream confidence heuristic."""
        results = [
            SearchResult(chunk_id="c1", content="First result", score=0.9),
            SearchResult(chunk_id="c2", content="Second result", score=0.8),
        ]

        confidence = estimate_stream_confidence("Answer with [1] citation.", results)

        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_stream_generate_empty_results(self) -> None:
        """Test streaming with no results."""
        from typing import Any

        class MockAdapter:
            async def stream_chat(self, messages: list[Any], **kwargs: Any):
                yield "No data"

        generator = StreamingAnswerGenerator(adapter=MockAdapter())  # type: ignore[arg-type]
        events = []
        async for event in generator.stream_generate("test query", []):
            events.append(event)

        assert len(events) == 3  # START, CHUNK, END
        assert events[0].event_type == StreamEventType.START
        assert events[1].event_type == StreamEventType.CHUNK
        assert events[2].event_type == StreamEventType.END
        assert events[2].data["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_stream_generate_with_results(self) -> None:
        """Test streaming with search results."""
        from typing import Any

        class MockAdapter:
            async def stream_chat(self, messages: list[Any], **kwargs: Any):
                yield StreamChunk(content_delta="The answer ")
                yield StreamChunk(content_delta="is based on ")
                yield StreamChunk(content_delta="[1] source.")

        generator = StreamingAnswerGenerator(adapter=MockAdapter())  # type: ignore[arg-type]

        results = [
            SearchResult(
                chunk_id="c1",
                content="Test content about the topic.",
                score=0.9,
                source=Source(file_path="/docs/test.md"),
            ),
        ]

        events = []
        async for event in generator.stream_generate("What is the topic?", results):
            events.append(event)

        # Should have START, multiple CHUNKs, and END
        assert events[0].event_type == StreamEventType.START
        assert "sources" in events[0].data

        chunk_events = [e for e in events if e.event_type == StreamEventType.CHUNK]
        assert len(chunk_events) == 3

        assert events[-1].event_type == StreamEventType.END
        assert "confidence" in events[-1].data

    @pytest.mark.asyncio
    async def test_stream_generate_error_handling(self) -> None:
        """Test streaming handles errors gracefully."""
        from typing import Any

        class FailingAdapter:
            async def stream_chat(self, messages: list[Any], **kwargs: Any):
                yield StreamChunk(content_delta="Starting...")
                raise RuntimeError("Stream failed")

        generator = StreamingAnswerGenerator(adapter=FailingAdapter())  # type: ignore[arg-type]

        results = [
            SearchResult(chunk_id="c1", content="Test", score=0.9),
        ]

        events = []
        async for event in generator.stream_generate("query", results):
            events.append(event)

        # Should have START, CHUNK, ERROR, END
        event_types = [e.event_type for e in events]
        assert StreamEventType.START in event_types
        assert StreamEventType.ERROR in event_types
        assert StreamEventType.END in event_types


class TestStreamSSE:
    """Tests for stream_sse helper."""

    @pytest.mark.asyncio
    async def test_stream_sse(self) -> None:
        """Test SSE stream conversion."""

        async def mock_events():
            yield StreamEvent(StreamEventType.START, {"test": True})
            yield StreamEvent(StreamEventType.CHUNK, "Hello")
            yield StreamEvent(StreamEventType.END, {"done": True})

        sse_strings = []
        async for sse in stream_sse(mock_events()):
            sse_strings.append(sse)

        assert len(sse_strings) == 3
        assert all("event:" in s for s in sse_strings)
        assert all("data:" in s for s in sse_strings)


class TestRAGStreamQuery:
    """Tests for RAG.stream_query."""

    @pytest.mark.asyncio
    async def test_stream_query_agentic_fallback(self) -> None:
        """Test stream_query falls back for agentic mode."""
        import tempfile
        from pathlib import Path

        from houyi.rag import RAG

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()
            (kb_dir / "doc.md").write_text("Test content about RAG.")

            rag = RAG(str(kb_dir), mode="agentic")

            events = []
            async for event in rag.stream_query("What is RAG?"):
                events.append(event)

            # Should have START, CHUNK, END
            event_types = [e.event_type for e in events]
            assert StreamEventType.START in event_types
            assert StreamEventType.CHUNK in event_types
            assert StreamEventType.END in event_types
