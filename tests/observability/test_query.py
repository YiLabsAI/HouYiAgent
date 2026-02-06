"""Tests for observability query interface."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from houyi.observability.content_store import (
    ContentStoreConfig,
    ContentType,
    FileContentStore,
)
from houyi.observability.query import (
    ObservabilityQuery,
    QueryResult,
    TraceView,
    get_query,
    reset_query,
)
from houyi.observability.storage import (
    SpanStorageConfig,
    SQLiteSpanStorage,
)
from houyi.observability.types import (
    CostInfo,
    SpanSchema,
    SpanType,
    TokenUsage,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def span_storage(temp_dir):
    """Create a span storage instance."""
    config = SpanStorageConfig(db_path=temp_dir / "spans.db")
    storage = SQLiteSpanStorage(config)
    yield storage
    storage.close()


@pytest.fixture
def content_store(temp_dir):
    """Create a content store instance."""
    config = ContentStoreConfig(base_path=temp_dir / "content")
    return FileContentStore(config)


@pytest.fixture
def query(span_storage, content_store):
    """Create a query instance."""
    return ObservabilityQuery(
        span_storage=span_storage,
        content_store=content_store,
    )


@pytest.fixture
def sample_trace(span_storage, content_store):
    """Create a sample trace with spans and content."""
    trace_id = "trace_sample"
    now = datetime.now(timezone.utc).timestamp()

    # Create hierarchical spans
    spans = [
        SpanSchema(
            span_id="exec_1",
            trace_id=trace_id,
            name="execution",
            span_type=SpanType.EXECUTION,
            status="ok",
            start_time=now,
            end_time=now + 5.0,
        ),
        SpanSchema(
            span_id="node_1",
            trace_id=trace_id,
            parent_id="exec_1",
            name="node.llm",
            span_type=SpanType.NODE,
            node_id="llm_node",
            status="ok",
            start_time=now + 0.1,
            end_time=now + 2.0,
        ),
        SpanSchema(
            span_id="llm_1",
            trace_id=trace_id,
            parent_id="node_1",
            name="llm.completion",
            span_type=SpanType.LLM,
            node_id="llm_node",
            model="gpt-4",
            provider="openai",
            tokens=TokenUsage(input=100, output=50, total=150),
            cost=CostInfo(usd=0.005),
            status="ok",
            start_time=now + 0.2,
            end_time=now + 1.8,
        ),
        SpanSchema(
            span_id="node_2",
            trace_id=trace_id,
            parent_id="exec_1",
            name="node.tool",
            span_type=SpanType.NODE,
            node_id="tool_node",
            status="ok",
            start_time=now + 2.1,
            end_time=now + 3.5,
        ),
        SpanSchema(
            span_id="tool_1",
            trace_id=trace_id,
            parent_id="node_2",
            name="tool.search",
            span_type=SpanType.TOOL,
            node_id="tool_node",
            tool_name="web_search",
            status="ok",
            start_time=now + 2.2,
            end_time=now + 3.4,
        ),
        SpanSchema(
            span_id="node_3",
            trace_id=trace_id,
            parent_id="exec_1",
            name="node.error",
            span_type=SpanType.NODE,
            node_id="error_node",
            status="error",
            start_time=now + 3.6,
            end_time=now + 4.0,
        ),
    ]

    span_storage.save_batch(spans)

    # Store some content
    content_store.store(
        content="What is the weather?",
        content_type=ContentType.LLM_PROMPT,
        span_id="llm_1",
        trace_id=trace_id,
    )
    content_store.store(
        content="The weather is sunny.",
        content_type=ContentType.LLM_RESPONSE,
        span_id="llm_1",
        trace_id=trace_id,
    )

    return trace_id


class TestTraceView:
    """Tests for TraceView."""

    def test_from_spans(self):
        """Test creating TraceView from spans."""
        spans = [
            SpanSchema(
                span_id="root",
                trace_id="trace_1",
                name="execution",
                span_type=SpanType.EXECUTION,
                start_time=1000.0,
                end_time=1005.0,
            ),
            SpanSchema(
                span_id="llm_1",
                trace_id="trace_1",
                parent_id="root",
                name="llm",
                span_type=SpanType.LLM,
                tokens=TokenUsage(input=100, output=50, total=150),
                cost=CostInfo(usd=0.01),
                start_time=1001.0,
                end_time=1003.0,
            ),
            SpanSchema(
                span_id="tool_1",
                trace_id="trace_1",
                parent_id="root",
                name="tool",
                span_type=SpanType.TOOL,
                status="error",
                start_time=1003.0,
                end_time=1004.0,
            ),
        ]

        view = TraceView.from_spans("trace_1", spans)

        assert view.trace_id == "trace_1"
        assert view.span_count == 3
        assert view.root_span is not None
        assert view.root_span.span_id == "root"
        assert view.total_duration_ms == 5000.0
        assert view.llm_call_count == 1
        assert view.tool_call_count == 1
        assert view.error_count == 1
        assert view.total_tokens == 150
        assert view.total_cost_usd == 0.01


class TestObservabilityQuery:
    """Tests for ObservabilityQuery."""

    def test_get_trace(self, query, sample_trace):
        """Test getting a complete trace."""
        trace_view = query.get_trace(sample_trace)

        assert trace_view is not None
        assert trace_view.trace_id == sample_trace
        assert trace_view.span_count == 6
        assert trace_view.root_span is not None
        assert trace_view.root_span.span_id == "exec_1"
        assert trace_view.llm_call_count == 1
        assert trace_view.tool_call_count == 1
        assert trace_view.error_count == 1

    def test_get_trace_with_content(self, query, sample_trace):
        """Test getting trace with content references."""
        trace_view = query.get_trace(sample_trace, include_content=True)

        assert trace_view is not None
        assert len(trace_view.content_refs) == 2

    def test_get_trace_nonexistent(self, query):
        """Test getting nonexistent trace."""
        result = query.get_trace("nonexistent")
        assert result is None

    def test_get_span(self, query, sample_trace):
        """Test getting a single span."""
        result = query.get_span("llm_1")

        assert result is not None
        assert result.span.span_id == "llm_1"
        assert result.span.model == "gpt-4"

    def test_get_span_with_content(self, query, sample_trace):
        """Test getting span with content."""
        result = query.get_span("llm_1", include_content=True)

        assert result is not None
        assert ContentType.LLM_PROMPT in result.content
        assert result.content[ContentType.LLM_PROMPT] == "What is the weather?"
        assert ContentType.LLM_RESPONSE in result.content
        assert result.content[ContentType.LLM_RESPONSE] == "The weather is sunny."

    def test_get_span_nonexistent(self, query):
        """Test getting nonexistent span."""
        result = query.get_span("nonexistent")
        assert result is None

    def test_query_spans_by_trace(self, query, sample_trace):
        """Test querying spans by trace ID."""
        result = query.query_spans(trace_id=sample_trace)

        assert isinstance(result, QueryResult)
        assert len(result.spans) == 6

    def test_query_spans_by_type(self, query, sample_trace):
        """Test querying spans by type."""
        result = query.query_spans(
            trace_id=sample_trace,
            span_type=SpanType.LLM,
        )

        assert len(result.spans) == 1
        assert result.spans[0].span_type == SpanType.LLM

    def test_query_spans_by_status(self, query, sample_trace):
        """Test querying spans by status."""
        result = query.query_spans(
            trace_id=sample_trace,
            status="error",
        )

        assert len(result.spans) == 1
        assert result.spans[0].status == "error"

    def test_query_spans_pagination(self, query, span_storage):
        """Test query pagination."""
        trace_id = "trace_pagination"
        now = datetime.now(timezone.utc).timestamp()

        # Create many spans
        spans = [
            SpanSchema(
                span_id=f"span_{i}",
                trace_id=trace_id,
                name=f"span_{i}",
                span_type=SpanType.NODE,
                start_time=now + i,
            )
            for i in range(20)
        ]
        span_storage.save_batch(spans)

        # Query with limit
        result1 = query.query_spans(trace_id=trace_id, limit=10)
        assert len(result1.spans) == 10
        assert result1.has_more is True

        # Query with offset
        result2 = query.query_spans(trace_id=trace_id, limit=10, offset=10)
        assert len(result2.spans) == 10
        assert result2.has_more is False

    def test_get_error_spans(self, query, sample_trace):
        """Test getting error spans."""
        errors = query.get_error_spans(trace_id=sample_trace)

        assert len(errors) == 1
        assert errors[0].status == "error"

    def test_get_llm_spans(self, query, sample_trace):
        """Test getting LLM spans."""
        llm_spans = query.get_llm_spans(trace_id=sample_trace)

        assert len(llm_spans) == 1
        assert llm_spans[0].span_type == SpanType.LLM
        assert llm_spans[0].model == "gpt-4"

    def test_get_llm_spans_by_model(self, query, sample_trace):
        """Test getting LLM spans filtered by model."""
        # Filter by existing model
        spans = query.get_llm_spans(trace_id=sample_trace, model="gpt-4")
        assert len(spans) == 1

        # Filter by non-existing model
        spans = query.get_llm_spans(trace_id=sample_trace, model="claude-3")
        assert len(spans) == 0

    def test_get_tool_spans(self, query, sample_trace):
        """Test getting tool spans."""
        tool_spans = query.get_tool_spans(trace_id=sample_trace)

        assert len(tool_spans) == 1
        assert tool_spans[0].span_type == SpanType.TOOL
        assert tool_spans[0].tool_name == "web_search"

    def test_get_tool_spans_by_name(self, query, sample_trace):
        """Test getting tool spans filtered by name."""
        spans = query.get_tool_spans(trace_id=sample_trace, tool_name="web_search")
        assert len(spans) == 1

        spans = query.get_tool_spans(trace_id=sample_trace, tool_name="other_tool")
        assert len(spans) == 0

    def test_get_node_spans(self, query, sample_trace):
        """Test getting node spans."""
        node_spans = query.get_node_spans(sample_trace)

        assert len(node_spans) == 3  # node_1, node_2, node_3

    def test_get_node_spans_by_id(self, query, sample_trace):
        """Test getting node spans by node ID."""
        spans = query.get_node_spans(sample_trace, node_id="llm_node")

        assert len(spans) == 1
        assert spans[0].node_id == "llm_node"

    def test_get_span_children(self, query, sample_trace):
        """Test getting child spans."""
        children = query.get_span_children("exec_1")

        assert len(children) == 3  # node_1, node_2, node_3

    def test_get_span_children_nonexistent(self, query):
        """Test getting children of nonexistent span."""
        children = query.get_span_children("nonexistent")
        assert len(children) == 0

    def test_get_trace_statistics(self, query, sample_trace):
        """Test getting trace statistics."""
        stats = query.get_trace_statistics(sample_trace)

        assert stats["trace_id"] == sample_trace
        assert stats["span_count"] == 6
        assert stats["error_count"] == 1
        assert stats["llm_call_count"] == 1
        assert stats["tool_call_count"] == 1
        assert stats["total_tokens"] == 150
        assert stats["total_cost_usd"] == 0.005
        assert stats["content_count"] == 2
        assert "execution" in stats["spans_by_type"]
        assert "node" in stats["spans_by_type"]

    def test_get_trace_statistics_nonexistent(self, query):
        """Test getting statistics for nonexistent trace."""
        stats = query.get_trace_statistics("nonexistent")
        assert stats == {}


class TestGlobalQuery:
    """Tests for global query functions."""

    def test_get_reset_query(self):
        """Test global query management."""
        reset_query()

        query1 = get_query()
        assert query1 is not None

        query2 = get_query()
        assert query2 is query1

        reset_query()
