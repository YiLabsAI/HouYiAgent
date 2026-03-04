"""Tests for span storage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from houyi.observability.storage import (
    SpanFilter,
    SpanStorageConfig,
    SQLiteSpanStorage,
    get_storage,
    reset_storage,
    set_storage,
)
from houyi.observability.types import (
    CostInfo,
    SpanSchema,
    SpanType,
    TokenUsage,
)


@pytest.fixture
def temp_db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_spans.db"
        yield db_path


@pytest.fixture
def storage(temp_db):
    """Create a storage instance with temp database."""
    config = SpanStorageConfig(db_path=temp_db)
    storage = SQLiteSpanStorage(config)
    yield storage
    storage.close()


@pytest.fixture
def sample_span():
    """Create a sample span."""
    return SpanSchema(
        span_id="span_001",
        trace_id="trace_001",
        parent_id=None,
        name="test_execution",
        span_type=SpanType.EXECUTION,
        status="ok",
        start_time=1000.0,
        end_time=1005.0,
        node_id="node_001",
        model="gpt-4",
        provider="openai",
        tokens=TokenUsage(input=100, output=50, total=150),
        cost=CostInfo(usd=0.005),
        cache_hit=False,
        attributes={"key": "value"},
    )


class TestSQLiteSpanStorage:
    """Tests for SQLiteSpanStorage."""

    def test_save_and_get(self, storage, sample_span):
        """Test saving and retrieving a span."""
        storage.save(sample_span)

        retrieved = storage.get(sample_span.span_id)
        assert retrieved is not None
        assert retrieved.span_id == sample_span.span_id
        assert retrieved.trace_id == sample_span.trace_id
        assert retrieved.name == sample_span.name
        assert retrieved.span_type == sample_span.span_type
        assert retrieved.start_time == sample_span.start_time
        assert retrieved.end_time == sample_span.end_time
        assert retrieved.model == sample_span.model
        assert retrieved.tokens.input == sample_span.tokens.input
        assert retrieved.tokens.output == sample_span.tokens.output
        assert retrieved.cost.usd == sample_span.cost.usd
        assert retrieved.cache_hit == sample_span.cache_hit
        assert retrieved.attributes == sample_span.attributes

    def test_get_nonexistent(self, storage):
        """Test getting a nonexistent span."""
        result = storage.get("nonexistent")
        assert result is None

    def test_save_batch(self, storage):
        """Test batch saving spans."""
        spans = [
            SpanSchema(
                span_id=f"span_{i}",
                trace_id="trace_batch",
                name=f"span_{i}",
                span_type=SpanType.NODE,
                start_time=1000.0 + i,
            )
            for i in range(10)
        ]

        storage.save_batch(spans)

        for span in spans:
            retrieved = storage.get(span.span_id)
            assert retrieved is not None
            assert retrieved.span_id == span.span_id

    def test_save_batch_with_non_serializable_attributes(self, storage):
        """Non-JSON-serializable attributes/events should not break persistence."""

        def _callable_marker():
            return "ok"

        span = SpanSchema(
            span_id="span_non_serializable",
            trace_id="trace_non_serializable",
            name="span_non_serializable",
            span_type=SpanType.NODE,
            start_time=1000.0,
            attributes={"callable": _callable_marker, "path": Path("/tmp/demo")},
        )

        storage.save_batch([span])

        retrieved = storage.get("span_non_serializable")
        assert retrieved is not None
        assert retrieved.trace_id == "trace_non_serializable"

    def test_query_by_trace_id(self, storage):
        """Test querying by trace_id."""
        spans = [
            SpanSchema(
                span_id=f"span_{i}",
                trace_id="trace_query",
                name=f"span_{i}",
                span_type=SpanType.NODE,
                start_time=1000.0 + i,
            )
            for i in range(5)
        ]
        storage.save_batch(spans)

        # Add span with different trace
        storage.save(
            SpanSchema(
                span_id="other_span",
                trace_id="other_trace",
                name="other",
                span_type=SpanType.NODE,
                start_time=1000.0,
            )
        )

        results = storage.query(SpanFilter(trace_id="trace_query"))
        assert len(results) == 5
        assert all(s.trace_id == "trace_query" for s in results)

    def test_query_by_span_type(self, storage):
        """Test querying by span_type."""
        storage.save_batch(
            [
                SpanSchema(
                    span_id="exec_1",
                    trace_id="trace_1",
                    name="exec",
                    span_type=SpanType.EXECUTION,
                    start_time=1000.0,
                ),
                SpanSchema(
                    span_id="node_1",
                    trace_id="trace_1",
                    name="node",
                    span_type=SpanType.NODE,
                    start_time=1001.0,
                ),
                SpanSchema(
                    span_id="llm_1",
                    trace_id="trace_1",
                    name="llm",
                    span_type=SpanType.LLM,
                    start_time=1002.0,
                ),
            ]
        )

        llm_spans = storage.query(SpanFilter(span_type=SpanType.LLM))
        assert len(llm_spans) == 1
        assert llm_spans[0].span_type == SpanType.LLM

    def test_query_by_time_range(self, storage):
        """Test querying by time range."""
        for i in range(10):
            storage.save(
                SpanSchema(
                    span_id=f"span_{i}",
                    trace_id="trace_time",
                    name=f"span_{i}",
                    span_type=SpanType.NODE,
                    start_time=1000.0 + i * 10,
                )
            )

        results = storage.query(SpanFilter(start_time_gte=1020.0, start_time_lte=1060.0))
        assert len(results) == 5  # spans 2, 3, 4, 5, 6

    def test_get_trace(self, storage):
        """Test getting all spans for a trace."""
        # Create hierarchical spans
        storage.save_batch(
            [
                SpanSchema(
                    span_id="exec",
                    trace_id="trace_hier",
                    name="execution",
                    span_type=SpanType.EXECUTION,
                    start_time=1000.0,
                ),
                SpanSchema(
                    span_id="node_1",
                    trace_id="trace_hier",
                    parent_id="exec",
                    name="node_1",
                    span_type=SpanType.NODE,
                    start_time=1001.0,
                ),
                SpanSchema(
                    span_id="llm_1",
                    trace_id="trace_hier",
                    parent_id="node_1",
                    name="llm_1",
                    span_type=SpanType.LLM,
                    start_time=1002.0,
                ),
            ]
        )

        trace = storage.get_trace("trace_hier")
        assert len(trace) == 3
        assert trace[0].span_id == "exec"  # Ordered by start_time

    def test_delete_trace(self, storage):
        """Test deleting a trace."""
        storage.save_batch(
            [
                SpanSchema(
                    span_id=f"span_{i}",
                    trace_id="trace_delete",
                    name=f"span_{i}",
                    span_type=SpanType.NODE,
                    start_time=1000.0 + i,
                )
                for i in range(5)
            ]
        )

        count = storage.delete_trace("trace_delete")
        assert count == 5

        results = storage.get_trace("trace_delete")
        assert len(results) == 0

    def test_cleanup_old_spans(self, storage):
        """Test cleaning up old spans."""
        # Old spans
        for i in range(5):
            storage.save(
                SpanSchema(
                    span_id=f"old_{i}",
                    trace_id="trace_old",
                    name=f"old_{i}",
                    span_type=SpanType.NODE,
                    start_time=100.0 + i,
                )
            )

        # New spans
        for i in range(5):
            storage.save(
                SpanSchema(
                    span_id=f"new_{i}",
                    trace_id="trace_new",
                    name=f"new_{i}",
                    span_type=SpanType.NODE,
                    start_time=1000.0 + i,
                )
            )

        count = storage.cleanup_old_spans(500.0)
        assert count == 5

        old_trace = storage.get_trace("trace_old")
        assert len(old_trace) == 0

        new_trace = storage.get_trace("trace_new")
        assert len(new_trace) == 5

    def test_get_statistics(self, storage):
        """Test getting storage statistics."""
        storage.save_batch(
            [
                SpanSchema(
                    span_id="exec_1",
                    trace_id="trace_1",
                    name="exec",
                    span_type=SpanType.EXECUTION,
                    start_time=1000.0,
                ),
                SpanSchema(
                    span_id="node_1",
                    trace_id="trace_1",
                    name="node",
                    span_type=SpanType.NODE,
                    start_time=1001.0,
                ),
                SpanSchema(
                    span_id="llm_1",
                    trace_id="trace_2",
                    name="llm",
                    span_type=SpanType.LLM,
                    start_time=1002.0,
                ),
            ]
        )

        stats = storage.get_statistics()
        assert stats["total_spans"] == 3
        assert stats["total_traces"] == 2
        assert stats["spans_by_type"]["execution"] == 1
        assert stats["spans_by_type"]["node"] == 1
        assert stats["spans_by_type"]["llm"] == 1

    def test_upsert_span(self, storage, sample_span):
        """Test upserting (update on conflict) a span."""
        storage.save(sample_span)

        # Update the span
        updated_span = SpanSchema(
            span_id=sample_span.span_id,
            trace_id=sample_span.trace_id,
            name="updated_name",
            span_type=sample_span.span_type,
            status="error",
            start_time=sample_span.start_time,
            end_time=2000.0,
        )
        storage.save(updated_span)

        retrieved = storage.get(sample_span.span_id)
        assert retrieved.name == "updated_name"
        assert retrieved.status == "error"
        assert retrieved.end_time == 2000.0

    def test_parallel_fields(self, storage):
        """Test parallel execution fields."""
        span = SpanSchema(
            span_id="parallel_1",
            trace_id="trace_parallel",
            name="parallel_node",
            span_type=SpanType.NODE,
            start_time=1000.0,
            group_id="group_001",
            lane_id=2,
            seq=3,
        )
        storage.save(span)

        retrieved = storage.get("parallel_1")
        assert retrieved.group_id == "group_001"
        assert retrieved.lane_id == 2
        assert retrieved.seq == 3

    def test_checkpoint_lineage_fields(self, storage):
        """Test checkpoint lineage fields."""
        span = SpanSchema(
            span_id="restored_1",
            trace_id="trace_restored",
            name="restored_execution",
            span_type=SpanType.EXECUTION,
            start_time=1000.0,
            parent_trace_id="original_trace",
            restore_checkpoint_id="checkpoint_001",
            replay_mode=True,
        )
        storage.save(span)

        retrieved = storage.get("restored_1")
        assert retrieved.parent_trace_id == "original_trace"
        assert retrieved.restore_checkpoint_id == "checkpoint_001"
        assert retrieved.replay_mode is True


class TestGlobalStorage:
    """Tests for global storage functions."""

    def test_get_set_reset_storage(self, temp_db):
        """Test global storage management."""
        reset_storage()

        # Get default storage
        storage1 = get_storage()
        assert storage1 is not None

        # Set custom storage
        config = SpanStorageConfig(db_path=temp_db)
        custom_storage = SQLiteSpanStorage(config)
        set_storage(custom_storage)

        storage2 = get_storage()
        assert storage2 is custom_storage

        # Reset
        reset_storage()
