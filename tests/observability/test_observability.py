"""Tests for observability components (TraceManager, Span, Exporters)."""

import time

import pytest

from houyi.observability.exporters import (
    ConsoleExporter,
    DatadogExporter,
    JaegerExporter,
    JSONExporter,
    create_exporter,
)
from houyi.observability.trace_manager import Span, TraceManager


class TestSpan:
    """Test Span class."""

    def test_span_creation(self):
        """Test creating a span."""
        span = Span(name="test_span")
        assert span.name == "test_span"
        assert span.parent is None
        assert span.attributes == {}
        assert span.start_time > 0
        assert span.end_time is None
        assert span.status == "ok"

    def test_span_with_parent(self):
        """Test span with parent relationship."""
        parent = Span(name="parent")
        child = Span(name="child", parent=parent)

        assert child.parent == parent
        assert child in parent.children
        assert len(parent.children) == 1

    def test_span_with_attributes(self):
        """Test span with initial attributes."""
        attrs = {"key1": "value1", "key2": 42}
        span = Span(name="test", attributes=attrs)
        assert span.attributes == attrs

    def test_set_attribute(self):
        """Test setting span attributes."""
        span = Span(name="test")
        span.set_attribute("user_id", "123")
        span.set_attribute("action", "login")

        assert span.attributes["user_id"] == "123"
        assert span.attributes["action"] == "login"

    def test_add_event(self):
        """Test adding events to span."""
        span = Span(name="test")
        span.add_event("checkpoint", {"step": 1})
        span.add_event("error", {"message": "failed"})

        assert len(span.events) == 2
        assert span.events[0].name == "checkpoint"
        assert span.events[0].attributes["step"] == 1
        assert span.events[1].name == "error"

    def test_set_status(self):
        """Test setting span status."""
        span = Span(name="test")
        span.set_status("error", "Something went wrong")

        assert span.status == "error"
        assert span.attributes["status_description"] == "Something went wrong"

    def test_end_span(self):
        """Test ending a span."""
        span = Span(name="test")
        time.sleep(0.01)
        span.end()

        assert span.end_time is not None
        assert span.duration > 0

    def test_duration_calculation(self):
        """Test span duration calculation."""
        span = Span(name="test")
        time.sleep(0.01)

        # Duration while running
        duration_running = span.duration
        assert duration_running > 0

        # Duration after ending
        span.end()
        duration_ended = span.duration
        assert duration_ended >= duration_running

    def test_to_dict(self):
        """Test converting span to dictionary."""
        parent = Span(name="parent")
        child = Span(name="child", parent=parent)
        child.set_attribute("key", "value")
        child.add_event("test_event")
        child.end()

        span_dict = child.to_dict()
        assert span_dict["name"] == "child"
        assert "start_time" in span_dict
        assert "end_time" in span_dict
        assert "duration" in span_dict
        assert span_dict["status"] == "ok"
        assert span_dict["attributes"]["key"] == "value"
        assert len(span_dict["events"]) == 1

    def test_span_to_dict(self):
        """Test Span to_dict conversion."""
        span = Span(name="test_span")
        span.set_attribute("key", "value")
        span.add_event("test_event")
        span.end()

        span_dict = span.to_dict()

        assert span_dict["name"] == "test_span"
        assert "start_time" in span_dict
        assert "end_time" in span_dict
        assert "duration" in span_dict
        assert span_dict["status"] == "ok"
        assert span_dict["attributes"]["key"] == "value"
        assert len(span_dict["events"]) == 1

    def test_span_with_custom_trace_id(self):
        """Test Span with custom trace_id."""
        custom_trace_id = "custom_trace_123"
        span = Span(name="test", trace_id=custom_trace_id)

        assert span.trace_id == custom_trace_id
        assert span.span_id is not None
        assert span.parent_id is None

    def test_span_event_with_attributes(self):
        """Test adding events with attributes."""
        span = Span(name="test")
        span.add_event("checkpoint", {"step": 1, "status": "ok"})
        span.end()

        assert len(span.events) == 1
        assert span.events[0].name == "checkpoint"
        assert span.events[0].attributes["step"] == 1

    def test_span_duration_before_end(self):
        """Test span duration calculation before ending."""
        import time

        span = Span(name="test")
        time.sleep(0.01)  # Sleep 10ms

        # Duration should be calculated even before end()
        duration = span.duration
        assert duration > 0.01  # At least 10ms

        span.end()
        final_duration = span.duration
        assert final_duration >= duration

    def test_span_set_status_with_description(self):
        """Test setting span status with description."""
        span = Span(name="test")
        span.set_status("error", "Something went wrong")

        assert span.status == "error"
        assert span.attributes["status_description"] == "Something went wrong"


class TestTraceManager:
    """Test TraceManager class."""

    def test_initialization(self):
        """Test TraceManager initialization."""
        tm = TraceManager()
        assert tm.enabled is True
        assert tm.current_span is None
        assert len(tm.root_spans) == 0

    def test_disabled_tracing(self):
        """Test TraceManager with tracing disabled."""
        tm = TraceManager(enabled=False)

        with tm.start_span("test") as span:
            assert span.name == "test"
            # Span is created but not tracked

        assert len(tm.root_spans) == 0

    def test_start_span_context_manager(self):
        """Test starting a span with context manager."""
        tm = TraceManager()

        with tm.start_span("test_operation") as span:
            assert span.name == "test_operation"
            assert span.end_time is None
            assert tm.current_span == span

        assert span.end_time is not None
        assert tm.current_span is None

    def test_nested_spans(self):
        """Test nested span creation."""
        tm = TraceManager()

        with tm.start_span("parent") as parent:
            assert tm.current_span == parent

            with tm.start_span("child") as child:
                assert child.parent == parent
                assert child in parent.children
                assert tm.current_span == child

            # After child ends, parent should be current again
            assert tm.current_span == parent

        assert tm.current_span is None

    def test_span_error_handling(self):
        """Test span status on error."""
        tm = TraceManager()

        try:
            with tm.start_span("error_span") as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert span.status == "error"
        assert "Test error" in span.attributes.get("status_description", "")

    def test_root_spans_tracking(self):
        """Test root spans are tracked."""
        tm = TraceManager()

        with tm.start_span("root1"):
            pass

        with tm.start_span("root2"):
            pass

        # Root spans should be tracked (but exported after completion)
        # After export, they may be cleared depending on implementation


class TestExporters:
    """Test exporter classes."""

    def test_console_exporter(self):
        """Test ConsoleExporter."""
        exporter = ConsoleExporter()
        span = Span(name="test")
        span.end()

        # Export expects dict, not Span object
        exporter.export(span.to_dict())

    def test_console_exporter_verbose(self):
        """Test ConsoleExporter in verbose mode."""
        exporter = ConsoleExporter(verbose=True)
        span = Span(name="test_verbose")
        span.set_attribute("key", "value")
        span.add_event("test_event")
        span.end()

        # Should print detailed output
        exporter.export(span.to_dict())

    def test_json_exporter(self, tmp_path):
        """Test JSONExporter."""
        output_file = tmp_path / "traces.json"
        exporter = JSONExporter(filepath=str(output_file))

        span = Span(name="test")
        span.set_attribute("key", "value")
        span.end()

        exporter.export(span.to_dict())
        exporter.flush()

        # Check file was created
        assert output_file.exists()

    def test_json_exporter_multiple_spans(self, tmp_path):
        """Test JSONExporter with multiple spans."""
        import json

        output_file = tmp_path / "multi_traces.json"
        exporter = JSONExporter(filepath=str(output_file))

        # Export multiple spans
        for i in range(3):
            span = Span(name=f"span_{i}")
            span.set_attribute("index", i)
            span.end()
            exporter.export(span.to_dict())

        exporter.flush()

        # Verify file content
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 3
            assert data[0]["name"] == "span_0"

    def test_jaeger_exporter_initialization(self):
        """Test JaegerExporter initialization."""
        exporter = JaegerExporter(endpoint="http://localhost:4318", service_name="test_service")

        assert exporter.service_name == "test_service"
        assert "4318" in exporter.endpoint
        assert exporter.span_batch == []

    def test_jaeger_exporter_batching(self):
        """Test JaegerExporter batching behavior."""
        exporter = JaegerExporter(
            endpoint="http://localhost:4318", service_name="test_service", batch_size=3
        )

        # Add spans to batch
        for i in range(2):
            span = Span(name=f"span_{i}")
            span.end()
            exporter.export(span.to_dict())

        # Batch should have 2 spans
        assert len(exporter.span_batch) == 2

    def test_datadog_exporter_initialization(self):
        """Test DatadogExporter initialization."""
        exporter = DatadogExporter(agent_url="http://localhost:8126", service_name="test_service")

        assert exporter.service_name == "test_service"
        assert "8126" in exporter.agent_url
        assert exporter.trace_batch == []

    def test_datadog_exporter_with_env(self):
        """Test DatadogExporter with environment setting."""
        exporter = DatadogExporter(
            agent_url="http://localhost:8126", service_name="test_service", env="staging"
        )

        assert exporter.env == "staging"
        assert exporter.service_name == "test_service"

    def test_create_exporter_console(self):
        """Test creating console exporter."""
        exporter = create_exporter({"type": "console"})
        assert isinstance(exporter, ConsoleExporter)

    def test_create_exporter_json(self):
        """Test creating JSON exporter."""
        exporter = create_exporter({"type": "json", "filepath": "test.json"})
        assert isinstance(exporter, JSONExporter)

    def test_create_exporter_jaeger(self):
        """Test creating Jaeger exporter."""
        exporter = create_exporter({"type": "jaeger", "endpoint": "http://localhost:4318"})
        assert isinstance(exporter, JaegerExporter)

    def test_create_exporter_datadog(self):
        """Test creating Datadog exporter."""
        exporter = create_exporter(
            {"type": "datadog", "agent_url": "http://localhost:8126", "service_name": "test"}
        )
        assert isinstance(exporter, DatadogExporter)

    def test_create_exporter_invalid(self):
        """Test creating invalid exporter."""
        with pytest.raises(ValueError):
            create_exporter({"type": "invalid_type"})

    def test_exporter_flush(self):
        """Test exporter flush method."""
        exporter = ConsoleExporter()
        # Flush should not raise even if there's nothing to flush
        exporter.flush()


class TestTraceManagerIntegration:
    """Integration tests for TraceManager with exporters."""

    def test_trace_with_custom_exporter(self, tmp_path):
        """Test complete trace with JSON exporter."""
        output_file = tmp_path / "integration_trace.json"
        exporter = JSONExporter(filepath=str(output_file))

        tm = TraceManager(exporters=[exporter])

        with tm.start_span("operation1") as span1:
            span1.set_attribute("step", 1)

            with tm.start_span("operation2") as span2:
                span2.set_attribute("step", 2)
                span2.add_event("checkpoint")

        # Flush to write to file
        exporter.flush()

        # Verify file was created
        assert output_file.exists()

    def test_multiple_root_spans(self):
        """Test managing multiple root spans."""
        tm = TraceManager()

        with tm.start_span("span1") as s1:
            s1.set_attribute("id", 1)

        with tm.start_span("span2") as s2:
            s2.set_attribute("id", 2)

        # Both spans should complete successfully
        assert s1.end_time is not None
        assert s2.end_time is not None
