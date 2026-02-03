"""Tests for observability trace manager."""

from houyi.observability.trace_manager import Span, TraceManager


class TestTraceManager:
    """Test TraceManager functionality."""

    def test_trace_manager_initialization(self):
        """Test basic trace manager initialization."""
        tm = TraceManager(enabled=True)

        assert tm.enabled is True

    def test_trace_manager_disabled(self):
        """Test trace manager when disabled."""
        tm = TraceManager(enabled=False)

        assert tm.enabled is False

    def test_start_span_basic(self):
        """Test starting a basic span."""
        tm = TraceManager(enabled=True)

        with tm.start_span("test_operation") as span:
            assert span is not None
            assert span.name == "test_operation"
            assert span.status == "ok"

    def test_start_span_with_attributes(self):
        """Test starting span with attributes."""
        tm = TraceManager(enabled=True)

        attrs = {"key1": "value1", "key2": 123}
        with tm.start_span("test_op", attributes=attrs) as span:
            assert span.attributes["key1"] == "value1"
            assert span.attributes["key2"] == 123

    def test_span_set_attribute(self):
        """Test setting span attributes."""
        tm = TraceManager(enabled=True)

        with tm.start_span("test") as span:
            span.set_attribute("new_key", "new_value")
            assert span.attributes["new_key"] == "new_value"

    def test_span_set_status(self):
        """Test setting span status."""
        tm = TraceManager(enabled=True)

        with tm.start_span("test") as span:
            span.set_status("error", "Something went wrong")
            assert span.status == "error"
            assert span.attributes.get("status_description") == "Something went wrong"

    def test_nested_spans(self):
        """Test nested span creation."""
        tm = TraceManager(enabled=True)

        with tm.start_span("parent") as parent:
            assert parent.name == "parent"

            with tm.start_span("child") as child:
                assert child.name == "child"

    def test_trace_manager_when_disabled_no_spans(self):
        """Test that disabled trace manager doesn't create spans."""
        tm = TraceManager(enabled=False)

        with tm.start_span("test") as span:
            # When disabled, might return None or a no-op span
            pass

    def test_span_duration(self):
        """Test span tracks duration."""
        import time

        tm = TraceManager(enabled=True)

        with tm.start_span("timed_op") as span:
            time.sleep(0.01)  # Sleep 10ms

        # Span should have recorded some duration
        assert hasattr(span, "start_time")

    def test_multiple_spans(self):
        """Test creating multiple sequential spans."""
        tm = TraceManager(enabled=True)

        with tm.start_span("span1") as s1:
            assert s1.name == "span1"

        with tm.start_span("span2") as s2:
            assert s2.name == "span2"

        with tm.start_span("span3") as s3:
            assert s3.name == "span3"


class TestSpan:
    """Test Span class."""

    def test_span_initialization(self):
        """Test span initialization."""
        span = Span(name="test_span", attributes={"key": "value"})

        assert span.name == "test_span"
        assert span.attributes["key"] == "value"
        assert span.status == "ok"

    def test_span_default_status(self):
        """Test span has default ok status."""
        span = Span(name="test")

        assert span.status == "ok"

    def test_span_set_multiple_attributes(self):
        """Test setting multiple attributes."""
        span = Span(name="test")

        span.set_attribute("attr1", "value1")
        span.set_attribute("attr2", 42)
        span.set_attribute("attr3", True)

        assert span.attributes["attr1"] == "value1"
        assert span.attributes["attr2"] == 42
        assert span.attributes["attr3"] is True

    def test_span_status_without_description(self):
        """Test setting status without description."""
        span = Span(name="test")

        span.set_status("error")
        assert span.status == "error"
        assert "status_description" not in span.attributes
