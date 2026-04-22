"""Tests for TraceContext and instrumentation."""

from houyi.infrastructure.observability import (
    Span,
    SpanType,
    TraceContext,
    get_current_span,
    set_current_span,
)


class TestTraceContext:
    """Tests for TraceContext class."""

    def test_current_none_when_empty(self):
        """Test that current() returns None when no span is active."""
        assert TraceContext.current() is None

    def test_push_and_pop(self):
        """Test push and pop operations."""
        span = Span(name="test", span_type=SpanType.EXECUTION)

        # Push span
        token = TraceContext.push(span)
        assert TraceContext.current() is span

        # Pop span
        TraceContext.pop(token)
        assert TraceContext.current() is None

    def test_nested_push_pop(self):
        """Test nested push/pop maintains correct stack order."""
        span1 = Span(name="span1", span_type=SpanType.EXECUTION)
        span2 = Span(name="span2", span_type=SpanType.NODE)

        token1 = TraceContext.push(span1)
        assert TraceContext.current() is span1

        token2 = TraceContext.push(span2)
        assert TraceContext.current() is span2

        TraceContext.pop(token2)
        assert TraceContext.current() is span1

        TraceContext.pop(token1)
        assert TraceContext.current() is None

    def test_activate_context_manager(self):
        """Test activate() as context manager."""
        span = Span(name="test", span_type=SpanType.NODE)

        assert TraceContext.current() is None

        with TraceContext.activate(span) as active_span:
            assert active_span is span
            assert TraceContext.current() is span

        assert TraceContext.current() is None

    def test_activate_nested_context_managers(self):
        """Test nested activate() context managers."""
        span1 = Span(name="span1", span_type=SpanType.EXECUTION)
        span2 = Span(name="span2", span_type=SpanType.NODE)

        with TraceContext.activate(span1):
            assert TraceContext.current() is span1

            with TraceContext.activate(span2):
                assert TraceContext.current() is span2

            assert TraceContext.current() is span1

        assert TraceContext.current() is None

    def test_create_child_with_parent(self):
        """Test create_child creates span with correct parent."""
        parent = Span(name="parent", span_type=SpanType.EXECUTION)

        with TraceContext.activate(parent):
            child = TraceContext.create_child("child", span_type=SpanType.NODE)

            assert child is not None
            assert child.name == "child"
            assert child.parent is parent
            assert child.trace_id == parent.trace_id
            assert child.parent_id == parent.span_id

    def test_child_without_parent_none(self):
        """Test create_child returns None when no parent span."""
        assert TraceContext.current() is None
        child = TraceContext.create_child("orphan")
        assert child is None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_current_span(self):
        """Test get_current_span function."""
        assert get_current_span() is None

        span = Span(name="test", span_type=SpanType.NODE)
        token = set_current_span(span)

        assert get_current_span() is span

        TraceContext.pop(token)
        assert get_current_span() is None

    def test_set_current_span(self):
        """Test set_current_span function."""
        span = Span(name="test", span_type=SpanType.NODE)

        token = set_current_span(span)
        assert TraceContext.current() is span

        TraceContext.pop(token)
        assert TraceContext.current() is None


class TestSpanHierarchy:
    """Tests for span hierarchy with TraceContext."""

    def test_child_inherits_trace_id(self):
        """Test that child spans inherit trace_id from parent."""
        root = Span(name="root", span_type=SpanType.EXECUTION)

        with TraceContext.activate(root):
            child = TraceContext.create_child("child", span_type=SpanType.NODE)
            assert child.trace_id == root.trace_id

            with TraceContext.activate(child):
                grandchild = TraceContext.create_child("grandchild", span_type=SpanType.LLM)
                assert grandchild.trace_id == root.trace_id

    def test_span_tree_structure(self):
        """Test that spans form correct tree structure."""
        root = Span(name="execution", span_type=SpanType.EXECUTION)

        with TraceContext.activate(root):
            node1 = Span(name="node1", parent=root, span_type=SpanType.NODE)

            with TraceContext.activate(node1):
                llm = Span(name="llm", parent=node1, span_type=SpanType.LLM)

            node2 = Span(name="node2", parent=root, span_type=SpanType.NODE)

            with TraceContext.activate(node2):
                tool = Span(name="tool", parent=node2, span_type=SpanType.TOOL)

        # Verify tree structure
        assert len(root.children) == 2
        assert root.children[0] is node1
        assert root.children[1] is node2

        assert len(node1.children) == 1
        assert node1.children[0] is llm

        assert len(node2.children) == 1
        assert node2.children[0] is tool

    def test_to_dict_includes_children(self):
        """Test that to_dict includes nested children."""
        root = Span(name="execution", span_type=SpanType.EXECUTION)
        node = Span(name="node", parent=root, span_type=SpanType.NODE)
        llm = Span(name="llm", parent=node, span_type=SpanType.LLM, model="gpt-4")

        root.end()
        node.end()
        llm.end()

        data = root.to_dict()

        assert data["name"] == "execution"
        assert len(data["children"]) == 1

        node_data = data["children"][0]
        assert node_data["name"] == "node"
        assert len(node_data["children"]) == 1

        llm_data = node_data["children"][0]
        assert llm_data["name"] == "llm"
        assert llm_data["model"] == "gpt-4"
