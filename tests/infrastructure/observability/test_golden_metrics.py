"""Golden metrics validation tests for the Observability system.

Two categories:
1. **Collection overhead** — p95 latency overhead of span creation + context
   propagation must be < 2% of a simulated workload.
2. **Data correctness** — ≥99.9% of spans must have valid temporal ordering
   (start_time ≤ end_time, parent starts before child).
"""

from __future__ import annotations

import time

from houyi.infrastructure.observability.context import TraceContext
from houyi.infrastructure.observability.trace_manager import Span
from houyi.infrastructure.observability.types import SpanType

# ---------------------------------------------------------------------------
# 1. Collection overhead benchmark
# ---------------------------------------------------------------------------


def _measure_span_overhead_absolute(iterations: int = 5000) -> float:
    """Measure absolute cost of creating root+child span pair.

    Returns average cost in microseconds.
    """
    t0 = time.perf_counter()
    for _ in range(iterations):
        root = Span(name="execution", span_type=SpanType.EXECUTION)
        token = TraceContext.push(root)
        child = Span(name="node", span_type=SpanType.NODE, parent=root)
        child.set_status("ok")
        child.end()
        root.set_status("ok")
        root.end()
        TraceContext.pop(token)
    elapsed = time.perf_counter() - t0
    return (elapsed / iterations) * 1e6  # microseconds


class TestCollectionOverhead:
    """Verify that instrumentation overhead is < 2%.

    Methodology: measure the absolute cost of span creation (root + node +
    llm + tool = 4 spans per node, typical execution), then compare against
    the minimum realistic node execution time.

    Real-world reference latencies:
    - Fast tool call (cache hit): ~50ms
    - LLM API call: 1-10s
    - web_search: 0.5-5s

    The 2% target means: span overhead < 1ms for a 50ms fast-path node.
    """

    # Minimum realistic node execution time (fast tool with cache hit)
    MIN_NODE_LATENCY_MS = 50.0
    MAX_OVERHEAD_PERCENT = 2.0

    def test_span_overhead_below_2_percent_of_min_node_latency(self) -> None:
        """Core golden metric: span creation < 2% of minimum node latency."""
        # Warm up
        _measure_span_overhead_absolute(100)

        # Measure: create 4 spans (execution root + node + llm + tool)
        # This matches a typical single-node execution.
        iterations = 2000
        t0 = time.perf_counter()
        for _ in range(iterations):
            root = Span(name="execution", span_type=SpanType.EXECUTION)
            rt = TraceContext.push(root)
            node = Span(name="node", span_type=SpanType.NODE, parent=root, node_id="n1")
            nt = TraceContext.push(node)
            llm = Span(name="llm.completion", span_type=SpanType.LLM, parent=node, model="gpt-4o")
            llm.set_tokens(input_tokens=500, output_tokens=100)
            llm.set_status("ok")
            llm.end()
            tool = Span(
                name="tool.web_search", span_type=SpanType.TOOL, parent=node, tool_name="web_search"
            )
            tool.set_status("ok")
            tool.end()
            node.set_status("ok")
            node.end()
            TraceContext.pop(nt)
            root.set_status("ok")
            root.end()
            TraceContext.pop(rt)
        elapsed_ms = (time.perf_counter() - t0) / iterations * 1000

        overhead_pct = (elapsed_ms / self.MIN_NODE_LATENCY_MS) * 100

        print(f"\n  4-span creation avg: {elapsed_ms:.3f} ms")
        print(f"  Min node latency:    {self.MIN_NODE_LATENCY_MS:.0f} ms")
        print(f"  Overhead:            {overhead_pct:.2f}%")

        assert overhead_pct < self.MAX_OVERHEAD_PERCENT, (
            f"Span overhead {overhead_pct:.2f}% exceeds "
            f"{self.MAX_OVERHEAD_PERCENT}% of {self.MIN_NODE_LATENCY_MS}ms min node latency"
        )

    def test_absolute_span_creation_cost_under_500us(self) -> None:
        """Span pair (root+child) creation must be < 500μs."""
        # Warm up
        _measure_span_overhead_absolute(100)
        avg_us = _measure_span_overhead_absolute(5000)

        print(f"\n  Span pair (root+child) avg: {avg_us:.1f} μs")
        assert avg_us < 500, f"Span creation too slow: {avg_us:.1f} μs"

    def test_context_propagation_overhead_is_negligible(self) -> None:
        """TraceContext push/current/pop should be microsecond-level.

        This is a micro-benchmark and is sensitive to CPU scheduling and Python
        version differences (e.g. GitHub runners, Python 3.13). We therefore run
        multiple rounds and assert a high percentile bound.
        """
        root = Span(name="root", span_type=SpanType.EXECUTION)

        iterations = 20_000
        rounds = 7
        samples_ns: list[float] = []

        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(iterations):
                token = TraceContext.push(root)
                _ = TraceContext.current()
                TraceContext.pop(token)
            elapsed = time.perf_counter() - t0
            samples_ns.append((elapsed / iterations) * 1e9)

        samples_ns_sorted = sorted(samples_ns)
        p95_ns = samples_ns_sorted[int(len(samples_ns_sorted) * 0.95)]
        median_ns = samples_ns_sorted[len(samples_ns_sorted) // 2]

        print(
            "\n  TraceContext push/current/pop ns per cycle "
            f"(median/p95 over {rounds} rounds): {median_ns:.0f}/{p95_ns:.0f} ns"
        )

        # Keep this in single-digit microseconds even on noisier CI / Python 3.13 runners.
        assert p95_ns < 10000, f"Context propagation too slow: p95={p95_ns:.0f} ns"


# ---------------------------------------------------------------------------
# 2. Data correctness (temporal validity)
# ---------------------------------------------------------------------------


def _build_span_tree() -> list[Span]:
    """Build a realistic span tree: execution → node → llm + tool."""
    spans: list[Span] = []

    root = Span(name="execution", span_type=SpanType.EXECUTION)
    root_token = TraceContext.push(root)
    spans.append(root)

    for i in range(3):
        node = Span(
            name=f"node_{i}",
            span_type=SpanType.NODE,
            parent=root,
            node_id=f"node_{i}",
        )
        node_token = TraceContext.push(node)
        spans.append(node)

        # LLM child
        llm = Span(
            name="llm.completion",
            span_type=SpanType.LLM,
            parent=node,
            model="test-model",
        )
        llm.set_tokens(input_tokens=100, output_tokens=50)
        llm.set_status("ok")
        llm.end()
        spans.append(llm)

        # Tool child
        tool = Span(
            name="tool.web_search",
            span_type=SpanType.TOOL,
            parent=node,
            tool_name="web_search",
        )
        tool.set_status("ok")
        tool.end()
        spans.append(tool)

        node.set_status("ok")
        node.end()
        TraceContext.pop(node_token)

    root.set_status("ok")
    root.end()
    TraceContext.pop(root_token)

    return spans


class TestDataCorrectness:
    """Verify ≥99.9% temporal validity of span data."""

    def test_all_spans_have_valid_start_end_times(self) -> None:
        """Every span must have start_time ≤ end_time."""
        spans = _build_span_tree()
        invalid = [s for s in spans if s.end_time is not None and s.start_time > s.end_time]
        validity = (len(spans) - len(invalid)) / len(spans) * 100

        print(f"\n  Total spans: {len(spans)}, Invalid: {len(invalid)}, Validity: {validity:.1f}%")
        assert validity >= 99.9, f"Temporal validity {validity:.1f}% < 99.9%"
        assert len(invalid) == 0, f"Found {len(invalid)} spans with start > end"

    def test_parent_starts_before_or_equal_to_child(self) -> None:
        """Parent span start_time must be ≤ child span start_time."""
        spans = _build_span_tree()
        span_map = {s.span_id: s for s in spans}

        violations = 0
        checked = 0
        for s in spans:
            if s.parent_id and s.parent_id in span_map:
                parent = span_map[s.parent_id]
                checked += 1
                if parent.start_time > s.start_time:
                    violations += 1

        validity = ((checked - violations) / checked * 100) if checked > 0 else 100
        print(
            f"\n  Parent-child pairs: {checked}, Violations: {violations}, Validity: {validity:.1f}%"
        )
        assert validity >= 99.9, f"Parent-child validity {validity:.1f}% < 99.9%"

    def test_parent_ends_after_or_equal_to_child(self) -> None:
        """Parent span end_time must be ≥ child span end_time."""
        spans = _build_span_tree()
        span_map = {s.span_id: s for s in spans}

        violations = 0
        checked = 0
        for s in spans:
            if s.parent_id and s.parent_id in span_map:
                parent = span_map[s.parent_id]
                if parent.end_time is not None and s.end_time is not None:
                    checked += 1
                    if parent.end_time < s.end_time:
                        violations += 1

        validity = ((checked - violations) / checked * 100) if checked > 0 else 100
        print(f"\n  End-time pairs: {checked}, Violations: {violations}, Validity: {validity:.1f}%")
        assert validity >= 99.9, f"End-time validity {validity:.1f}% < 99.9%"

    def test_trace_id_consistency(self) -> None:
        """All spans in a tree must share the same trace_id."""
        spans = _build_span_tree()
        trace_ids = {s.trace_id for s in spans}
        assert len(trace_ids) == 1, f"Expected 1 trace_id, got {len(trace_ids)}: {trace_ids}"

    def test_no_orphan_spans(self) -> None:
        """Every non-root span must have a parent that exists in the tree."""
        spans = _build_span_tree()
        span_ids = {s.span_id for s in spans}
        root_spans = [s for s in spans if s.parent_id is None]
        child_spans = [s for s in spans if s.parent_id is not None]

        orphans = [s for s in child_spans if s.parent_id not in span_ids]
        assert len(root_spans) == 1, f"Expected 1 root span, got {len(root_spans)}"
        assert len(orphans) == 0, f"Found {len(orphans)} orphan spans"

    def test_large_scale_temporal_validity(self) -> None:
        """Generate many span trees and verify ≥99.9% validity across all."""
        all_spans: list[Span] = []
        for _ in range(100):
            all_spans.extend(_build_span_tree())

        total = len(all_spans)
        invalid = sum(1 for s in all_spans if s.end_time is not None and s.start_time > s.end_time)
        validity = (total - invalid) / total * 100

        print(f"\n  Large-scale: {total} spans, {invalid} invalid, {validity:.2f}% valid")
        assert validity >= 99.9, f"Large-scale validity {validity:.2f}% < 99.9%"
