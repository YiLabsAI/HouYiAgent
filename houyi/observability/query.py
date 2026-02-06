"""Unified query interface for observability data.

Provides high-level query APIs that combine span storage and content store
for comprehensive trace analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from houyi.observability.content_store import (
    ContentRef,
    ContentStore,
    ContentType,
    get_content_store,
)
from houyi.observability.storage import (
    SpanFilter,
    SpanStorage,
    get_storage,
)
from houyi.observability.types import SpanSchema, SpanType


@dataclass
class TraceView:
    """Complete view of a trace with spans and content."""

    trace_id: str
    spans: list[SpanSchema]
    content_refs: list[ContentRef]
    root_span: SpanSchema | None = None
    total_duration_ms: float | None = None
    span_count: int = 0
    error_count: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @classmethod
    def from_spans(
        cls,
        trace_id: str,
        spans: list[SpanSchema],
        content_refs: list[ContentRef] | None = None,
    ) -> TraceView:
        """Create TraceView from spans."""
        root_span = None
        total_duration_ms = None
        error_count = 0
        llm_call_count = 0
        tool_call_count = 0
        total_tokens = 0
        total_cost_usd = 0.0

        for span in spans:
            if span.parent_id is None:
                root_span = span
                if span.end_time and span.start_time:
                    total_duration_ms = (span.end_time - span.start_time) * 1000

            if span.status == "error":
                error_count += 1

            if span.span_type == SpanType.LLM:
                llm_call_count += 1
                if span.tokens:
                    total_tokens += span.tokens.total or 0
                if span.cost:
                    total_cost_usd += span.cost.usd or 0.0

            if span.span_type == SpanType.TOOL:
                tool_call_count += 1

        return cls(
            trace_id=trace_id,
            spans=spans,
            content_refs=content_refs or [],
            root_span=root_span,
            total_duration_ms=total_duration_ms,
            span_count=len(spans),
            error_count=error_count,
            llm_call_count=llm_call_count,
            tool_call_count=tool_call_count,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
        )


@dataclass
class SpanWithContent:
    """Span with associated content."""

    span: SpanSchema
    content: dict[ContentType, str | dict[str, Any] | None] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of a query operation."""

    spans: list[SpanSchema]
    total_count: int
    has_more: bool
    query_time_ms: float


class ObservabilityQuery:
    """High-level query interface for observability data.

    Combines span storage and content store for comprehensive queries.
    """

    def __init__(
        self,
        span_storage: SpanStorage | None = None,
        content_store: ContentStore | None = None,
    ):
        self._span_storage = span_storage
        self._content_store = content_store

    @property
    def span_storage(self) -> SpanStorage:
        """Get span storage (lazy initialization)."""
        if self._span_storage is None:
            self._span_storage = get_storage()
        return self._span_storage

    @property
    def content_store(self) -> ContentStore:
        """Get content store (lazy initialization)."""
        if self._content_store is None:
            self._content_store = get_content_store()
        return self._content_store

    def get_trace(self, trace_id: str, include_content: bool = False) -> TraceView | None:
        """Get complete trace view.

        Args:
            trace_id: Trace identifier
            include_content: Whether to include content references

        Returns:
            TraceView or None if trace not found
        """
        spans = self.span_storage.get_trace(trace_id)
        if not spans:
            return None

        content_refs = []
        if include_content:
            content_refs = self.content_store.list_refs(trace_id)

        return TraceView.from_spans(trace_id, spans, content_refs)

    def get_span(self, span_id: str, include_content: bool = False) -> SpanWithContent | None:
        """Get span with optional content.

        Args:
            span_id: Span identifier
            include_content: Whether to fetch associated content

        Returns:
            SpanWithContent or None if span not found
        """
        span = self.span_storage.get(span_id)
        if not span:
            return None

        result = SpanWithContent(span=span)

        if include_content:
            # Find content for this span
            content_refs = self.content_store.list_refs(span.trace_id)
            for ref in content_refs:
                if ref.span_id == span_id:
                    content = self.content_store.retrieve(ref.content_id)
                    result.content[ref.content_type] = content

        return result

    def query_spans(
        self,
        trace_id: str | None = None,
        node_id: str | None = None,
        span_type: SpanType | None = None,
        status: str | None = None,
        start_time_gte: float | None = None,
        start_time_lte: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """Query spans with filters.

        Args:
            trace_id: Filter by trace ID
            node_id: Filter by node ID
            span_type: Filter by span type
            status: Filter by status (ok/error)
            start_time_gte: Filter by start time >= value
            start_time_lte: Filter by start time <= value
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            QueryResult with matching spans
        """
        import time

        start = time.time()

        filter = SpanFilter(
            trace_id=trace_id,
            node_id=node_id,
            span_type=span_type,
            status=status,
            start_time_gte=start_time_gte,
            start_time_lte=start_time_lte,
            limit=limit + 1,  # Fetch one extra to check has_more
            offset=offset,
        )

        spans = self.span_storage.query(filter)

        has_more = len(spans) > limit
        if has_more:
            spans = spans[:limit]

        query_time_ms = (time.time() - start) * 1000

        return QueryResult(
            spans=spans,
            total_count=len(spans),
            has_more=has_more,
            query_time_ms=query_time_ms,
        )

    def query_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime | None = None,
        span_type: SpanType | None = None,
        limit: int = 100,
    ) -> QueryResult:
        """Query spans by time range.

        Args:
            start_time: Start of time range
            end_time: End of time range (defaults to now)
            span_type: Optional span type filter
            limit: Maximum results

        Returns:
            QueryResult with matching spans
        """
        if end_time is None:
            end_time = datetime.utcnow()

        return self.query_spans(
            span_type=span_type,
            start_time_gte=start_time.timestamp(),
            start_time_lte=end_time.timestamp(),
            limit=limit,
        )

    def get_recent_traces(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> list[TraceView]:
        """Get recent traces.

        Args:
            hours: Look back this many hours
            limit: Maximum traces to return

        Returns:
            List of TraceView objects
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Query root spans (execution type, no parent)
        result = self.query_spans(
            span_type=SpanType.EXECUTION,
            start_time_gte=cutoff.timestamp(),
            limit=limit,
        )

        traces = []
        for span in result.spans:
            trace_view = self.get_trace(span.trace_id)
            if trace_view:
                traces.append(trace_view)

        return traces

    def get_error_spans(
        self,
        trace_id: str | None = None,
        hours: int | None = None,
        limit: int = 100,
    ) -> list[SpanSchema]:
        """Get spans with errors.

        Args:
            trace_id: Optional trace ID filter
            hours: Optional time filter (last N hours)
            limit: Maximum results

        Returns:
            List of error spans
        """
        start_time_gte = None
        if hours:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            start_time_gte = cutoff.timestamp()

        result = self.query_spans(
            trace_id=trace_id,
            status="error",
            start_time_gte=start_time_gte,
            limit=limit,
        )

        return result.spans

    def get_llm_spans(
        self,
        trace_id: str | None = None,
        model: str | None = None,
        limit: int = 100,
    ) -> list[SpanSchema]:
        """Get LLM spans.

        Args:
            trace_id: Optional trace ID filter
            model: Optional model filter (not implemented in storage filter)
            limit: Maximum results

        Returns:
            List of LLM spans
        """
        result = self.query_spans(
            trace_id=trace_id,
            span_type=SpanType.LLM,
            limit=limit,
        )

        spans = result.spans
        if model:
            spans = [s for s in spans if s.model == model]

        return spans

    def get_tool_spans(
        self,
        trace_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 100,
    ) -> list[SpanSchema]:
        """Get tool spans.

        Args:
            trace_id: Optional trace ID filter
            tool_name: Optional tool name filter
            limit: Maximum results

        Returns:
            List of tool spans
        """
        result = self.query_spans(
            trace_id=trace_id,
            span_type=SpanType.TOOL,
            limit=limit,
        )

        spans = result.spans
        if tool_name:
            spans = [s for s in spans if s.tool_name == tool_name]

        return spans

    def get_node_spans(
        self,
        trace_id: str,
        node_id: str | None = None,
    ) -> list[SpanSchema]:
        """Get node spans for a trace.

        Args:
            trace_id: Trace ID
            node_id: Optional specific node ID

        Returns:
            List of node spans
        """
        result = self.query_spans(
            trace_id=trace_id,
            node_id=node_id,
            span_type=SpanType.NODE,
            limit=1000,
        )

        return result.spans

    def get_span_children(self, span_id: str) -> list[SpanSchema]:
        """Get child spans of a span.

        Args:
            span_id: Parent span ID

        Returns:
            List of child spans
        """
        # Get the span first to find trace_id
        span = self.span_storage.get(span_id)
        if not span:
            return []

        # Query children
        filter = SpanFilter(
            trace_id=span.trace_id,
            parent_span_id=span_id,
            limit=1000,
        )

        return self.span_storage.query(filter)

    def get_trace_statistics(self, trace_id: str) -> dict[str, Any]:
        """Get statistics for a trace.

        Args:
            trace_id: Trace ID

        Returns:
            Dictionary with trace statistics
        """
        trace_view = self.get_trace(trace_id, include_content=True)
        if not trace_view:
            return {}

        # Group spans by type
        by_type: dict[str, int] = {}
        by_node: dict[str, int] = {}
        latencies: list[float] = []

        for span in trace_view.spans:
            span_type = (
                span.span_type.value if hasattr(span.span_type, "value") else str(span.span_type)
            )
            by_type[span_type] = by_type.get(span_type, 0) + 1

            if span.node_id:
                by_node[span.node_id] = by_node.get(span.node_id, 0) + 1

            if span.end_time and span.start_time:
                latencies.append((span.end_time - span.start_time) * 1000)

        avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
        max_latency_ms = max(latencies) if latencies else 0

        return {
            "trace_id": trace_id,
            "span_count": trace_view.span_count,
            "error_count": trace_view.error_count,
            "llm_call_count": trace_view.llm_call_count,
            "tool_call_count": trace_view.tool_call_count,
            "total_tokens": trace_view.total_tokens,
            "total_cost_usd": trace_view.total_cost_usd,
            "total_duration_ms": trace_view.total_duration_ms,
            "avg_span_latency_ms": avg_latency_ms,
            "max_span_latency_ms": max_latency_ms,
            "spans_by_type": by_type,
            "spans_by_node": by_node,
            "content_count": len(trace_view.content_refs),
        }


# Global query instance
_query: ObservabilityQuery | None = None


def get_query() -> ObservabilityQuery:
    """Get global query instance."""
    global _query
    if _query is None:
        _query = ObservabilityQuery()
    return _query


def reset_query() -> None:
    """Reset global query instance."""
    global _query
    _query = None
