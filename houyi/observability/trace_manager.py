"""Trace manager for OpenTelemetry integration.

Provides Span and TraceManager for AI/Agent observability with:
- AI-native fields (model, tokens, cost, cache_hit)
- Parallel execution tracking (group_id, lane_id)
- Checkpoint/restore lineage
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from houyi.observability.exporters import ExporterConfig, create_exporter
from houyi.observability.types import (
    CostInfo,
    SpanEvent,
    SpanType,
    TokenUsage,
)


class Span(BaseModel):
    """Span for tracing with AI-native fields.

    Represents a single operation in the trace. Supports:
    - Hierarchical span types (execution/node/llm/tool/retriever)
    - AI-native fields (model, tokens, cost)
    - Parallel execution tracking
    - Checkpoint/restore lineage
    """

    # Core identifiers
    name: str
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None

    # Timing
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None

    # Status
    status: str = "ok"
    status_description: str | None = None

    # Span type and hierarchy
    span_type: SpanType = SpanType.NODE
    node_id: str | None = None

    # AI-native fields (LLM)
    model: str | None = None
    provider: str | None = None
    tokens: TokenUsage | None = None
    cost: CostInfo | None = None
    cache_hit: bool | None = None

    # AI-native fields (Tool)
    tool_name: str | None = None

    # AI-native fields (Retriever)
    kb_name: str | None = None
    docs_count: int | None = None
    top_k: int | None = None

    # Parallel execution fields
    group_id: str | None = None
    lane_id: int | None = None
    seq: int | None = None

    # Checkpoint/restore lineage
    parent_trace_id: str | None = None
    restore_checkpoint_id: str | None = None
    replay_mode: bool = False

    # Generic attributes (for extensibility)
    attributes: dict[str, Any] = Field(default_factory=dict)

    # Events within span
    events: list[SpanEvent] = Field(default_factory=list)

    # Children spans (for tree structure, not serialized by default)
    children: list[Span] = Field(default_factory=list)

    # Private: parent reference (not serialized)
    _parent: Span | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        span_type: SpanType = SpanType.NODE,
        **kwargs: Any,
    ):
        """Initialize span with backward-compatible signature.

        Args:
            name: Span name
            parent: Parent span (optional)
            attributes: Initial attributes (optional)
            trace_id: Trace ID (optional, inherited from parent or generated)
            span_type: Span type (default: NODE)
            **kwargs: Additional fields (model, tokens, etc.)
        """
        # Resolve trace_id and parent_id from parent
        resolved_trace_id = trace_id
        resolved_parent_id = None
        if parent is not None:
            resolved_trace_id = parent.trace_id
            resolved_parent_id = parent.span_id

        super().__init__(
            name=name,
            trace_id=resolved_trace_id or uuid.uuid4().hex,
            parent_id=resolved_parent_id,
            attributes=attributes or {},
            span_type=span_type,
            **kwargs,
        )

        # Set private parent reference and register as child
        self._parent = parent
        if parent is not None:
            parent.children.append(self)

    @property
    def parent(self) -> Span | None:
        """Get parent span."""
        return self._parent

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to span."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=time.time(),
                attributes=attributes or {},
            )
        )

    def set_status(self, status: str, description: str | None = None) -> None:
        """Set span status."""
        self.status = status
        if description:
            self.status_description = description
            self.attributes["status_description"] = description

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = time.time()

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set token usage for LLM spans."""
        self.tokens = TokenUsage(
            input=input_tokens,
            output=output_tokens,
            total=input_tokens + output_tokens,
        )

    def set_cost(self, usd: float) -> None:
        """Set cost for LLM spans."""
        self.cost = CostInfo(usd=usd)

    @property
    def duration(self) -> float:
        """Get span duration in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for export."""
        data = self.model_dump(
            exclude={"children"},
            exclude_none=True,
            mode="json",
        )
        data["duration"] = self.duration
        data["children"] = [child.to_dict() for child in self.children]
        return data


class TraceManager:
    """Trace manager for observability.

    Manages traces and spans, and exports to configured backends.
    Zero-config by default (console output).
    """

    def __init__(
        self,
        enabled: bool = True,
        exporters: list[dict | ExporterConfig] | None = None,
    ):
        """Initialize trace manager.

        Args:
            enabled: Whether tracing is enabled
            exporters: List of exporter configurations
        """
        self.enabled = enabled
        self.current_span: Span | None = None
        self.root_spans: list[Span] = []

        # Initialize exporters
        if exporters is None:
            # Default to console exporter
            exporters = [{"type": "console"}]

        self.exporters = [
            create_exporter(exp) if isinstance(exp, dict) else exp for exp in exporters
        ]

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Start a new span.

        Args:
            name: Span name
            attributes: Optional attributes

        Yields:
            Span instance
        """
        if not self.enabled:
            # Create a no-op span
            span = Span(name, attributes=attributes)
            yield span
            return

        # Create span
        span = Span(name, parent=self.current_span, attributes=attributes)

        # Track root spans
        if self.current_span is None:
            self.root_spans.append(span)

        # Set as current
        previous_span = self.current_span
        self.current_span = span

        try:
            yield span
        except Exception as e:
            span.set_status("error", str(e))
            raise
        finally:
            span.end()
            self.current_span = previous_span

            # Export if root span
            if previous_span is None:
                self._export_span(span)

    def start_agent_run(
        self,
        agent_role: str,
        input_text: str,
        **kwargs: Any,
    ) -> Span:
        """Start agent run span.

        Args:
            agent_role: Agent role
            input_text: Input text
            **kwargs: Additional attributes

        Returns:
            Span instance
        """
        attributes = {
            "agent.role": agent_role,
            "agent.input": input_text,
            **kwargs,
        }

        span = Span("agent.run", parent=self.current_span, attributes=attributes)

        if self.current_span is None:
            self.root_spans.append(span)

        self.current_span = span

        return span

    def start_llm_call(
        self,
        model: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> Span:
        """Start LLM call span.

        Args:
            model: Model name
            messages: Messages
            **kwargs: Additional attributes

        Returns:
            Span instance
        """
        attributes = {
            "llm.model": model,
            "llm.message_count": len(messages),
            **kwargs,
        }

        span = Span("llm.call", parent=self.current_span, attributes=attributes)

        if self.current_span is None:
            self.root_spans.append(span)

        return span

    def start_skill_execution(
        self,
        skill_name: str,
        input_data: dict,
        **kwargs: Any,
    ) -> Span:
        """Start skill execution span.

        Args:
            skill_name: Skill name
            input_data: Input data
            **kwargs: Additional attributes

        Returns:
            Span instance
        """
        attributes = {
            "skill.name": skill_name,
            "skill.input": str(input_data),
            **kwargs,
        }

        span = Span("skill.execute", parent=self.current_span, attributes=attributes)

        if self.current_span is None:
            self.root_spans.append(span)

        return span

    def end_span(self, span: Span, **attributes: Any) -> None:
        """End a span.

        Args:
            span: Span to end
            **attributes: Additional attributes to set
        """
        for key, value in attributes.items():
            span.set_attribute(key, value)

        span.end()

        # Reset current span if this was it
        if self.current_span == span:
            self.current_span = span.parent

        # Export if root span
        if span.parent is None:
            self._export_span(span)

    def _export_span(self, span: Span) -> None:
        """Export span to all exporters.

        Args:
            span: Span to export
        """
        if not self.enabled:
            return

        span_data = span.to_dict()

        for exporter in self.exporters:
            try:
                exporter.export(span_data)  # type: ignore[union-attr]
            except Exception as e:
                # Don't let exporter errors break execution
                print(f"Exporter error: {e}")

    def flush(self) -> None:
        """Flush all exporters."""
        for exporter in self.exporters:
            with contextlib.suppress(Exception):
                exporter.flush()  # type: ignore[union-attr]
