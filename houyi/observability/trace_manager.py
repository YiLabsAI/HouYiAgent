"""Trace manager for OpenTelemetry integration."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from houyi.observability.exporters import ExporterConfig, create_exporter


class Span:
    """Lightweight span for tracing.

    Represents a single operation in the trace.
    """

    def __init__(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ):
        self.name = name
        self.parent = parent
        self.attributes = attributes or {}
        self.start_time = time.time()
        self.end_time: float | None = None
        self.status = "ok"
        self.events: list[dict] = []
        self.children: list[Span] = []

        # Generate IDs for production-grade observability
        self.span_id = uuid.uuid4().hex[:16]  # 16-char hex
        if parent:
            self.trace_id = parent.trace_id
            self.parent_id = parent.span_id
            parent.children.append(self)
        else:
            self.trace_id = trace_id or uuid.uuid4().hex  # 32-char hex
            self.parent_id = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_status(self, status: str, description: str | None = None) -> None:
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get span duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "children": [child.to_dict() for child in self.children],
        }


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
            create_exporter(exp) if isinstance(exp, dict) else exp
            for exp in exporters
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
                exporter.export(span_data)
            except Exception as e:
                # Don't let exporter errors break execution
                print(f"Exporter error: {e}")

    def flush(self) -> None:
        """Flush all exporters."""
        for exporter in self.exporters:
            try:
                exporter.flush()
            except Exception:
                pass
