"""Trace context for span propagation.

Provides TraceContext for managing span hierarchy and context propagation
across async boundaries. Inspired by OpenTelemetry context propagation.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.observability.trace_manager import Span

# Context variable for current span (async-safe)
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


class TraceContext:
    """Trace context for span propagation.

    Manages the current span stack using contextvars for async-safe propagation.
    Supports both sync and async code paths.

    Usage:
        # Push a span onto the stack
        token = TraceContext.push(span)
        try:
            # ... do work, child spans will auto-parent to this span
        finally:
            TraceContext.pop(token)

        # Or use as context manager
        with TraceContext.activate(span):
            # ... do work
    """

    @staticmethod
    def current() -> Span | None:
        """Get the current active span.

        Returns:
            Current span or None if no span is active.
        """
        return _current_span.get()

    @staticmethod
    def push(span: Span) -> Token[Span | None]:
        """Push a span onto the context stack.

        Args:
            span: Span to make current.

        Returns:
            Token for restoring previous state.
        """
        return _current_span.set(span)

    @staticmethod
    def pop(token: Token[Span | None]) -> None:
        """Pop a span from the context stack.

        Args:
            token: Token from push() call.
        """
        _current_span.reset(token)

    @staticmethod
    def activate(span: Span) -> _SpanContextManager:
        """Activate a span as current (context manager).

        Args:
            span: Span to activate.

        Returns:
            Context manager that restores previous span on exit.
        """
        return _SpanContextManager(span)

    @staticmethod
    def create_child(
        name: str,
        span_type: Any = None,
        **kwargs: Any,
    ) -> Span | None:
        """Create a child span of the current span.

        Args:
            name: Span name.
            span_type: Span type (SpanType enum).
            **kwargs: Additional span attributes.

        Returns:
            New child span, or None if no current span.
        """
        from houyi.observability.trace_manager import Span
        from houyi.observability.types import SpanType

        parent = TraceContext.current()
        if parent is None:
            return None

        return Span(
            name=name,
            parent=parent,
            span_type=span_type or SpanType.NODE,
            **kwargs,
        )


class _SpanContextManager:
    """Context manager for span activation."""

    def __init__(self, span: Span) -> None:
        self._span = span
        self._token: Token[Span | None] | None = None

    def __enter__(self) -> Span:
        self._token = TraceContext.push(self._span)
        return self._span

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            TraceContext.pop(self._token)


def get_current_span() -> Span | None:
    """Convenience function to get current span.

    Returns:
        Current span or None.
    """
    return TraceContext.current()


def set_current_span(span: Span) -> Token[Span | None]:
    """Convenience function to set current span.

    Args:
        span: Span to set as current.

    Returns:
        Token for restoring previous state.
    """
    return TraceContext.push(span)
