"""Instrumentation decorators for automatic span creation.

Provides decorators and utilities for instrumenting LLM calls, tool executions,
and retriever operations with automatic span creation and AI-native field capture.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast

from houyi.infrastructure.observability.context import TraceContext
from houyi.infrastructure.observability.trace_manager import Span
from houyi.infrastructure.observability.types import SpanType

P = ParamSpec("P")
R = TypeVar("R")


def _current_parent() -> Span | None:
    return TraceContext.current()


def _finalize_span(span: Span, exc: Exception | None = None) -> None:
    if exc is not None:
        span.set_status("error", str(exc))
    else:
        span.set_status("ok")
    span.end()


async def _run_async_in_span(
    span: Span,
    func: Callable[P, Awaitable[R]],
    args: P.args,
    kwargs: P.kwargs,
    on_success: Callable[[Span, R], None] | None = None,
) -> R:
    with TraceContext.activate(span):
        try:
            result = await func(*args, **kwargs)
            if on_success is not None:
                on_success(span, result)
            _finalize_span(span)
            return result
        except Exception as exc:
            _finalize_span(span, exc)
            raise


def _run_sync_in_span(
    span: Span,
    func: Callable[P, R],
    args: P.args,
    kwargs: P.kwargs,
    on_success: Callable[[Span, R], None] | None = None,
) -> R:
    token = TraceContext.push(span)
    try:
        result = func(*args, **kwargs)
        if on_success is not None:
            on_success(span, result)
        _finalize_span(span)
        return result
    except Exception as exc:
        _finalize_span(span, exc)
        raise
    finally:
        TraceContext.pop(token)


def _update_llm_span_from_result(
    span: Span,
    result: Any,
    *,
    capture_tokens: bool,
    capture_cost: bool,
) -> None:
    if not isinstance(result, dict):
        return

    if capture_tokens:
        usage = result.get("usage") or result.get("token_usage")
        if isinstance(usage, dict):
            span.set_tokens(
                input_tokens=usage.get("input", 0) or usage.get("prompt_tokens", 0),
                output_tokens=usage.get("output", 0) or usage.get("completion_tokens", 0),
            )

        metadata = result.get("metadata", {})
        if isinstance(metadata, dict) and (
            metadata.get("cache_hit") or metadata.get("llm_cache_hit")
        ):
            span.cache_hit = True
            span.set_attribute("llm.cache_hit", True)

    if capture_cost:
        cost = result.get("cost") or result.get("cost_usd")
        if isinstance(cost, (int, float)):
            span.set_cost(float(cost))


def _update_tool_span_from_result(span: Span, result: Any, *, capture_cache: bool) -> None:
    if not capture_cache or not isinstance(result, dict):
        return
    metadata = result.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("cache_hit"):
        span.cache_hit = True
        span.set_attribute("tool.cache_hit", True)


def _update_retriever_span_from_result(span: Span, result: Any, *, capture_docs: bool) -> None:
    if not capture_docs:
        return
    if isinstance(result, list):
        span.docs_count = len(result)
        span.set_attribute("retriever.docs_count", len(result))
        return
    if isinstance(result, dict) and isinstance(result.get("documents"), list):
        docs = result["documents"]
        span.docs_count = len(docs)
        span.set_attribute("retriever.docs_count", len(docs))


def _build_llm_span(parent: Span, *, model: str | None, provider: str | None) -> Span:
    return Span(
        name="llm.completion",
        parent=parent,
        span_type=SpanType.LLM,
        model=model,
        provider=provider,
        attributes={
            "llm.provider": provider,
            "llm.model": model,
        },
    )


def _resolve_model_name(value: Any, fallback: str | None) -> str | None:
    return value if isinstance(value, str) else fallback


def _build_tool_span(parent: Span, *, tool_name: str) -> Span:
    return Span(
        name=f"tool.{tool_name}",
        parent=parent,
        span_type=SpanType.TOOL,
        tool_name=tool_name,
        attributes={"tool.name": tool_name},
    )


def _build_retriever_span(parent: Span, *, kb_name: str | None, top_k: Any) -> Span:
    return Span(
        name="retriever.query",
        parent=parent,
        span_type=SpanType.RETRIEVER,
        kb_name=kb_name,
        top_k=top_k if isinstance(top_k, int) else None,
        attributes={"retriever.kb_name": kb_name},
    )


def instrument_llm(
    model: str | None = None,
    provider: str | None = None,
    capture_tokens: bool = True,
    capture_cost: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to instrument LLM calls with automatic span creation.

    Creates an 'llm' span as child of current span, capturing:
    - model, provider
    - tokens (input/output/total)
    - cost (if available)
    - cache_hit status

    Args:
        model: Model name (can be overridden at call time).
        provider: Provider name (e.g., 'openai', 'anthropic').
        capture_tokens: Whether to capture token usage.
        capture_cost: Whether to capture cost info.

    Returns:
        Decorated function.

    Example:
        @instrument_llm(provider="openai")
        async def call_gpt(prompt: str, model: str = "gpt-4") -> str:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _awaitable = cast("Callable[P, Awaitable[R]]", func)

        def on_success(span: Span, result: R) -> None:
            _update_llm_span_from_result(
                span,
                result,
                capture_tokens=capture_tokens,
                capture_cost=capture_cost,
            )

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = _current_parent()
            if parent is None:
                return await _awaitable(*args, **kwargs)
            span = _build_llm_span(
                parent,
                model=_resolve_model_name(kwargs.get("model"), model),
                provider=provider,
            )
            return await _run_async_in_span(span, _awaitable, args, kwargs, on_success)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = _current_parent()
            if parent is None:
                return func(*args, **kwargs)
            span = _build_llm_span(
                parent,
                model=_resolve_model_name(kwargs.get("model"), model),
                provider=provider,
            )
            return _run_sync_in_span(span, func, args, kwargs, on_success)

        return (
            cast("Callable[P, R]", async_wrapper)
            if asyncio_iscoroutinefunction(func)
            else sync_wrapper
        )

    return decorator


def instrument_tool(
    tool_name: str | None = None,
    capture_cache: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to instrument tool/skill executions with automatic span creation.

    Creates a 'tool' span as child of current span, capturing:
    - tool.name
    - cache_hit status
    - execution duration

    Args:
        tool_name: Tool name (can be extracted from function name if not provided).
        capture_cache: Whether to capture cache hit status.

    Returns:
        Decorated function.

    Example:
        @instrument_tool(tool_name="web_search")
        async def search(query: str) -> dict:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        effective_name = tool_name or func.__name__
        _awaitable = cast("Callable[P, Awaitable[R]]", func)

        def on_success(span: Span, result: R) -> None:
            _update_tool_span_from_result(span, result, capture_cache=capture_cache)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = _current_parent()
            if parent is None:
                return await _awaitable(*args, **kwargs)
            span = _build_tool_span(parent, tool_name=effective_name)
            return await _run_async_in_span(span, _awaitable, args, kwargs, on_success)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = _current_parent()
            if parent is None:
                return func(*args, **kwargs)
            span = _build_tool_span(parent, tool_name=effective_name)
            return _run_sync_in_span(span, func, args, kwargs, on_success)

        return (
            cast("Callable[P, R]", async_wrapper)
            if asyncio_iscoroutinefunction(func)
            else sync_wrapper
        )

    return decorator


def instrument_retriever(
    kb_name: str | None = None,
    capture_docs: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to instrument retriever operations with automatic span creation.

    Creates a 'retriever' span as child of current span, capturing:
    - kb.name (knowledge base name)
    - docs.count (number of documents retrieved)
    - top_k parameter

    Args:
        kb_name: Knowledge base name.
        capture_docs: Whether to capture document count.

    Returns:
        Decorated function.

    Example:
        @instrument_retriever(kb_name="product_docs")
        async def retrieve(query: str, top_k: int = 5) -> list[dict]:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _awaitable = cast("Callable[P, Awaitable[R]]", func)

        def on_success(span: Span, result: R) -> None:
            _update_retriever_span_from_result(span, result, capture_docs=capture_docs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = _current_parent()
            if parent is None:
                return await _awaitable(*args, **kwargs)
            span = _build_retriever_span(
                parent, kb_name=kb_name, top_k=kwargs.get("top_k") or kwargs.get("k")
            )
            return await _run_async_in_span(span, _awaitable, args, kwargs, on_success)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = _current_parent()
            if parent is None:
                return func(*args, **kwargs)
            span = _build_retriever_span(
                parent, kb_name=kb_name, top_k=kwargs.get("top_k") or kwargs.get("k")
            )
            return _run_sync_in_span(span, func, args, kwargs, on_success)

        return (
            cast("Callable[P, R]", async_wrapper)
            if asyncio_iscoroutinefunction(func)
            else sync_wrapper
        )

    return decorator


class LLMSpanContext:
    """Context manager for manual LLM span creation.

    For cases where decorator-based instrumentation is not suitable.

    Example:
        async with LLMSpanContext(model="gpt-4", provider="openai") as span:
            result = await llm_call(...)
            span.set_tokens(input_tokens=100, output_tokens=50)
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        name: str = "llm.completion",
    ) -> None:
        self._model = model
        self._provider = provider
        self._name = name
        self._span: Span | None = None
        self._token: Any = None

    async def __aenter__(self) -> Span | None:
        parent = TraceContext.current()
        if parent is None:
            return None

        self._span = Span(
            name=self._name,
            parent=parent,
            span_type=SpanType.LLM,
            model=self._model,
            provider=self._provider,
        )
        self._token = TraceContext.push(self._span)
        return self._span

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span is not None:
            if exc_type is not None:
                self._span.set_status("error", str(exc_val))
            else:
                self._span.set_status("ok")
            self._span.end()

        if self._token is not None:
            TraceContext.pop(self._token)


class ToolSpanContext:
    """Context manager for manual tool span creation.

    Example:
        async with ToolSpanContext(tool_name="web_search") as span:
            result = await search(...)
            if result.get("cache_hit"):
                span.cache_hit = True
    """

    def __init__(self, tool_name: str, name: str | None = None) -> None:
        self._tool_name = tool_name
        self._name = name or f"tool.{tool_name}"
        self._span: Span | None = None
        self._token: Any = None

    async def __aenter__(self) -> Span | None:
        parent = TraceContext.current()
        if parent is None:
            return None

        self._span = Span(
            name=self._name,
            parent=parent,
            span_type=SpanType.TOOL,
            tool_name=self._tool_name,
        )
        self._token = TraceContext.push(self._span)
        return self._span

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span is not None:
            if exc_type is not None:
                self._span.set_status("error", str(exc_val))
            else:
                self._span.set_status("ok")
            self._span.end()

        if self._token is not None:
            TraceContext.pop(self._token)


def asyncio_iscoroutinefunction(func: Any) -> bool:
    """Check if function is async (handles wrapped functions)."""
    import asyncio

    # Check the function itself
    if asyncio.iscoroutinefunction(func):
        return True

    # Check wrapped function (for decorators)
    wrapped = getattr(func, "__wrapped__", None)
    if wrapped is not None:
        return asyncio.iscoroutinefunction(wrapped)

    # Check if it's a partial
    if hasattr(func, "func"):
        return asyncio.iscoroutinefunction(func.func)

    return False
