"""Instrumentation decorators for automatic span creation.

Provides decorators and utilities for instrumenting LLM calls, tool executions,
and retriever operations with automatic span creation and AI-native field capture.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast

from houyi.observability.context import TraceContext
from houyi.observability.trace_manager import Span
from houyi.observability.types import SpanType

P = ParamSpec("P")
R = TypeVar("R")


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

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            effective_model = kwargs.get("model", model)

            parent = TraceContext.current()
            if parent is None:
                return await _awaitable(*args, **kwargs)

            span = Span(
                name="llm.completion",
                parent=parent,
                span_type=SpanType.LLM,
                model=effective_model,
                provider=provider,
                attributes={
                    "llm.provider": provider,
                    "llm.model": effective_model,
                },
            )

            with TraceContext.activate(span):
                try:
                    result = await _awaitable(*args, **kwargs)

                    if capture_tokens and isinstance(result, dict):
                        usage = result.get("usage") or result.get("token_usage")
                        if isinstance(usage, dict):
                            span.set_tokens(
                                input_tokens=usage.get("input", 0) or usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("output", 0)
                                or usage.get("completion_tokens", 0),
                            )

                        metadata = result.get("metadata", {})
                        if isinstance(metadata, dict):
                            if metadata.get("cache_hit") or metadata.get("llm_cache_hit"):
                                span.cache_hit = True
                                span.set_attribute("llm.cache_hit", True)

                    if capture_cost and isinstance(result, dict):
                        cost = result.get("cost") or result.get("cost_usd")
                        if isinstance(cost, (int, float)):
                            span.set_cost(float(cost))

                    span.set_status("ok")
                    return result

                except Exception as e:
                    span.set_status("error", str(e))
                    raise
                finally:
                    span.end()

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            effective_model = kwargs.get("model", model)

            parent = TraceContext.current()
            if parent is None:
                return func(*args, **kwargs)

            span = Span(
                name="llm.completion",
                parent=parent,
                span_type=SpanType.LLM,
                model=effective_model,
                provider=provider,
            )

            token = TraceContext.push(span)
            try:
                result = func(*args, **kwargs)
                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error", str(e))
                raise
            finally:
                span.end()
                TraceContext.pop(token)

        if asyncio_iscoroutinefunction(func):
            return cast("Callable[P, R]", async_wrapper)
        return sync_wrapper

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

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = TraceContext.current()
            if parent is None:
                return await _awaitable(*args, **kwargs)

            span = Span(
                name=f"tool.{effective_name}",
                parent=parent,
                span_type=SpanType.TOOL,
                tool_name=effective_name,
                attributes={
                    "tool.name": effective_name,
                },
            )

            with TraceContext.activate(span):
                try:
                    result = await _awaitable(*args, **kwargs)

                    if capture_cache and isinstance(result, dict):
                        metadata = result.get("metadata", {})
                        if isinstance(metadata, dict):
                            if metadata.get("cache_hit"):
                                span.cache_hit = True
                                span.set_attribute("tool.cache_hit", True)

                    span.set_status("ok")
                    return result

                except Exception as e:
                    span.set_status("error", str(e))
                    raise
                finally:
                    span.end()

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = TraceContext.current()
            if parent is None:
                return func(*args, **kwargs)

            span = Span(
                name=f"tool.{effective_name}",
                parent=parent,
                span_type=SpanType.TOOL,
                tool_name=effective_name,
            )

            token = TraceContext.push(span)
            try:
                result = func(*args, **kwargs)
                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error", str(e))
                raise
            finally:
                span.end()
                TraceContext.pop(token)

        if asyncio_iscoroutinefunction(func):
            return cast("Callable[P, R]", async_wrapper)
        return sync_wrapper

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

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = TraceContext.current()
            if parent is None:
                return await _awaitable(*args, **kwargs)

            top_k = kwargs.get("top_k") or kwargs.get("k")

            span = Span(
                name="retriever.query",
                parent=parent,
                span_type=SpanType.RETRIEVER,
                kb_name=kb_name,
                top_k=top_k if isinstance(top_k, int) else None,
                attributes={
                    "retriever.kb_name": kb_name,
                },
            )

            with TraceContext.activate(span):
                try:
                    result = await _awaitable(*args, **kwargs)

                    if capture_docs:
                        if isinstance(result, list):
                            span.docs_count = len(result)
                            span.set_attribute("retriever.docs_count", len(result))
                        elif isinstance(result, dict) and "documents" in result:
                            docs = result["documents"]
                            if isinstance(docs, list):
                                span.docs_count = len(docs)
                                span.set_attribute("retriever.docs_count", len(docs))

                    span.set_status("ok")
                    return result

                except Exception as e:
                    span.set_status("error", str(e))
                    raise
                finally:
                    span.end()

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent = TraceContext.current()
            if parent is None:
                return func(*args, **kwargs)

            top_k = kwargs.get("top_k") or kwargs.get("k")

            span = Span(
                name="retriever.query",
                parent=parent,
                span_type=SpanType.RETRIEVER,
                kb_name=kb_name,
                top_k=top_k if isinstance(top_k, int) else None,
            )

            token = TraceContext.push(span)
            try:
                result = func(*args, **kwargs)

                if capture_docs and isinstance(result, list):
                    span.docs_count = len(result)

                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error", str(e))
                raise
            finally:
                span.end()
                TraceContext.pop(token)

        if asyncio_iscoroutinefunction(func):
            return cast("Callable[P, R]", async_wrapper)
        return sync_wrapper

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
