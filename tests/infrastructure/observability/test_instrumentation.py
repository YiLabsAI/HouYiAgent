"""Tests for observability instrumentation helpers and manual span contexts."""

from __future__ import annotations

import functools
from typing import Any

import pytest

from houyi.infrastructure.observability.api import (
    ObservabilityQuery,
    SpanStatus,
    TokenUsage,
)
from houyi.infrastructure.observability.api import (
    Span as ApiSpan,
)
from houyi.infrastructure.observability.api import (
    SpanType as ApiSpanType,
)
from houyi.infrastructure.observability.config import (
    ObservabilityConfig,
    PrivacyConfig,
    get_config,
    reset_config,
    set_config,
)
from houyi.infrastructure.observability.context import TraceContext
from houyi.infrastructure.observability.instrumentation import (
    LLMSpanContext,
    ToolSpanContext,
    asyncio_iscoroutinefunction,
    instrument_llm,
    instrument_retriever,
    instrument_tool,
)
from houyi.infrastructure.observability.trace_manager import Span
from houyi.infrastructure.observability.types import SpanType


class TestInstrumentationDecorators:
    def test_llm_returns_result(self) -> None:
        @instrument_llm(model="gpt-4", provider="openai")
        def call() -> dict[str, Any]:
            return {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}

        result = call()

        assert result["usage"]["prompt_tokens"] == 1
        assert TraceContext.current() is None

    @pytest.mark.asyncio
    async def test_llm_records_cost(self) -> None:
        root = Span(name="root", span_type=SpanType.EXECUTION)

        @instrument_llm(model="fallback-model", provider="openai")
        async def call(**_kwargs: Any) -> dict[str, Any]:
            return {
                "usage": {"input": 3, "output": 5},
                "cost_usd": 1.25,
                "metadata": {"cache_hit": True},
            }

        with TraceContext.activate(root):
            result = await call(model="runtime-model")

        assert result["usage"]["input"] == 3
        assert len(root.children) == 1
        llm_span = root.children[0]
        assert llm_span.span_type == SpanType.LLM
        assert llm_span.model == "runtime-model"
        assert llm_span.provider == "openai"
        assert llm_span.tokens is not None
        assert llm_span.tokens.input == 3
        assert llm_span.tokens.output == 5
        assert llm_span.cost is not None
        assert llm_span.cost.usd == 1.25
        assert llm_span.cache_hit is True
        assert llm_span.attributes["llm.cache_hit"] is True
        assert llm_span.status == "ok"
        assert llm_span.end_time is not None

    @pytest.mark.asyncio
    async def test_llm_records_error(self) -> None:
        root = Span(name="root", span_type=SpanType.EXECUTION)

        @instrument_llm(provider="openai")
        async def call() -> dict[str, Any]:
            raise RuntimeError("llm failed")

        with TraceContext.activate(root):
            with pytest.raises(RuntimeError, match="llm failed"):
                await call()

        llm_span = root.children[0]
        assert llm_span.status == "error"
        assert llm_span.attributes["status_description"] == "llm failed"

    def test_tool_uses_name(self) -> None:
        root = Span(name="root", span_type=SpanType.EXECUTION)

        @instrument_tool()
        def search_tool() -> dict[str, Any]:
            return {"metadata": {"cache_hit": True}}

        with TraceContext.activate(root):
            result = search_tool()

        assert result["metadata"]["cache_hit"] is True
        tool_span = root.children[0]
        assert tool_span.name == "tool.search_tool"
        assert tool_span.tool_name == "search_tool"
        assert tool_span.cache_hit is True
        assert tool_span.attributes["tool.cache_hit"] is True

    @pytest.mark.asyncio
    async def test_retriever_records_docs(self) -> None:
        root = Span(name="root", span_type=SpanType.EXECUTION)

        @instrument_retriever(kb_name="kb-main")
        async def retrieve(**_kwargs: Any) -> dict[str, Any]:
            return {"documents": [{"id": 1}, {"id": 2}, {"id": 3}]}

        with TraceContext.activate(root):
            result = await retrieve(top_k=7)

        assert len(result["documents"]) == 3
        retriever_span = root.children[0]
        assert retriever_span.span_type == SpanType.RETRIEVER
        assert retriever_span.kb_name == "kb-main"
        assert retriever_span.top_k == 7
        assert retriever_span.docs_count == 3
        assert retriever_span.attributes["retriever.docs_count"] == 3

    def test_retriever_skips_span(self) -> None:
        @instrument_retriever(kb_name="kb")
        def retrieve(**_kwargs: Any) -> list[dict[str, Any]]:
            return [{"id": 1}]

        result = retrieve(k=2)

        assert result == [{"id": 1}]
        assert TraceContext.current() is None


class TestManualSpanContexts:
    @pytest.mark.asyncio
    async def test_span_context_none(self) -> None:
        async with LLMSpanContext(model="gpt-4") as span:
            assert span is None

    @pytest.mark.asyncio
    async def test_span_context_restores(self) -> None:
        root = Span(name="root", span_type=SpanType.EXECUTION)

        with TraceContext.activate(root):
            async with LLMSpanContext(model="gpt-4", provider="openai") as span:
                assert span is not None
                assert TraceContext.current() is span
                span.set_tokens(input_tokens=2, output_tokens=4)
            assert TraceContext.current() is root

        llm_span = root.children[0]
        assert llm_span.status == "ok"
        assert llm_span.tokens is not None
        assert llm_span.tokens.total == 6

    @pytest.mark.asyncio
    async def test_tool_span_error(self) -> None:
        root = Span(name="root", span_type=SpanType.EXECUTION)

        with TraceContext.activate(root):
            with pytest.raises(ValueError, match="tool boom"):
                async with ToolSpanContext(tool_name="web_search"):
                    raise ValueError("tool boom")

        tool_span = root.children[0]
        assert tool_span.name == "tool.web_search"
        assert tool_span.tool_name == "web_search"
        assert tool_span.status == "error"
        assert tool_span.attributes["status_description"] == "tool boom"


class TestCoroutineDetection:
    def test_detects_wrapped_async(self) -> None:
        async def original() -> str:
            return "ok"

        @functools.wraps(original)
        async def wrapped() -> str:
            return await original()

        assert asyncio_iscoroutinefunction(wrapped) is True

    def test_detects_partial_async(self) -> None:
        async def original(arg: str) -> str:
            return arg

        partial_fn = functools.partial(original, "x")

        assert asyncio_iscoroutinefunction(partial_fn) is True

    def test_rejects_sync_function(self) -> None:
        def sync_fn() -> str:
            return "ok"

        assert asyncio_iscoroutinefunction(sync_fn) is False


class TestObservabilityConfigAndApi:
    def teardown_method(self) -> None:
        reset_config()

    def test_config_builds_profiles(self) -> None:
        default = ObservabilityConfig.default()
        development = ObservabilityConfig.development()

        assert default.enabled is True
        assert default.privacy.should_capture_llm_content() is False
        assert default.privacy.should_capture_tool_content() is False
        assert development.privacy.should_capture_llm_content() is True
        assert development.privacy.should_capture_tool_content() is True
        assert development.privacy.capture_retriever_docs is True

    def test_config_round_trip(self) -> None:
        config = ObservabilityConfig(
            enabled=False,
            privacy=PrivacyConfig(capture_prompts=True),
            exporters=[{"type": "json", "filepath": "trace.json"}],
            sample_rate=0.5,
        )

        set_config(config)
        current = get_config()
        assert current is config

        reset_config()
        reset_value = get_config()
        assert reset_value is not config
        assert reset_value.enabled is True

    def test_api_exports_types(self) -> None:
        assert ApiSpan is Span
        assert ApiSpanType.LLM == "llm"
        assert SpanStatus.OK == "ok"
        assert TokenUsage(input=1, output=2, total=3).total == 3
        assert ObservabilityQuery.__name__ == "ObservabilityQuery"
