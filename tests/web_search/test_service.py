"""Unit tests for WebSearchService."""

from __future__ import annotations

import pytest

from houyi.verification.cache import LRUCache
from houyi.web_search.errors import (
    ContentFetchError,
    DependencyMissingError,
    ProviderInvalidResponse,
    ProviderRateLimitError,
    ProviderTimeoutError,
    WebSearchError,
)
from houyi.web_search.providers import DuckDuckGoWebSearchProvider
from houyi.web_search.service import (
    WebSearchRetryPolicy,
    WebSearchService,
    _reset_global_cache_for_tests,
)
from houyi.web_search.types import WebSearchResult


class _Provider:
    def __init__(self, *, results=None, error: WebSearchError | None = None) -> None:
        self.name = "tavily"
        self._results = results or []
        self._error = error

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult | dict]:
        if self._error:
            raise self._error
        return self._results


def test_web_search_service_from_env_invalid_provider() -> None:
    """from_env should reject unsupported providers."""

    with pytest.raises(ValueError):
        WebSearchService.from_env(provider="unknown")


def test_web_search_service_from_env_override_env(monkeypatch) -> None:
    """Explicit provider should override WEB_SEARCH_PROVIDER env."""

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "serper")
    service = WebSearchService.from_env(provider="ddg")
    assert service.provider.name == "ddg"


def test_web_search_service_skip_fallback_without_keys(monkeypatch) -> None:
    """Fallback providers requiring keys should be skipped when missing."""

    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    service = WebSearchService.from_env(provider="ddg")
    providers = service._resolve_providers()
    assert [provider.name for provider in providers] == ["ddg"]


def test_web_search_service_from_env_global_cache_enabled(monkeypatch) -> None:
    """from_env should reuse a global cache when WEB_SEARCH_CACHE_TTL is set."""

    _reset_global_cache_for_tests()
    monkeypatch.setenv("WEB_SEARCH_CACHE_TTL", "60")
    monkeypatch.delenv("WEB_SEARCH_CACHE_ENABLED", raising=False)
    service_a = WebSearchService.from_env(provider="ddg")
    service_b = WebSearchService.from_env(provider="ddg")
    assert service_a.cache is not None
    assert service_a.cache is service_b.cache


def test_web_search_service_from_env_global_cache_disabled(monkeypatch) -> None:
    """WEB_SEARCH_CACHE_ENABLED=false should disable cache usage entirely."""

    _reset_global_cache_for_tests()
    monkeypatch.setenv("WEB_SEARCH_CACHE_TTL", "60")
    monkeypatch.setenv("WEB_SEARCH_CACHE_ENABLED", "false")
    service = WebSearchService.from_env(provider="ddg")
    assert service.cache is None


@pytest.mark.asyncio
async def test_web_search_service_search_success() -> None:
    """search should normalize results and return metadata."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1)
    assert response.provider == "tavily"
    assert response.results[0].title == "t"
    assert response.metadata.latency_ms is not None
    assert response.metadata.error_count == 0
    assert response.raw is None


@pytest.mark.asyncio
async def test_web_search_service_search_error() -> None:
    """search should capture provider errors in metadata."""

    provider = _Provider(error=ProviderInvalidResponse("boom"))
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1)
    assert response.results == []
    assert response.metadata.error_count == 1


@pytest.mark.asyncio
async def test_web_search_service_raw_payload() -> None:
    """search should include provider raw payload when available."""

    class _RawProvider(_Provider):
        def __init__(self, *, results=None) -> None:
            super().__init__(results=results)
            self.last_raw_payload = {"raw": True}

    provider = _RawProvider(results=[{"title": "t", "url": "u"}])
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1)
    assert response.raw == {"provider_payload": {"raw": True}}


@pytest.mark.asyncio
async def test_web_search_service_fallback_provider() -> None:
    """Fallback provider should be used when primary fails."""

    class _FailingProvider:
        name = "ddg"

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
            raise ProviderTimeoutError("timeout")

    success_provider = _Provider(results=[{"title": "t", "url": "u"}])
    service = WebSearchService(
        provider=_FailingProvider(),
        fallback_providers=[success_provider],
    )
    response = await service.search("q", max_results=1)
    assert response.results[0].title == "t"
    assert response.metadata.provider == "tavily"
    assert response.metadata.errors[0]["type"] == "ProviderTimeoutError"


@pytest.mark.asyncio
async def test_web_search_service_include_content(monkeypatch) -> None:
    """include_content should populate content when fetcher succeeds."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        return {"u": "content\n\n---\nCopyright 2024"}

    class _Readability:
        async def fetch(self, urls):
            return {}

    monkeypatch.setattr("houyi.web_search.service.JinaContentFetcher.fetch", _fetch)
    monkeypatch.setattr(
        "houyi.web_search.service.ReadabilityContentFetcher", lambda: _Readability()
    )
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1, include_content=True)
    assert response.results[0].content == "content"
    assert response.metadata.extraction_provider == "jina"


@pytest.mark.asyncio
async def test_web_search_service_include_content_error(monkeypatch) -> None:
    """include_content should record errors when fetchers fail."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        raise ContentFetchError("fetch-failed")

    class _Readability:
        async def fetch(self, urls):
            raise ContentFetchError("fetch-failed")

    monkeypatch.setattr("houyi.web_search.service.JinaContentFetcher.fetch", _fetch)
    monkeypatch.setattr(
        "houyi.web_search.service.ReadabilityContentFetcher", lambda: _Readability()
    )
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1, include_content=True)
    assert response.metadata.errors
    assert response.metadata.extraction_provider is None


@pytest.mark.asyncio
async def test_web_search_service_include_content_dependency_missing(monkeypatch) -> None:
    """include_content should record DependencyMissingError with provider label when readability deps are missing."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        return {}

    def _missing_readability():
        raise DependencyMissingError(
            "Missing optional dependency 'readability-lxml' or 'beautifulsoup4'. Install: pip install 'houyi[websearch-readability]'"
        )

    monkeypatch.setattr("houyi.web_search.service.JinaContentFetcher.fetch", _fetch)
    monkeypatch.setattr("houyi.web_search.service.ReadabilityContentFetcher", _missing_readability)
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1, include_content=True)

    assert any(err.get("type") == "DependencyMissingError" for err in response.metadata.errors)
    assert any(err.get("provider") == "readability" for err in response.metadata.errors)
    assert response.metadata.extraction_provider == "readability_unavailable"


@pytest.mark.asyncio
async def test_web_search_service_cache_hit() -> None:
    """Cache should return cached responses and mark metadata.cached."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])
    cache = LRUCache(max_size=10, default_ttl=60)
    service = WebSearchService(provider=provider, cache=cache, cache_ttl=60)
    first = await service.search("q", max_results=1)
    second = await service.search("q", max_results=1)
    assert first.results[0].title == "t"
    assert second.metadata.cached is True
    assert second.metadata.cache_hit is True
    assert second.metadata.provider == "tavily"


@pytest.mark.asyncio
async def test_web_search_service_use_cache_false_disables_cache() -> None:
    """Per-call cache gating should skip cache reads/writes when use_cache is False."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])
    cache = LRUCache(max_size=10, default_ttl=60)
    service = WebSearchService(provider=provider, cache=cache, cache_ttl=60)

    first = await service.search("q", max_results=1)
    # Even though cache is populated, the next call should behave as a non-cache-hit.
    second = await service.search("q", max_results=1, use_cache=False)

    assert first.results[0].title == "t"
    assert second.metadata.cache_hit is False
    assert second.metadata.cached is False


@pytest.mark.asyncio
async def test_web_search_service_cache_hit_preserves_extraction_metadata(monkeypatch) -> None:
    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        return {"u": "content"}

    class _Readability:
        async def fetch(self, urls):
            return {}

    monkeypatch.setattr("houyi.web_search.service.JinaContentFetcher.fetch", _fetch)
    monkeypatch.setattr(
        "houyi.web_search.service.ReadabilityContentFetcher", lambda: _Readability()
    )

    cache = LRUCache(max_size=10, default_ttl=60)
    service = WebSearchService(provider=provider, cache=cache, cache_ttl=60)
    first = await service.search("q", max_results=1, include_content=True)
    second = await service.search("q", max_results=1, include_content=True)

    assert first.metadata.extraction_provider == "jina"
    assert second.metadata.cached is True
    assert second.metadata.cache_hit is True
    assert second.metadata.extraction_provider == "jina"
    assert second.metadata.provider_latency_ms == 0
    assert second.metadata.extraction_latency_ms == 0


@pytest.mark.asyncio
async def test_web_search_service_empty_results_not_cached() -> None:
    """Empty provider results should not be cached (so transient empties don't persist)."""

    calls = {"count": 0}

    class _EmptyThenSuccessProvider:
        name = "tavily"

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult | dict]:
            calls["count"] += 1
            if calls["count"] == 1:
                return []
            return [{"title": "t", "url": "u"}]

    cache = LRUCache(max_size=10, default_ttl=60)
    service = WebSearchService(provider=_EmptyThenSuccessProvider(), cache=cache, cache_ttl=60)

    first = await service.search("q", max_results=1)
    assert first.results == []
    assert first.metadata.error_count == 1
    assert any(err.get("type") == "EmptyResults" for err in first.metadata.errors)

    second = await service.search("q", max_results=1)
    assert second.results
    assert second.results[0].title == "t"
    assert second.metadata.cache_hit is False
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_web_search_service_retries(monkeypatch) -> None:
    """Retry policy should retry for transient errors."""

    attempts = {"count": 0}

    class _RetryProvider:
        name = "tavily"

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise ProviderTimeoutError("timeout")
            return [WebSearchResult(title="t", url="u")]

    async def _sleep(_delay: float) -> None:
        return None

    policy = WebSearchRetryPolicy(max_retries=2, min_delay=0.0, max_delay=0.0, jitter=0.0)
    service = WebSearchService(provider=_RetryProvider(), retry_policy=policy, sleep_func=_sleep)
    response = await service.search("q", max_results=1)
    assert response.results[0].title == "t"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_web_search_service_retry_exhausted() -> None:
    """Retry policy should surface error after exhaustion."""

    class _RetryProvider:
        name = "tavily"

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
            raise ProviderRateLimitError("rate")

    policy = WebSearchRetryPolicy(max_retries=1, min_delay=0.0, max_delay=0.0, jitter=0.0)

    async def _sleep(_delay: float) -> None:
        return None

    service = WebSearchService(provider=_RetryProvider(), retry_policy=policy, sleep_func=_sleep)
    response = await service.search("q", max_results=1)
    assert response.results == []
    assert response.metadata.rate_limit_count == 1


@pytest.mark.asyncio
async def test_query_provider_creates_internal_spans_when_trace_active() -> None:
    """_query_provider should create INTERNAL sub-spans when TraceContext has a parent."""
    from houyi.observability import Span, SpanType, TraceContext

    parent = Span(name="tool.web_search", span_type=SpanType.TOOL)
    token = TraceContext.push(parent)

    try:
        provider = _Provider(results=[{"title": "t", "url": "u"}])
        service = WebSearchService(provider=provider)
        await service.search("q", max_results=1)
    finally:
        TraceContext.pop(token)

    # Parent should have at least one child (provider.tavily)
    assert len(parent.children) >= 1
    provider_span = parent.children[0]
    assert provider_span.span_type == SpanType.INTERNAL
    assert "provider.tavily" in provider_span.name
    assert provider_span.end_time is not None
    assert provider_span.status == "ok"


@pytest.mark.asyncio
async def test_query_provider_no_spans_without_trace_context() -> None:
    """No spans should be created when TraceContext has no active parent."""
    from houyi.observability import TraceContext

    # Ensure no active context
    assert TraceContext.current() is None

    provider = _Provider(results=[{"title": "t", "url": "u"}])
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1)
    # Should still work normally
    assert response.results[0].title == "t"


@pytest.mark.asyncio
async def test_query_provider_error_span_on_failure() -> None:
    """Provider failure should create an error span."""
    from houyi.observability import Span, SpanType, TraceContext

    parent = Span(name="tool.web_search", span_type=SpanType.TOOL)
    token = TraceContext.push(parent)

    try:
        provider = _Provider(error=ProviderTimeoutError("timeout"))
        service = WebSearchService(provider=provider)
        await service.search("q", max_results=1)
    finally:
        TraceContext.pop(token)

    assert len(parent.children) >= 1
    error_span = parent.children[0]
    assert error_span.span_type == SpanType.INTERNAL
    assert error_span.status == "error"
    assert error_span.end_time is not None


@pytest.mark.asyncio
async def test_content_fetch_creates_sub_spans(monkeypatch) -> None:
    """include_content should create content.fetch and fetch.jina sub-spans."""
    from houyi.observability import Span, SpanType, TraceContext

    parent = Span(name="tool.web_search", span_type=SpanType.TOOL)
    token = TraceContext.push(parent)

    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        return {"u": "content text"}

    monkeypatch.setattr("houyi.web_search.service.JinaContentFetcher.fetch", _fetch)

    try:
        service = WebSearchService(provider=provider)
        response = await service.search("q", max_results=1, include_content=True)
    finally:
        TraceContext.pop(token)

    assert response.results[0].content == "content text"

    # Should have provider span + content.fetch span (with fetch.jina child)
    span_names = [c.name for c in parent.children]
    assert any("provider." in n for n in span_names)
    assert any("content.fetch" in n for n in span_names)

    # content.fetch should have a fetch.jina child
    fetch_span = next(c for c in parent.children if "content.fetch" in c.name)
    jina_children = [c for c in fetch_span.children if "fetch.jina" in c.name]
    assert len(jina_children) == 1
    assert jina_children[0].status == "ok"


@pytest.mark.asyncio
async def test_ddg_provider_remote_disconnected_translates_to_timeout(monkeypatch) -> None:
    """DDG network disconnects should raise ProviderTimeoutError (so retries/fallback can engage)."""

    from http.client import RemoteDisconnected

    def _urlopen(*_args, **_kwargs):
        raise RemoteDisconnected("Remote end closed connection")

    monkeypatch.setattr("houyi.web_search.providers.request.urlopen", _urlopen)

    provider = DuckDuckGoWebSearchProvider()
    with pytest.raises(ProviderTimeoutError):
        await provider.search("q", max_results=1)
