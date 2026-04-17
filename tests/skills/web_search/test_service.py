"""Unit tests for WebSearchService."""

from __future__ import annotations

import pytest

from houyi.assurance.verification.cache import LRUCache
from houyi.infrastructure.config.env_config import (
    ENV_BOCHA_API_KEY,
    ENV_PROXY_URL,
    ENV_SEARXNG_BASE_URL,
    ENV_SERPER_API_KEY,
    ENV_TAVILY_API_KEY,
    ENV_WEB_SEARCH_CACHE_ENABLED,
    ENV_WEB_SEARCH_CACHE_TTL,
    ENV_WEB_SEARCH_PROVIDER,
    ENV_WEB_SEARCH_PROXY_POLICY,
)
from houyi.skills.web_search import service as web_search_service_module
from houyi.skills.web_search.errors import (
    ContentFetchError,
    DependencyMissingError,
    ProviderAuthError,
    ProviderInvalidResponse,
    ProviderRateLimitError,
    ProviderTimeoutError,
    WebSearchError,
)
from houyi.skills.web_search.service import (
    WebSearchRetryPolicy,
    WebSearchService,
    _reset_global_cache_for_tests,
)
from houyi.skills.web_search.types import WebSearchResult


class _Provider:
    def __init__(self, *, results=None, error: WebSearchError | None = None) -> None:
        self.name = "tavily"
        self._results = results or []
        self._error = error

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult | dict]:
        if self._error:
            raise self._error
        return self._results


def test_from_env_invalid_falls_back():
    """from_env should degrade unsupported providers to a supported fallback."""

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.delenv(ENV_SERPER_API_KEY, raising=False)
        monkeypatch.delenv(ENV_TAVILY_API_KEY, raising=False)
        monkeypatch.delenv(ENV_BOCHA_API_KEY, raising=False)
        service = WebSearchService.from_env(provider="unknown")
        assert service.provider.name == "ddg"
    finally:
        monkeypatch.undo()


def test_invalid_falls_back(monkeypatch):
    """Unsupported WEB_SEARCH_PROVIDER env should degrade to auto-detected fallback."""

    monkeypatch.setenv(ENV_WEB_SEARCH_PROVIDER, "google_scholar")
    monkeypatch.delenv(ENV_SERPER_API_KEY, raising=False)
    monkeypatch.delenv(ENV_TAVILY_API_KEY, raising=False)
    monkeypatch.delenv(ENV_BOCHA_API_KEY, raising=False)
    service = WebSearchService.from_env()
    assert service.provider.name == "ddg"


def test_prefers_detected_provider(monkeypatch):
    """Unsupported WEB_SEARCH_PROVIDER env should still honor auto-detect priority."""

    monkeypatch.setenv(ENV_WEB_SEARCH_PROVIDER, "google_scholar")
    monkeypatch.setenv(ENV_SERPER_API_KEY, "test-key")
    service = WebSearchService.from_env()
    assert service.provider.name == "serper"


def test_from_env_overrides(monkeypatch):
    """Explicit provider should override WEB_SEARCH_PROVIDER env."""

    monkeypatch.setenv(ENV_WEB_SEARCH_PROVIDER, "serper")
    service = WebSearchService.from_env(provider="ddg")
    assert service.provider.name == "ddg"


def test_skips_fallback_keys(monkeypatch):
    """Fallback providers requiring keys should be skipped when missing."""

    monkeypatch.delenv(ENV_SERPER_API_KEY, raising=False)
    monkeypatch.delenv(ENV_TAVILY_API_KEY, raising=False)
    monkeypatch.delenv(ENV_SEARXNG_BASE_URL, raising=False)
    service = WebSearchService.from_env(provider="ddg")
    providers = service._resolve_providers()
    assert [provider.name for provider in providers] == ["ddg"]


def test_from_env_cache(monkeypatch):
    """from_env should reuse a global cache when WEB_SEARCH_CACHE_TTL is set."""

    _reset_global_cache_for_tests()
    monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_TTL, "60")
    monkeypatch.delenv(ENV_WEB_SEARCH_CACHE_ENABLED, raising=False)
    service_a = WebSearchService.from_env(provider="ddg")
    service_b = WebSearchService.from_env(provider="ddg")
    assert service_a.cache is not None
    assert service_a.cache is service_b.cache


def test_env_cache_off(monkeypatch):
    """WEB_SEARCH_CACHE_ENABLED=false should disable cache usage entirely."""

    _reset_global_cache_for_tests()
    monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_TTL, "60")
    monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_ENABLED, "false")
    service = WebSearchService.from_env(provider="ddg")
    assert service.cache is None


@pytest.mark.asyncio
async def test_search_succeeds():
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
async def test_search_fails():
    """search should capture provider errors in metadata."""

    provider = _Provider(error=ProviderInvalidResponse("boom"))
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1)
    assert response.results == []
    assert response.metadata.error_count == 1


@pytest.mark.asyncio
async def test_returns_raw_payload():
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
async def test_uses_fallback_provider():
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
async def test_auth_skips_retry():
    """ProviderAuthError should not be retried."""

    attempts = {"count": 0}

    class _AuthProvider:
        name = "serper"

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
            attempts["count"] += 1
            raise ProviderAuthError("bad key")

    policy = WebSearchRetryPolicy(max_retries=3, base_delay=0.0, max_delay=0.0)
    service = WebSearchService(provider=_AuthProvider(), retry_policy=policy)
    response = await service.search("q", max_results=1)

    assert response.results == []
    assert attempts["count"] == 1
    assert response.metadata.error_count == 1
    assert response.metadata.errors[0]["type"] == "ProviderAuthError"


@pytest.mark.asyncio
async def test_auth_uses_fallback():
    """Current runtime still allows fallback after a non-retryable auth error."""

    class _AuthProvider:
        name = "serper"

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
            raise ProviderAuthError("bad key")

    success_provider = _Provider(results=[{"title": "t", "url": "u"}])
    service = WebSearchService(
        provider=_AuthProvider(),
        fallback_providers=[success_provider],
    )
    response = await service.search("q", max_results=1)

    assert response.results[0].title == "t"
    assert response.metadata.provider == "tavily"
    assert response.metadata.errors[0]["type"] == "ProviderAuthError"


@pytest.mark.asyncio
async def test_includes_content(monkeypatch):
    """include_content should populate content when fetcher succeeds."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        return {"u": "content\n\n---\nCopyright 2024"}

    class _Readability:
        async def fetch(self, urls):
            return {}

    monkeypatch.setattr(web_search_service_module.JinaContentFetcher, "fetch", _fetch)
    monkeypatch.setattr(
        web_search_service_module, "ReadabilityContentFetcher", lambda: _Readability()
    )
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1, include_content=True)
    assert response.results[0].content == "content"
    assert response.metadata.extraction_provider == "jina"


@pytest.mark.asyncio
async def test_include_content_fails(monkeypatch):
    """include_content should record errors when fetchers fail."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        raise ContentFetchError("fetch-failed")

    class _Readability:
        async def fetch(self, urls):
            raise ContentFetchError("fetch-failed")

    monkeypatch.setattr(web_search_service_module.JinaContentFetcher, "fetch", _fetch)
    monkeypatch.setattr(
        web_search_service_module, "ReadabilityContentFetcher", lambda: _Readability()
    )
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1, include_content=True)
    assert response.metadata.errors
    assert response.metadata.extraction_provider is None


@pytest.mark.asyncio
async def test_include_content_dep(monkeypatch):
    """include_content should record DependencyMissingError with provider label when readability deps are missing."""

    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        return {}

    def _missing_readability():
        raise DependencyMissingError(
            "Missing optional dependency 'readability-lxml' or 'beautifulsoup4'. Install: pip install 'houyi[websearch-readability]'"
        )

    monkeypatch.setattr(web_search_service_module.JinaContentFetcher, "fetch", _fetch)
    monkeypatch.setattr(
        web_search_service_module, "ReadabilityContentFetcher", _missing_readability
    )
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1, include_content=True)

    assert any(err.get("type") == "DependencyMissingError" for err in response.metadata.errors)
    assert any(err.get("provider") == "readability" for err in response.metadata.errors)
    assert response.metadata.extraction_provider == "readability_unavailable"


@pytest.mark.asyncio
async def test_cache_hits():
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
async def test_cache_key_scopes():
    """Switching provider should NOT hit the other provider's cache."""

    class _ProviderA(_Provider):
        def __init__(self):
            super().__init__(results=[{"title": "from_a", "url": "u"}])
            self.name = "provider_a"

    class _ProviderB(_Provider):
        def __init__(self):
            super().__init__(results=[{"title": "from_b", "url": "u"}])
            self.name = "provider_b"

    cache = LRUCache(max_size=10, default_ttl=60)
    svc_a = WebSearchService(provider=_ProviderA(), cache=cache, cache_ttl=60)
    svc_b = WebSearchService(provider=_ProviderB(), cache=cache, cache_ttl=60)

    resp_a = await svc_a.search("q", max_results=1)
    assert resp_a.results[0].title == "from_a"

    resp_b = await svc_b.search("q", max_results=1)
    assert resp_b.results[0].title == "from_b"
    assert resp_b.metadata.cache_hit is False


@pytest.mark.asyncio
async def test_use_cache_off(monkeypatch):
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
async def test_cache_hit_keeps(monkeypatch):
    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        return {"u": "content"}

    class _Readability:
        async def fetch(self, urls):
            return {}

    monkeypatch.setattr(web_search_service_module.JinaContentFetcher, "fetch", _fetch)
    monkeypatch.setattr(
        web_search_service_module, "ReadabilityContentFetcher", lambda: _Readability()
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
async def test_empty_results_skip(monkeypatch):
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
async def test_retries(monkeypatch):
    """Retry policy should retry for transient errors."""

    attempts = {"count": 0}

    class _RetryProvider:
        name = "tavily"

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise ProviderTimeoutError("timeout")
            return [WebSearchResult(title="t", url="u")]

    policy = WebSearchRetryPolicy(max_retries=2, base_delay=0.0, max_delay=0.0)
    service = WebSearchService(provider=_RetryProvider(), retry_policy=policy)
    response = await service.search("q", max_results=1)
    assert response.results[0].title == "t"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_retry_exhausts():
    """Retry policy should surface error after exhaustion."""

    class _RetryProvider:
        name = "tavily"

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
            raise ProviderRateLimitError("rate")

    policy = WebSearchRetryPolicy(max_retries=1, base_delay=0.0, max_delay=0.0)
    service = WebSearchService(provider=_RetryProvider(), retry_policy=policy)
    response = await service.search("q", max_results=1)
    assert response.results == []
    assert response.metadata.rate_limit_count == 1


@pytest.mark.asyncio
async def test_include_provider_errors():
    class _TimeoutProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
            raise ProviderTimeoutError(f"{self.name} timeout")

    service = WebSearchService(
        provider=_TimeoutProvider("serper"),
        fallback_providers=[_TimeoutProvider("tavily")],
    )
    response = await service.search("q", max_results=1)
    assert response.results == []
    assert response.metadata.error_count >= 2
    providers = {err.get("provider") for err in response.metadata.errors}
    assert "serper" in providers
    assert "tavily" in providers


@pytest.mark.asyncio
async def test_query_creates_spans():
    """_query_provider should create INTERNAL sub-spans when TraceContext has a parent."""
    from houyi.infrastructure.observability import Span, SpanType, TraceContext

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
async def test_query_skips_spans():
    """No spans should be created when TraceContext has no active parent."""
    from houyi.infrastructure.observability import TraceContext

    # Ensure no active context
    assert TraceContext.current() is None

    provider = _Provider(results=[{"title": "t", "url": "u"}])
    service = WebSearchService(provider=provider)
    response = await service.search("q", max_results=1)
    # Should still work normally
    assert response.results[0].title == "t"


@pytest.mark.asyncio
async def test_query_error_span():
    """Provider failure should create an error span."""
    from houyi.infrastructure.observability import Span, SpanType, TraceContext

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
async def test_content_fetch_spans(monkeypatch):
    """include_content should create content.fetch and fetch.jina sub-spans."""
    from houyi.infrastructure.observability import Span, SpanType, TraceContext

    parent = Span(name="tool.web_search", span_type=SpanType.TOOL)
    token = TraceContext.push(parent)

    provider = _Provider(results=[{"title": "t", "url": "u"}])

    async def _fetch(self, urls):
        return {"u": "content text"}

    monkeypatch.setattr(web_search_service_module.JinaContentFetcher, "fetch", _fetch)

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
async def test_from_env_bocha(monkeypatch):
    """from_env should create a Bocha provider when BOCHA_API_KEY is set."""

    monkeypatch.delenv(ENV_SERPER_API_KEY, raising=False)
    monkeypatch.delenv(ENV_TAVILY_API_KEY, raising=False)
    monkeypatch.delenv(ENV_SEARXNG_BASE_URL, raising=False)
    monkeypatch.setenv(ENV_BOCHA_API_KEY, "test-bocha-key")
    _reset_global_cache_for_tests()
    service = WebSearchService.from_env()
    assert service.provider.name == "bocha"


def test_proxy_auto(monkeypatch):
    """Proxy policy should default to auto and inherit detected system proxy."""

    monkeypatch.delenv(ENV_WEB_SEARCH_PROXY_POLICY, raising=False)
    monkeypatch.setenv(ENV_PROXY_URL, "http://127.0.0.1:7890")
    _reset_global_cache_for_tests()
    service = WebSearchService.from_env(provider="ddg")
    assert getattr(service.provider, "proxy_url", None) == "http://127.0.0.1:7890"
    assert getattr(service.provider, "proxy_policy", None) == "auto"


def test_proxy_off(monkeypatch):
    """WEB_SEARCH_PROXY_POLICY=off should disable explicit proxy injection."""

    monkeypatch.setenv(ENV_WEB_SEARCH_PROXY_POLICY, "off")
    monkeypatch.setenv(ENV_PROXY_URL, "http://127.0.0.1:7890")
    _reset_global_cache_for_tests()
    service = WebSearchService.from_env(provider="ddg")
    assert getattr(service.provider, "proxy_url", None) is None
    assert getattr(service.provider, "proxy_policy", None) == "off"
