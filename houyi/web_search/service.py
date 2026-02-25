from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from houyi.config.env_config import (
    ENV_BOCHA_API_KEY,
    ENV_SEARXNG_BASE_URL,
    ENV_SERPER_API_KEY,
    ENV_TAVILY_API_KEY,
    ENV_WEB_SEARCH_CACHE_ENABLED,
    ENV_WEB_SEARCH_CACHE_MAX_SIZE,
    ENV_WEB_SEARCH_CACHE_TTL,
    ENV_WEB_SEARCH_PROVIDER,
    ENV_WEB_SEARCH_PROXY_ENABLED,
    ENV_WEB_SEARCH_TIMEOUT,
)
from houyi.verification.cache import LRUCache
from houyi.web_search.content_fetchers import JinaContentFetcher, ReadabilityContentFetcher
from houyi.web_search.errors import (
    ContentFetchError,
    DependencyMissingError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    WebSearchError,
)
from houyi.web_search.providers import (
    DEFAULT_PROVIDER_TIMEOUT,
    BochaWebSearchProvider,
    DuckDuckGoWebSearchProvider,
    SearxNGWebSearchProvider,
    SerperWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchProvider,
)
from houyi.web_search.types import WebSearchMetadata, WebSearchResponse, WebSearchResult

logger = logging.getLogger(__name__)

# Observability: optional auto-instrumentation for tool-internal sub-spans.
# Import is safe — observability module is in the same SDK layer.
try:
    from houyi.observability.context import TraceContext as _TraceContext
    from houyi.observability.trace_manager import Span as _Span
    from houyi.observability.types import SpanType as _SpanType

    _HAS_OBSERVABILITY = True
except ImportError:
    _HAS_OBSERVABILITY = False


def _start_internal_span(name: str, attributes: dict[str, Any] | None = None) -> Any:
    """Create an INTERNAL sub-span if observability is active.

    Returns the span object or None if no active trace context.
    """
    if not _HAS_OBSERVABILITY:
        return None
    parent = _TraceContext.current()
    if parent is None:
        return None
    span = _Span(
        name=name,
        parent=parent,
        span_type=_SpanType.INTERNAL,
        attributes=attributes or {},
    )
    return span


def _end_span(span: Any, status: str = "ok", error: str | None = None) -> None:
    """End an internal span if it exists."""
    if span is None:
        return
    if error:
        span.set_status("error", error)
    else:
        span.set_status(status)
    span.end()


_GLOBAL_CACHE: LRUCache | None = None


def _reset_global_cache_for_tests() -> None:
    global _GLOBAL_CACHE
    _GLOBAL_CACHE = None


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class WebSearchRetryPolicy:
    """Retry policy with exponential backoff (reuses LLM retry strategy)."""

    max_retries: int = 1
    base_delay: float = 1.0
    max_delay: float = 5.0

    def should_retry(self, error: WebSearchError) -> bool:
        return isinstance(error, (ProviderTimeoutError, ProviderRateLimitError))

    async def backoff(self, attempt: int) -> None:
        """Exponential backoff with full-jitter (same strategy as LLM retry)."""
        import random

        delay = min(self.base_delay * (2**attempt), self.max_delay)
        jitter = random.uniform(0, delay)
        logger.info("web_search retry %d: backing off %.2fs", attempt + 1, jitter)
        await asyncio.sleep(jitter)


class WebSearchService:
    def __init__(
        self,
        *,
        provider: WebSearchProvider,
        fallback_providers: list[WebSearchProvider] | None = None,
        cache: LRUCache | None = None,
        cache_ttl: int | None = None,
        retry_policy: WebSearchRetryPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.fallback_providers = fallback_providers or []
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.retry_policy = retry_policy or WebSearchRetryPolicy()

    @classmethod
    def from_env(cls, *, provider: str | None = None) -> WebSearchService:
        explicit_provider = (provider or "").strip()
        env_provider = (os.getenv(ENV_WEB_SEARCH_PROVIDER) or "").strip()
        provider_name = explicit_provider or env_provider
        auto_detected = False

        if not provider_name:
            auto_detected = True
            if os.getenv(ENV_SERPER_API_KEY):
                provider_name = "serper"
            elif os.getenv(ENV_TAVILY_API_KEY):
                provider_name = "tavily"
            elif os.getenv(ENV_BOCHA_API_KEY):
                provider_name = "bocha"
            elif os.getenv(ENV_SEARXNG_BASE_URL):
                provider_name = "searxng"
            else:
                provider_name = "ddg"

        # --- Cache ---
        ttl_raw = (os.getenv(ENV_WEB_SEARCH_CACHE_TTL) or "").strip()
        ttl = int(ttl_raw) if ttl_raw else 300

        enabled_raw = os.getenv(ENV_WEB_SEARCH_CACHE_ENABLED)
        cache_enabled = True if enabled_raw is None else _is_truthy(enabled_raw)

        max_size_raw = (os.getenv(ENV_WEB_SEARCH_CACHE_MAX_SIZE) or "").strip()
        max_size = int(max_size_raw) if max_size_raw else 256

        cache: LRUCache | None = None
        if cache_enabled and ttl is not None:
            global _GLOBAL_CACHE
            if _GLOBAL_CACHE is None:
                _GLOBAL_CACHE = LRUCache(max_size=max_size, default_ttl=ttl)
            cache = _GLOBAL_CACHE

        # --- Proxy ---
        proxy_url: str | None = None
        if _is_truthy(os.getenv(ENV_WEB_SEARCH_PROXY_ENABLED, "false")):
            from houyi.net.proxy import detect_proxy

            proxy_url = detect_proxy()

        primary = cls._build_provider(provider_name, proxy_url=proxy_url)

        # Only build fallback chain when provider was auto-detected.
        fallbacks: list[WebSearchProvider] = []
        if auto_detected:
            fallback_order = ["serper", "tavily", "bocha", "ddg", "searxng"]
            for name in fallback_order:
                if name == provider_name:
                    continue
                try:
                    fallbacks.append(cls._build_provider(name, proxy_url=proxy_url))
                except (ProviderAuthError, DependencyMissingError, ValueError):
                    continue

        return cls(
            provider=primary,
            fallback_providers=fallbacks,
            cache=cache,
            cache_ttl=ttl,
        )

    @staticmethod
    def _build_provider(
        name: str,
        *,
        timeout: float | None = None,
        proxy_url: str | None = None,
    ) -> WebSearchProvider:
        normalised = (name or "").strip().lower()
        timeout_raw = (os.getenv(ENV_WEB_SEARCH_TIMEOUT) or "").strip()
        resolved_timeout = (
            timeout or (float(timeout_raw) if timeout_raw else None) or DEFAULT_PROVIDER_TIMEOUT
        )
        if normalised == "ddg":
            return DuckDuckGoWebSearchProvider(
                timeout=resolved_timeout,
                proxy_url=proxy_url,
            )
        if normalised == "searxng":
            return SearxNGWebSearchProvider(
                base_url=os.getenv(ENV_SEARXNG_BASE_URL),
                timeout=resolved_timeout,
                proxy_url=proxy_url,
            )
        if normalised == "tavily":
            return TavilyWebSearchProvider(
                api_key=os.getenv(ENV_TAVILY_API_KEY),
                timeout=resolved_timeout,
            )
        if normalised == "serper":
            return SerperWebSearchProvider(
                api_key=os.getenv(ENV_SERPER_API_KEY),
                timeout=resolved_timeout,
                proxy_url=proxy_url,
            )
        if normalised == "bocha":
            return BochaWebSearchProvider(
                api_key=os.getenv(ENV_BOCHA_API_KEY),
                timeout=resolved_timeout,
                proxy_url=proxy_url,
            )
        raise ValueError(f"Unsupported web search provider: {normalised}")

    def _resolve_providers(self) -> list[WebSearchProvider]:
        providers: list[WebSearchProvider] = [self.provider]
        providers.extend(self.fallback_providers)
        return providers

    def _cache_key(
        self, query: str, *, provider: str, max_results: int, include_content: bool
    ) -> str:
        return f"{query}|{provider}|{max_results}|{int(include_content)}"

    @staticmethod
    def _normalize_results(results: list[Any]) -> list[WebSearchResult]:
        normalized: list[WebSearchResult] = []
        for item in results:
            if isinstance(item, WebSearchResult):
                normalized.append(item)
            elif isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("snippet") or title).strip()
                if title and url:
                    normalized.append(
                        WebSearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            content=item.get("content"),
                        )
                    )
        return normalized

    async def _query_provider(
        self,
        provider: WebSearchProvider,
        query: str,
        *,
        max_results: int,
        errors: list[dict[str, Any]],
    ) -> tuple[list[WebSearchResult], dict[str, Any] | None, int | None, int]:
        provider_results: list[WebSearchResult] = []
        raw_payload: dict[str, Any] | None = None
        provider_latency_ms: int | None = None
        rate_limit_count = 0
        rate_limit_recorded = False
        attempt = 0
        provider_timeout = getattr(provider, "timeout", DEFAULT_PROVIDER_TIMEOUT)
        hard_timeout = provider_timeout * 1.5
        while True:
            attempt += 1
            span = _start_internal_span(
                f"provider.{provider.name}",
                {"provider.name": provider.name, "provider.attempt": attempt},
            )
            provider_start = time.monotonic()
            try:
                results = await asyncio.wait_for(
                    provider.search(query, max_results=max_results),
                    timeout=hard_timeout,
                )
                provider_latency_ms = int((time.monotonic() - provider_start) * 1000)
                raw = getattr(provider, "last_raw_payload", None)
                if isinstance(raw, dict):
                    raw_payload = {"provider_payload": raw}
                provider_results = self._normalize_results(results)
                if span:
                    span.set_attribute("provider.result_count", len(provider_results))
                    span.set_attribute("provider.latency_ms", provider_latency_ms)
                _end_span(span)
                break
            except TimeoutError:
                elapsed = int((time.monotonic() - provider_start) * 1000)
                msg = f"{provider.name} timeout after {elapsed}ms"
                _end_span(span, error=msg)
                errors.append(
                    {"type": "ProviderTimeoutError", "message": msg, "provider": provider.name}
                )
                if attempt <= self.retry_policy.max_retries:
                    await self.retry_policy.backoff(attempt - 1)
                    continue
                break
            except ProviderRateLimitError as exc:
                _end_span(span, error=str(exc))
                if not rate_limit_recorded:
                    rate_limit_count += 1
                    errors.append(
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "provider": provider.name,
                        }
                    )
                    rate_limit_recorded = True
                if attempt <= self.retry_policy.max_retries and self.retry_policy.should_retry(exc):
                    await self.retry_policy.backoff(attempt - 1)
                    continue
                break
            except ProviderTimeoutError as exc:
                _end_span(span, error=str(exc))
                errors.append(
                    {"type": type(exc).__name__, "message": str(exc), "provider": provider.name}
                )
                if attempt <= self.retry_policy.max_retries and self.retry_policy.should_retry(exc):
                    await self.retry_policy.backoff(attempt - 1)
                    continue
                break
            except ProviderError as exc:
                _end_span(span, error=str(exc))
                errors.append(
                    {"type": type(exc).__name__, "message": str(exc), "provider": provider.name}
                )
                break
            except WebSearchError as exc:
                _end_span(span, error=str(exc))
                errors.append(
                    {"type": type(exc).__name__, "message": str(exc), "provider": provider.name}
                )
                break

        return (provider_results, raw_payload, provider_latency_ms, rate_limit_count)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        include_content: bool = False,
        max_content_chars: int | None = None,
        use_cache: bool = True,
    ) -> WebSearchResponse:
        start = time.monotonic()
        errors: list[dict[str, Any]] = []

        providers = self._resolve_providers()
        primary_provider_name = providers[0].name if providers else "unknown"

        cached = False
        cache_hit = False
        cache_key = self._cache_key(
            query,
            provider=primary_provider_name,
            max_results=max_results,
            include_content=include_content,
        )
        logger.info(
            "web_search cache lookup: use_cache=%s cache_exists=%s cache_key=%s query=%r",
            use_cache,
            self.cache is not None,
            cache_key,
            query[:80],
        )
        if use_cache and self.cache is not None:
            cached_value = self.cache.get(cache_key)
            if isinstance(cached_value, WebSearchResponse):
                cached = True
                cache_hit = True
                resp = cached_value
                resp.metadata.cached = True
                resp.metadata.cache_hit = True
                resp.metadata.provider_latency_ms = 0
                resp.metadata.extraction_latency_ms = 0
                logger.info("web_search CACHE HIT: cache_key=%s", cache_key)
                return resp
            logger.info("web_search CACHE MISS: cache_key=%s", cache_key)

        provider_used: WebSearchProvider | None = None
        provider_results: list[WebSearchResult] = []
        raw_payload: dict[str, Any] | None = None
        provider_latency_ms: int | None = None
        rate_limit_count = 0

        for idx, provider in enumerate(providers):
            provider_used = provider
            if idx > 0:
                logger.info(
                    "web_search fallback: trying provider '%s' (attempt %d/%d)",
                    provider.name,
                    idx + 1,
                    len(providers),
                )
            (
                provider_results,
                raw_payload,
                provider_latency_ms,
                provider_rate_limits,
            ) = await self._query_provider(
                provider,
                query,
                max_results=max_results,
                errors=errors,
            )
            rate_limit_count += provider_rate_limits
            if provider_results:
                break

        extraction_provider: str | None = None
        extraction_latency_ms: int | None = None

        if not provider_results and not errors:
            errors.append(
                {
                    "type": "EmptyResults",
                    "message": "Provider returned no results",
                    "provider": (provider_used.name if provider_used else None),
                }
            )

        if include_content and provider_results:
            urls = [result.url for result in provider_results if result.url]
            extraction_start = time.monotonic()
            # Instrumentation: content fetch sub-span
            fetch_span = _start_internal_span(
                "content.fetch",
                {"content.url_count": len(urls)},
            )
            # Push fetch_span so child spans (jina/readability) attach to it
            _fetch_tc_token = None
            if fetch_span and _HAS_OBSERVABILITY:
                _fetch_tc_token = _TraceContext.push(fetch_span)
            try:
                jina = JinaContentFetcher()
                # Instrumentation: Jina fetch sub-span
                jina_span = _start_internal_span(
                    "fetch.jina",
                    {"fetch.provider": "jina", "fetch.url_count": len(urls)},
                )
                jina_results = await jina.fetch(urls)
                jina_hit_count = sum(1 for v in jina_results.values() if v)
                if jina_span:
                    jina_span.set_attribute("fetch.hit_count", jina_hit_count)
                _end_span(jina_span)
                extraction_provider = "jina" if any(jina_results.values()) else None
                for result in provider_results:
                    content = jina_results.get(result.url)
                    if content:
                        result.content = content.split("\n\n---\n", 1)[0]
                if not any(r.content for r in provider_results):
                    # Instrumentation: Readability fallback sub-span
                    read_span = _start_internal_span(
                        "fetch.readability",
                        {"fetch.provider": "readability", "fetch.url_count": len(urls)},
                    )
                    readability = ReadabilityContentFetcher()
                    extraction_provider = "readability"
                    read_results = await readability.fetch(urls)
                    read_hit_count = sum(1 for v in read_results.values() if v)
                    if read_span:
                        read_span.set_attribute("fetch.hit_count", read_hit_count)
                    _end_span(read_span)
                    for result in provider_results:
                        content = read_results.get(result.url)
                        if content:
                            result.content = content
            except DependencyMissingError as exc:
                errors.append(
                    {"type": type(exc).__name__, "message": str(exc), "provider": "readability"}
                )
                extraction_provider = "readability_unavailable"
            except ContentFetchError as exc:
                errors.append(
                    {"type": type(exc).__name__, "message": str(exc), "provider": "content"}
                )
            finally:
                extraction_latency_ms = int((time.monotonic() - extraction_start) * 1000)
                if fetch_span:
                    fetch_span.set_attribute("content.latency_ms", extraction_latency_ms)
                    fetch_span.set_attribute("content.provider", extraction_provider or "none")
                _end_span(fetch_span)
                if _fetch_tc_token is not None:
                    _TraceContext.pop(_fetch_tc_token)

        if max_content_chars and provider_results:
            for r in provider_results:
                if r.content and len(r.content) > max_content_chars:
                    r.content = r.content[:max_content_chars] + "..."

        latency_ms = int((time.monotonic() - start) * 1000)
        provider_name = provider_used.name if provider_used else self.provider.name
        metadata = WebSearchMetadata(
            cached=cached,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
            provider=provider_name,
            provider_latency_ms=provider_latency_ms,
            extraction_latency_ms=extraction_latency_ms,
            extraction_provider=extraction_provider,
            error_count=len(errors),
            rate_limit_count=rate_limit_count,
            errors=errors,
        )

        response = WebSearchResponse(
            query=query,
            provider=provider_name,
            results=provider_results,
            metadata=metadata,
            raw=raw_payload,
        )

        if use_cache and self.cache is not None and provider_results:
            ttl = self.cache_ttl if self.cache_ttl is not None else self.cache.default_ttl
            self.cache.put(cache_key, response, ttl=ttl)

        return response
