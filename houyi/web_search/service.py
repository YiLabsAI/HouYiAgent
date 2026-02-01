from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    DuckDuckGoWebSearchProvider,
    SearxNGWebSearchProvider,
    SerperWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchProvider,
)
from houyi.web_search.types import WebSearchMetadata, WebSearchResponse, WebSearchResult

_GLOBAL_CACHE: LRUCache | None = None


def _reset_global_cache_for_tests() -> None:
    global _GLOBAL_CACHE
    _GLOBAL_CACHE = None


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class WebSearchRetryPolicy:
    max_retries: int = 0
    min_delay: float = 0.2
    max_delay: float = 2.0
    jitter: float = 0.1

    def should_retry(self, error: WebSearchError) -> bool:
        return isinstance(error, (ProviderTimeoutError, ProviderRateLimitError))


class WebSearchService:
    def __init__(
        self,
        *,
        provider: WebSearchProvider,
        fallback_providers: list[WebSearchProvider] | None = None,
        cache: LRUCache | None = None,
        cache_ttl: int | None = None,
        retry_policy: WebSearchRetryPolicy | None = None,
        sleep_func: Callable[[float], Any] | None = None,
    ) -> None:
        self.provider = provider
        self.fallback_providers = fallback_providers or []
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.retry_policy = retry_policy or WebSearchRetryPolicy(max_retries=0)
        self._sleep = sleep_func or asyncio.sleep

    @classmethod
    def from_env(cls, *, provider: str | None = None) -> WebSearchService:
        provider_name = (
            (provider or "").strip() or (os.getenv("WEB_SEARCH_PROVIDER") or "").strip() or "ddg"
        )

        ttl_raw = (os.getenv("WEB_SEARCH_CACHE_TTL") or "").strip()
        ttl = int(ttl_raw) if ttl_raw else None

        enabled_raw = os.getenv("WEB_SEARCH_CACHE_ENABLED")
        cache_enabled = True if enabled_raw is None else _is_truthy(enabled_raw)

        max_size_raw = (os.getenv("WEB_SEARCH_CACHE_MAX_SIZE") or "").strip()
        max_size = int(max_size_raw) if max_size_raw else 256

        cache: LRUCache | None = None
        if cache_enabled and ttl is not None:
            global _GLOBAL_CACHE
            if _GLOBAL_CACHE is None:
                _GLOBAL_CACHE = LRUCache(max_size=max_size, default_ttl=ttl)
            cache = _GLOBAL_CACHE

        primary = cls._build_provider(provider_name)
        return cls(provider=primary, cache=cache, cache_ttl=ttl)

    @staticmethod
    def _build_provider(name: str) -> WebSearchProvider:
        normalized = (name or "").strip().lower()
        if normalized == "ddg":
            return DuckDuckGoWebSearchProvider()
        if normalized == "searxng":
            base_url = os.getenv("SEARXNG_BASE_URL")
            return SearxNGWebSearchProvider(base_url=base_url)
        if normalized == "tavily":
            api_key = os.getenv("TAVILY_API_KEY")
            return TavilyWebSearchProvider(api_key=api_key)
        if normalized == "serper":
            api_key = os.getenv("SERPER_API_KEY")
            return SerperWebSearchProvider(api_key=api_key)
        raise ValueError(f"Unsupported web search provider: {normalized}")

    def _resolve_providers(self) -> list[WebSearchProvider]:
        providers: list[WebSearchProvider] = [self.provider]
        providers.extend(self.fallback_providers)

        resolved: list[WebSearchProvider] = []
        for provider in providers:
            try:
                if provider.name in {"serper", "tavily"}:
                    resolved.append(provider)
                elif provider.name == "searxng":
                    resolved.append(provider)
                else:
                    resolved.append(provider)
            except ProviderAuthError:
                continue
        return resolved

    def _cache_key(self, query: str, *, max_results: int, include_content: bool) -> str:
        return f"{self.provider.name}|{query}|{max_results}|{int(include_content)}"

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
        while True:
            attempt += 1
            provider_start = time.monotonic()
            try:
                results = await provider.search(query, max_results=max_results)
                provider_latency_ms = int((time.monotonic() - provider_start) * 1000)
                raw = getattr(provider, "last_raw_payload", None)
                if isinstance(raw, dict):
                    raw_payload = {"provider_payload": raw}
                provider_results = self._normalize_results(results)
                break
            except ProviderRateLimitError as exc:
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
                    await self._sleep(0.0)
                    continue
                break
            except ProviderTimeoutError as exc:
                errors.append(
                    {"type": type(exc).__name__, "message": str(exc), "provider": provider.name}
                )
                if attempt <= self.retry_policy.max_retries and self.retry_policy.should_retry(exc):
                    await self._sleep(0.0)
                    continue
                break
            except ProviderError as exc:
                errors.append(
                    {"type": type(exc).__name__, "message": str(exc), "provider": provider.name}
                )
                break
            except WebSearchError as exc:
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
        use_cache: bool = True,
    ) -> WebSearchResponse:
        start = time.monotonic()
        errors: list[dict[str, Any]] = []

        cached = False
        cache_hit = False
        cache_key = self._cache_key(query, max_results=max_results, include_content=include_content)
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
                return resp

        providers = self._resolve_providers()
        provider_used: WebSearchProvider | None = None
        provider_results: list[WebSearchResult] = []
        raw_payload: dict[str, Any] | None = None
        provider_latency_ms: int | None = None
        rate_limit_count = 0

        for provider in providers:
            provider_used = provider
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
            try:
                jina = JinaContentFetcher()
                jina_results = await jina.fetch(urls)
                extraction_provider = "jina" if any(jina_results.values()) else None
                for result in provider_results:
                    content = jina_results.get(result.url)
                    if content:
                        result.content = content.split("\n\n---\n", 1)[0]
                if not any(r.content for r in provider_results):
                    readability = ReadabilityContentFetcher()
                    extraction_provider = "readability"
                    read_results = await readability.fetch(urls)
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
