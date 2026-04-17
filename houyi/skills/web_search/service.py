from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from houyi.assurance.verification.cache import LRUCache
from houyi.infrastructure.config.env_config import (
    ENV_BOCHA_API_KEY,
    ENV_SEARXNG_BASE_URL,
    ENV_SERPER_API_KEY,
    ENV_TAVILY_API_KEY,
    ENV_WEB_SEARCH_CACHE_ENABLED,
    ENV_WEB_SEARCH_CACHE_MAX_SIZE,
    ENV_WEB_SEARCH_CACHE_TTL,
    ENV_WEB_SEARCH_PROVIDER,
    ENV_WEB_SEARCH_TIMEOUT,
)
from houyi.infrastructure.net.proxy import ProxyResolution, resolve_web_search_proxy
from houyi.skills.web_search.content_fetchers import JinaContentFetcher, ReadabilityContentFetcher
from houyi.skills.web_search.errors import (
    ContentFetchError,
    DependencyMissingError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    WebSearchError,
)
from houyi.skills.web_search.provider_resolution import (
    normalize_web_search_provider_name,
    resolve_supported_web_search_provider,
)
from houyi.skills.web_search.providers import (
    DEFAULT_PROVIDER_TIMEOUT,
    BochaWebSearchProvider,
    DuckDuckGoWebSearchProvider,
    SearxNGWebSearchProvider,
    SerperWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchProvider,
)
from houyi.skills.web_search.types import WebSearchMetadata, WebSearchResponse, WebSearchResult

logger = logging.getLogger(__name__)

# Observability: optional auto-instrumentation for tool-internal sub-spans.
# Import is safe — observability module is in the same SDK layer.
try:
    from houyi.infrastructure.observability.context import TraceContext as _TraceContext
    from houyi.infrastructure.observability.trace_manager import Span as _Span
    from houyi.infrastructure.observability.types import SpanType as _SpanType

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


def _summarize_provider_errors(errors: list[dict[str, Any]]) -> str:
    """Build a compact, deterministic summary string for provider failures."""
    if not errors:
        return "none"
    grouped: dict[str, list[str]] = {}
    for item in errors:
        provider = str(item.get("provider") or "unknown")
        err_type = str(item.get("type") or "UnknownError")
        grouped.setdefault(provider, []).append(err_type)
    parts: list[str] = []
    for provider, types in grouped.items():
        parts.append(f"{provider}:{','.join(types[:4])}")
    return "; ".join(parts)


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


def _normalize_provider_name(value: str | None) -> str:
    return normalize_web_search_provider_name(value)


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


@dataclass(slots=True)
class _ResolvedEnvConfig:
    provider_name: str
    provider_source: str
    auto_detected: bool
    env_provider_set: bool
    cache: LRUCache | None
    cache_ttl: int
    proxy_policy: str
    proxy_url: str | None
    proxy_source: str


@dataclass(slots=True)
class _ProviderQueryResult:
    results: list[WebSearchResult]
    raw_payload: dict[str, Any] | None
    latency_ms: int | None


@dataclass(slots=True)
class _SearchExecutionResult:
    provider_used: WebSearchProvider | None
    provider_results: list[WebSearchResult]
    raw_payload: dict[str, Any] | None
    provider_latency_ms: int | None
    rate_limit_count: int


@dataclass(slots=True)
class _ContentFetchResult:
    extraction_provider: str | None
    extraction_latency_ms: int | None


@dataclass(slots=True)
class _ProviderErrorOutcome:
    should_retry: bool
    rate_limit_increment: int = 0
    rate_limit_recorded: bool = False


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
        resolved = cls._resolve_env_config(provider)
        primary = cls._build_provider(
            resolved.provider_name,
            proxy=ProxyResolution(
                policy=resolved.proxy_policy,
                proxy_url=resolved.proxy_url,
                proxy_source=resolved.proxy_source,
            ),
        )
        fallbacks = cls._build_fallback_providers(
            resolved.provider_name,
            auto_detected=resolved.auto_detected,
            proxy=ProxyResolution(
                policy=resolved.proxy_policy,
                proxy_url=resolved.proxy_url,
                proxy_source=resolved.proxy_source,
            ),
        )

        logger.info(
            "web_search: primary=%s proxy_policy=%s proxy_source=%s",
            resolved.provider_name,
            resolved.proxy_policy,
            resolved.proxy_source,
        )
        logger.debug(
            "web_search init detail: source=%s auto_detected=%s WEB_SEARCH_PROVIDER_set=%s "
            "proxy_url=%s fallback_chain=%s",
            resolved.provider_source,
            resolved.auto_detected,
            resolved.env_provider_set,
            bool(resolved.proxy_url),
            [p.name for p in fallbacks],
        )

        return cls(
            provider=primary,
            fallback_providers=fallbacks,
            cache=resolved.cache,
            cache_ttl=resolved.cache_ttl,
        )

    @classmethod
    def _resolve_env_config(cls, provider: str | None) -> _ResolvedEnvConfig:
        explicit_provider = (provider or "").strip()
        env_provider = (os.getenv(ENV_WEB_SEARCH_PROVIDER) or "").strip()
        auto_detected = not bool(explicit_provider or env_provider)
        provider_name = explicit_provider or env_provider or cls._auto_detect_provider()
        provider_name = cls._resolve_supported_provider_name(provider_name)
        provider_source = "tool_input" if explicit_provider else ("env" if env_provider else "auto")
        cache_ttl = cls._resolve_cache_ttl()
        cache = cls._resolve_cache(cache_ttl)
        proxy = cls._resolve_proxy()
        return _ResolvedEnvConfig(
            provider_name=provider_name,
            provider_source=provider_source,
            auto_detected=auto_detected,
            env_provider_set=bool(env_provider),
            cache=cache,
            cache_ttl=cache_ttl,
            proxy_policy=proxy.policy,
            proxy_url=proxy.proxy_url,
            proxy_source=proxy.proxy_source,
        )

    @staticmethod
    def _auto_detect_provider() -> str:
        if os.getenv(ENV_SERPER_API_KEY):
            return "serper"
        if os.getenv(ENV_TAVILY_API_KEY):
            return "tavily"
        if os.getenv(ENV_BOCHA_API_KEY):
            return "bocha"
        return "ddg"

    @classmethod
    def _resolve_supported_provider_name(cls, provider_name: str | None) -> str:
        normalized = _normalize_provider_name(provider_name)
        fallback = resolve_supported_web_search_provider(
            normalized,
            fallback_provider=cls._auto_detect_provider(),
        )
        if normalized == fallback:
            return fallback

        logger.warning(
            "web_search provider '%s' is unsupported; falling back to '%s'",
            normalized or "(empty)",
            fallback,
        )
        return fallback

    @staticmethod
    def _resolve_cache_ttl() -> int:
        ttl_raw = (os.getenv(ENV_WEB_SEARCH_CACHE_TTL) or "").strip()
        return int(ttl_raw) if ttl_raw else 3600

    @classmethod
    def _resolve_cache(cls, ttl: int) -> LRUCache | None:
        enabled_raw = os.getenv(ENV_WEB_SEARCH_CACHE_ENABLED)
        cache_enabled = True if enabled_raw is None else _is_truthy(enabled_raw)
        if not cache_enabled:
            return None

        max_size_raw = (os.getenv(ENV_WEB_SEARCH_CACHE_MAX_SIZE) or "").strip()
        max_size = int(max_size_raw) if max_size_raw else 256
        global _GLOBAL_CACHE
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = LRUCache(max_size=max_size, default_ttl=ttl)
        return _GLOBAL_CACHE

    @staticmethod
    def _resolve_proxy() -> ProxyResolution:
        return resolve_web_search_proxy()

    @classmethod
    def _build_fallback_providers(
        cls,
        provider_name: str,
        *,
        auto_detected: bool,
        proxy: ProxyResolution,
    ) -> list[WebSearchProvider]:
        if not auto_detected:
            return []

        fallbacks: list[WebSearchProvider] = []
        for name in ["serper", "tavily", "bocha", "ddg"]:
            if name == provider_name:
                continue
            try:
                fallbacks.append(cls._build_provider(name, proxy=proxy))
            except (ProviderAuthError, DependencyMissingError, ValueError):
                continue
        return fallbacks

    @staticmethod
    def _build_provider(
        name: str,
        *,
        timeout: float | None = None,
        proxy: ProxyResolution,
    ) -> WebSearchProvider:
        normalised = (name or "").strip().lower()
        timeout_raw = (os.getenv(ENV_WEB_SEARCH_TIMEOUT) or "").strip()
        resolved_timeout = (
            timeout or (float(timeout_raw) if timeout_raw else None) or DEFAULT_PROVIDER_TIMEOUT
        )
        if normalised == "ddg":
            return DuckDuckGoWebSearchProvider(
                timeout=resolved_timeout,
                proxy_url=proxy.proxy_url,
                proxy_policy=proxy.policy,
            )
        if normalised == "searxng":
            return SearxNGWebSearchProvider(
                base_url=os.getenv(ENV_SEARXNG_BASE_URL),
                timeout=resolved_timeout,
                proxy_url=proxy.proxy_url,
                proxy_policy=proxy.policy,
                proxy_source=proxy.proxy_source,
            )
        if normalised == "tavily":
            return TavilyWebSearchProvider(
                api_key=os.getenv(ENV_TAVILY_API_KEY),
                timeout=resolved_timeout,
                proxy_url=proxy.proxy_url,
                proxy_policy=proxy.policy,
            )
        if normalised == "serper":
            return SerperWebSearchProvider(
                api_key=os.getenv(ENV_SERPER_API_KEY),
                timeout=resolved_timeout,
                proxy_url=proxy.proxy_url,
                proxy_policy=proxy.policy,
                proxy_source=proxy.proxy_source,
            )
        if normalised == "bocha":
            return BochaWebSearchProvider(
                api_key=os.getenv(ENV_BOCHA_API_KEY),
                timeout=resolved_timeout,
                proxy_url=proxy.proxy_url,
                proxy_policy=proxy.policy,
                proxy_source=proxy.proxy_source,
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
        query_result = _ProviderQueryResult(results=[], raw_payload=None, latency_ms=None)
        rate_limit_count = 0
        rate_limit_recorded = False
        attempt = 0
        provider_timeout = getattr(provider, "timeout", DEFAULT_PROVIDER_TIMEOUT)
        hard_timeout = provider_timeout * 1.5
        while True:
            attempt += 1
            try:
                query_result = await self._query_provider_once(
                    provider,
                    query,
                    max_results=max_results,
                    hard_timeout=hard_timeout,
                    attempt=attempt,
                )
                break
            except (TimeoutError, ProviderError, WebSearchError) as exc:
                outcome = await self._handle_provider_exception(
                    errors,
                    provider_name=provider.name,
                    exc=exc,
                    attempt=attempt,
                    hard_timeout=hard_timeout,
                    rate_limit_recorded=rate_limit_recorded,
                )
                rate_limit_count += outcome.rate_limit_increment
                rate_limit_recorded = rate_limit_recorded or outcome.rate_limit_recorded
                if outcome.should_retry:
                    continue
                break

        return (
            query_result.results,
            query_result.raw_payload,
            query_result.latency_ms,
            rate_limit_count,
        )

    async def _query_provider_once(
        self,
        provider: WebSearchProvider,
        query: str,
        *,
        max_results: int,
        hard_timeout: float,
        attempt: int,
    ) -> _ProviderQueryResult:
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
            latency_ms = int((time.monotonic() - provider_start) * 1000)
            raw = getattr(provider, "last_raw_payload", None)
            raw_payload = {"provider_payload": raw} if isinstance(raw, dict) else None
            normalized = self._normalize_results(results)
            if span:
                span.set_attribute("provider.result_count", len(normalized))
                span.set_attribute("provider.latency_ms", latency_ms)
            _end_span(span)
            return _ProviderQueryResult(
                results=normalized,
                raw_payload=raw_payload,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            _end_span(span, error=str(exc))
            raise

    @staticmethod
    def _record_error(
        errors: list[dict[str, Any]],
        provider_name: str,
        *,
        error_type: str,
        message: str,
    ) -> None:
        errors.append({"type": error_type, "message": message, "provider": provider_name})

    async def _handle_retryable_error(
        self,
        errors: list[dict[str, Any]],
        provider_name: str,
        exc: WebSearchError,
        *,
        attempt: int,
    ) -> bool:
        self._record_error(errors, provider_name, error_type=type(exc).__name__, message=str(exc))
        if attempt <= self.retry_policy.max_retries and self.retry_policy.should_retry(exc):
            await self.retry_policy.backoff(attempt - 1)
            return True
        return False

    async def _handle_provider_exception(
        self,
        errors: list[dict[str, Any]],
        *,
        provider_name: str,
        exc: Exception,
        attempt: int,
        hard_timeout: float,
        rate_limit_recorded: bool,
    ) -> _ProviderErrorOutcome:
        if isinstance(exc, TimeoutError):
            self._record_error(
                errors,
                provider_name,
                error_type="ProviderTimeoutError",
                message=f"{provider_name} timeout after {int(hard_timeout * 1000)}ms",
            )
            if attempt <= self.retry_policy.max_retries:
                await self.retry_policy.backoff(attempt - 1)
                return _ProviderErrorOutcome(should_retry=True)
            return _ProviderErrorOutcome(should_retry=False)

        if isinstance(exc, ProviderRateLimitError):
            if not rate_limit_recorded:
                self._record_error(
                    errors,
                    provider_name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                if attempt <= self.retry_policy.max_retries and self.retry_policy.should_retry(exc):
                    await self.retry_policy.backoff(attempt - 1)
                    return _ProviderErrorOutcome(
                        should_retry=True,
                        rate_limit_increment=1,
                        rate_limit_recorded=True,
                    )
                return _ProviderErrorOutcome(
                    should_retry=False,
                    rate_limit_increment=1,
                    rate_limit_recorded=True,
                )
            if attempt <= self.retry_policy.max_retries and self.retry_policy.should_retry(exc):
                await self.retry_policy.backoff(attempt - 1)
                return _ProviderErrorOutcome(should_retry=True)
            return _ProviderErrorOutcome(should_retry=False)

        if isinstance(exc, ProviderTimeoutError):
            should_retry = await self._handle_retryable_error(
                errors,
                provider_name,
                exc,
                attempt=attempt,
            )
            return _ProviderErrorOutcome(should_retry=should_retry)

        if isinstance(exc, (ProviderError, WebSearchError)):
            self._record_error(
                errors, provider_name, error_type=type(exc).__name__, message=str(exc)
            )
            return _ProviderErrorOutcome(should_retry=False)

        raise exc

    def _get_cached_response(
        self,
        cache_key: str,
        *,
        query: str,
        use_cache: bool,
    ) -> WebSearchResponse | None:
        logger.info(
            "web_search cache lookup: use_cache=%s cache_exists=%s cache_key=%s query=%r",
            use_cache,
            self.cache is not None,
            cache_key,
            query[:80],
        )
        if not use_cache or self.cache is None:
            return None

        cached_value = self.cache.get(cache_key)
        if not isinstance(cached_value, WebSearchResponse):
            logger.info("web_search CACHE MISS: cache_key=%s", cache_key)
            return None

        cached_value.metadata.cached = True
        cached_value.metadata.cache_hit = True
        cached_value.metadata.provider_latency_ms = 0
        cached_value.metadata.extraction_latency_ms = 0
        logger.info("web_search CACHE HIT: cache_key=%s", cache_key)
        return cached_value

    async def _execute_provider_chain(
        self,
        providers: list[WebSearchProvider],
        query: str,
        *,
        max_results: int,
        errors: list[dict[str, Any]],
    ) -> _SearchExecutionResult:
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

        return _SearchExecutionResult(
            provider_used=provider_used,
            provider_results=provider_results,
            raw_payload=raw_payload,
            provider_latency_ms=provider_latency_ms,
            rate_limit_count=rate_limit_count,
        )

    async def _fetch_content_if_needed(
        self,
        provider_results: list[WebSearchResult],
        *,
        include_content: bool,
        errors: list[dict[str, Any]],
    ) -> _ContentFetchResult:
        if not include_content or not provider_results:
            return _ContentFetchResult(extraction_provider=None, extraction_latency_ms=None)

        urls = [result.url for result in provider_results if result.url]
        extraction_start = time.monotonic()
        fetch_span = _start_internal_span("content.fetch", {"content.url_count": len(urls)})
        fetch_token = None
        if fetch_span and _HAS_OBSERVABILITY:
            fetch_token = _TraceContext.push(fetch_span)

        extraction_provider: str | None = None
        try:
            extraction_provider = await self._apply_jina_content(provider_results, urls)
            if not any(r.content for r in provider_results):
                extraction_provider = await self._apply_readability_content(provider_results, urls)
        except DependencyMissingError as exc:
            self._record_error(
                errors,
                "readability",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            extraction_provider = "readability_unavailable"
        except ContentFetchError as exc:
            self._record_error(
                errors,
                "content",
                error_type=type(exc).__name__,
                message=str(exc),
            )
        finally:
            extraction_latency_ms = int((time.monotonic() - extraction_start) * 1000)
            if fetch_span:
                fetch_span.set_attribute("content.latency_ms", extraction_latency_ms)
                fetch_span.set_attribute("content.provider", extraction_provider or "none")
            _end_span(fetch_span)
            if fetch_token is not None:
                _TraceContext.pop(fetch_token)

        return _ContentFetchResult(
            extraction_provider=extraction_provider,
            extraction_latency_ms=extraction_latency_ms,
        )

    async def _apply_jina_content(
        self,
        provider_results: list[WebSearchResult],
        urls: list[str],
    ) -> str | None:
        jina = JinaContentFetcher()
        jina_span = _start_internal_span(
            "fetch.jina",
            {"fetch.provider": "jina", "fetch.url_count": len(urls)},
        )
        jina_results = await jina.fetch(urls)
        jina_hit_count = sum(1 for value in jina_results.values() if value)
        if jina_span:
            jina_span.set_attribute("fetch.hit_count", jina_hit_count)
        _end_span(jina_span)

        for result in provider_results:
            content = jina_results.get(result.url)
            if content:
                result.content = content.split("\n\n---\n", 1)[0]

        return "jina" if any(jina_results.values()) else None

    async def _apply_readability_content(
        self,
        provider_results: list[WebSearchResult],
        urls: list[str],
    ) -> str:
        read_span = _start_internal_span(
            "fetch.readability",
            {"fetch.provider": "readability", "fetch.url_count": len(urls)},
        )
        readability = ReadabilityContentFetcher()
        read_results = await readability.fetch(urls)
        read_hit_count = sum(1 for value in read_results.values() if value)
        if read_span:
            read_span.set_attribute("fetch.hit_count", read_hit_count)
        _end_span(read_span)
        for result in provider_results:
            content = read_results.get(result.url)
            if content:
                result.content = content
        return "readability"

    @staticmethod
    def _apply_content_length_limit(
        provider_results: list[WebSearchResult],
        max_content_chars: int | None,
    ) -> None:
        if not max_content_chars:
            return
        for result in provider_results:
            if result.content and len(result.content) > max_content_chars:
                result.content = result.content[:max_content_chars] + "..."

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
        cache_key = self._cache_key(
            query,
            provider=primary_provider_name,
            max_results=max_results,
            include_content=include_content,
        )

        cached_response = self._get_cached_response(cache_key, query=query, use_cache=use_cache)
        if cached_response is not None:
            return cached_response

        execution_result = await self._execute_provider_chain(
            providers,
            query,
            max_results=max_results,
            errors=errors,
        )

        if not execution_result.provider_results:
            logger.warning(
                "web_search empty results: query=%r providers=%s error_summary=%s",
                query[:160],
                [provider.name for provider in providers],
                _summarize_provider_errors(errors),
            )

        if not execution_result.provider_results and not errors:
            errors.append(
                {
                    "type": "EmptyResults",
                    "message": "Provider returned no results",
                    "provider": (
                        execution_result.provider_used.name
                        if execution_result.provider_used
                        else None
                    ),
                }
            )

        content_fetch_result = await self._fetch_content_if_needed(
            execution_result.provider_results,
            include_content=include_content,
            errors=errors,
        )

        self._apply_content_length_limit(execution_result.provider_results, max_content_chars)

        latency_ms = int((time.monotonic() - start) * 1000)
        provider_name = (
            execution_result.provider_used.name
            if execution_result.provider_used
            else self.provider.name
        )
        metadata = WebSearchMetadata(
            cached=False,
            cache_hit=False,
            latency_ms=latency_ms,
            provider=provider_name,
            provider_latency_ms=execution_result.provider_latency_ms,
            extraction_latency_ms=content_fetch_result.extraction_latency_ms,
            extraction_provider=content_fetch_result.extraction_provider,
            error_count=len(errors),
            rate_limit_count=execution_result.rate_limit_count,
            errors=errors,
        )

        response = WebSearchResponse(
            query=query,
            provider=provider_name,
            results=execution_result.provider_results,
            metadata=metadata,
            raw=execution_result.raw_payload,
        )

        if use_cache and self.cache is not None and execution_result.provider_results:
            ttl = self.cache_ttl if self.cache_ttl is not None else self.cache.default_ttl
            self.cache.put(cache_key, response, ttl=ttl)
            logger.info(
                "web_search CACHE PUT: cache_key=%s ttl=%ds entries=%d",
                cache_key,
                ttl,
                self.cache.stats.total_size,
            )

        return response
