"""Web search providers with shared HTTP infrastructure.

Each provider implements the ``WebSearchProvider`` protocol.  Shared concerns:

* **HTTP error mapping** — ``_http_json_request()`` translates HTTP status codes
  to typed ``ProviderError`` subclasses so the retry / fallback machinery in
  ``WebSearchService`` can react uniformly.
* **Proxy support** — every HTTP-based provider accepts an optional ``proxy_url``
  that is wired through ``urllib.request.ProxyHandler``.
* **Timeout enforcement** — per-provider ``timeout`` field with sensible default.

Adding a new provider:
    1. Create a ``@dataclass(slots=True)`` with ``name``, ``timeout``,
       ``proxy_url``, and ``last_raw_payload`` fields.
    2. Implement ``async def search(…) -> list[WebSearchResult]``.
    3. Register it in ``build_default_provider_registry()``.
    4. Add the ``elif`` branch in ``WebSearchService._build_provider()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field
from http.client import RemoteDisconnected
from typing import Any, Protocol
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from houyi.skills.web_search.errors import (
    DependencyMissingError,
    ProviderAuthError,
    ProviderInvalidResponse,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from houyi.skills.web_search.types import WebSearchResult

DEFAULT_PROVIDER_TIMEOUT: float = 10.0

_DEFAULT_USER_AGENT = "houyi/web-search"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class WebSearchProvider(Protocol):
    """Minimal provider contract — any object with ``name`` + ``search()``."""

    name: str

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        """Run search query and return normalised results."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class WebSearchProviderRegistry:
    """Named-factory registry for web search providers."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., WebSearchProvider]] = {}

    def register(self, name: str, factory: Callable[..., WebSearchProvider]) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> WebSearchProvider:
        if name not in self._factories:
            raise ValueError(f"Unsupported web search provider: {name}")
        return self._factories[name](**kwargs)

    def list(self) -> list[str]:
        return sorted(self._factories.keys())


# ---------------------------------------------------------------------------
# Shared HTTP infrastructure
# ---------------------------------------------------------------------------


def _http_json_request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    method: str = "GET",
    timeout: float = DEFAULT_PROVIDER_TIMEOUT,
    proxy_url: str | None = None,
    label: str = "Provider",
) -> dict[str, Any]:
    """Execute HTTP request → parse JSON → map errors to ``ProviderError``.

    All HTTP-based providers funnel requests through this function so that
    proxy wiring, timeout enforcement, and HTTP-to-ProviderError mapping are
    consistent across the board.
    """
    final_headers: dict[str, str] = {
        "User-Agent": _DEFAULT_USER_AGENT,
        "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
    }
    if headers:
        final_headers.update(headers)

    req = request.Request(url, data=data, headers=final_headers, method=method)

    try:
        if proxy_url:
            proxy_handler = request.ProxyHandler({"http": proxy_url, "https": proxy_url})
            opener = request.build_opener(proxy_handler)
            resp_ctx = opener.open(req, timeout=timeout)
        else:
            resp_ctx = request.urlopen(req, timeout=timeout)
        with resp_ctx as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = ""
        with contextlib.suppress(Exception):
            body = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code == 429:
            raise ProviderRateLimitError(f"{label} rate limited") from exc
        if exc.code in (401, 403):
            raise ProviderAuthError(
                f"{label} unauthorized (HTTP {exc.code}): {body}"
                if body
                else f"{label} unauthorized"
            ) from exc
        if 400 <= exc.code < 500:
            raise ProviderInvalidResponse(
                f"{label} request failed: HTTP {exc.code}: {body}"
                if body
                else f"{label} request failed: HTTP {exc.code}"
            ) from exc
        raise ProviderTimeoutError(f"{label} request failed: HTTP {exc.code}") from exc
    except (
        RemoteDisconnected,
        URLError,
        TimeoutError,
        ConnectionResetError,
        ssl.SSLError,
        OSError,
    ) as exc:
        raise ProviderTimeoutError(f"{label} request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Serper (Google via serper.dev)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SerperWebSearchProvider:
    """Serper.dev provider — Google search results via REST API."""

    name: str = "serper"
    api_key: str | None = None
    endpoint: str = "https://google.serper.dev/search"
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    proxy_url: str | None = None
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderAuthError("SERPER_API_KEY is required")

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        payload = json.dumps({"q": query, "num": max_results}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key or "",
        }

        response = await asyncio.to_thread(
            _http_json_request,
            self.endpoint,
            headers=headers,
            data=payload,
            method="POST",
            timeout=self.timeout,
            proxy_url=self.proxy_url,
            label=self.name,
        )
        self.last_raw_payload = response if isinstance(response, dict) else None
        results = response.get("organic") if isinstance(response, dict) else None
        if results is None:
            raise ProviderInvalidResponse("Serper response missing organic results")

        normalised: list[WebSearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("link") or "").strip()
            snippet = (item.get("snippet") or title).strip()
            if not title or not url:
                continue
            normalised.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=item.get("snippet") or None,
                    source=self.name,
                    published_at=item.get("date"),
                )
            )
        return normalised


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TavilyWebSearchProvider:
    """Tavily provider — uses the ``tavily-python`` SDK.

    Proxy support is limited to environment variables (``HTTP_PROXY`` /
    ``HTTPS_PROXY``) because the Tavily SDK controls its own HTTP client.
    """

    name: str = "tavily"
    api_key: str | None = None
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    _client: object = field(init=False, repr=False)
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderAuthError("TAVILY_API_KEY is required")
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise DependencyMissingError(
                "Missing optional dependency 'tavily-python'. "
                "Install: pip install 'houyi[websearch-tavily]'"
            ) from exc
        self._client = TavilyClient(api_key=self.api_key)

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        response = await asyncio.to_thread(
            self._client.search,  # type: ignore[attr-defined]
            query,
            max_results=max_results,
            search_depth="basic",
        )
        if isinstance(response, dict):
            self.last_raw_payload = response
        results = response.get("results") if isinstance(response, dict) else None
        if results is None:
            raise ProviderInvalidResponse("Tavily response missing results")

        normalised: list[WebSearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("content") or item.get("snippet") or title).strip()
            if not title or not url:
                continue
            normalised.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=item.get("content"),
                    score=item.get("score"),
                    source=self.name,
                )
            )
        return normalised


# ---------------------------------------------------------------------------
# SearxNG
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SearxNGWebSearchProvider:
    """SearxNG meta-search engine (self-hosted, no API key required)."""

    name: str = "searxng"
    base_url: str | None = None
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    proxy_url: str | None = None
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ProviderAuthError("SEARXNG_BASE_URL is required")

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        base = (self.base_url or "").rstrip("/")
        params = urlencode({"q": query, "format": "json"})
        url = f"{base}/search?{params}"

        response = await asyncio.to_thread(
            _http_json_request,
            url,
            timeout=self.timeout,
            proxy_url=self.proxy_url,
            label=self.name,
        )
        if isinstance(response, dict):
            self.last_raw_payload = response
        results = response.get("results") if isinstance(response, dict) else None
        if results is None:
            raise ProviderInvalidResponse("SearxNG response missing results")

        normalised: list[WebSearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("content") or title).strip()
            if not title or not url:
                continue
            normalised.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=item.get("content") or None,
                    score=item.get("score"),
                    source=self.name,
                )
            )
        return normalised[:max_results]


# ---------------------------------------------------------------------------
# DuckDuckGo (real HTML search via duckduckgo-search)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DuckDuckGoWebSearchProvider:
    """DuckDuckGo meta-search via the ``ddgs`` library.

    Requires optional dependency: ``pip install 'houyi[websearch-ddg]'``.
    The dependency check is deferred to ``search()`` so that provider
    instances can be created without the library present.

    ``ddgs`` exposes a sync-only ``DDGS`` class; the blocking call is
    wrapped with ``asyncio.to_thread`` to avoid stalling the event loop.
    """

    name: str = "ddg"
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    proxy_url: str | None = None
    region: str = "wt-wt"
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise DependencyMissingError(
                "Missing optional dependency 'ddgs'. Install: pip install 'houyi[websearch-ddg]'"
            ) from exc

        try:
            raw_results = await self._do_search(DDGS, query, max_results)
        except DependencyMissingError:
            raise
        except Exception as exc:
            exc_name = type(exc).__name__
            if "RatelimitE" in exc_name or "429" in str(exc):
                raise ProviderRateLimitError(f"DDG rate limited: {exc}") from exc
            raise ProviderTimeoutError(f"DDG search failed: {exc}") from exc

        if isinstance(raw_results, list):
            self.last_raw_payload = {"results": raw_results}

        results: list[WebSearchResult] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("href") or "").strip()
            snippet = (item.get("body") or "").strip()
            if not title or not url:
                continue
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=item.get("body"),
                    source=self.name,
                )
            )
        return results[:max_results]

    async def _do_search(self, ddgs_cls: type, query: str, max_results: int) -> list:
        """Run the sync DDGS.text() in a thread to keep the event loop free."""
        import asyncio

        def _sync_search() -> list:
            ddgs = ddgs_cls(proxy=self.proxy_url, timeout=int(self.timeout))
            return ddgs.text(query, max_results=max_results, region=self.region)

        return await asyncio.to_thread(_sync_search)


# ---------------------------------------------------------------------------
# Bocha — Chinese-friendly search API with free tier
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BochaWebSearchProvider:
    """Bocha search API — optimised for Chinese-language queries.

    API docs: https://open.bochaai.com
    Free tier: 100 queries/day.
    Response: ``{"code": 200, "data": {"webPages": {"value": [...]}}}``.
    """

    name: str = "bocha"
    api_key: str | None = None
    endpoint: str = "https://api.bochaai.com/v1/web-search"
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    proxy_url: str | None = None
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderAuthError("BOCHA_API_KEY is required")

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        payload = json.dumps(
            {
                "query": query,
                "freshness": "noLimit",
                "summary": True,
                "count": max_results,
            }
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = await asyncio.to_thread(
            _http_json_request,
            self.endpoint,
            headers=headers,
            data=payload,
            method="POST",
            timeout=self.timeout,
            proxy_url=self.proxy_url,
            label=self.name,
        )
        self.last_raw_payload = response if isinstance(response, dict) else None

        data = response.get("data", response) if isinstance(response, dict) else {}
        if not isinstance(data, dict):
            data = {}
        web_pages = data.get("webPages", data)
        if not isinstance(web_pages, dict):
            web_pages = {}
        items = web_pages.get("value", [])
        if not isinstance(items, list):
            raise ProviderInvalidResponse("Bocha response missing webPages.value")

        normalised: list[WebSearchResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = (item.get("name") or item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("snippet") or item.get("description") or "").strip()
            if not title or not url:
                continue
            normalised.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=item.get("summary") or None,
                    source=self.name,
                    published_at=item.get("datePublished") or item.get("dateLastCrawled"),
                )
            )
        return normalised[:max_results]


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def build_default_provider_registry() -> WebSearchProviderRegistry:
    registry = WebSearchProviderRegistry()
    registry.register("ddg", DuckDuckGoWebSearchProvider)
    registry.register("tavily", TavilyWebSearchProvider)
    registry.register("serper", SerperWebSearchProvider)
    registry.register("bocha", BochaWebSearchProvider)
    registry.register("searxng", SearxNGWebSearchProvider)
    return registry
