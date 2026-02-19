"""Web search providers."""

from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field
from http.client import RemoteDisconnected
from typing import Any, Protocol
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from houyi.web_search.errors import (
    DependencyMissingError,
    ProviderAuthError,
    ProviderInvalidResponse,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from houyi.web_search.types import WebSearchResult

DEFAULT_PROVIDER_TIMEOUT: float = 5.0


class WebSearchProvider(Protocol):
    """Provider interface for web search."""

    name: str

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        """Run search query and return normalized results."""


class WebSearchProviderRegistry:
    """Registry for web search providers."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., WebSearchProvider]] = {}

    def register(self, name: str, factory: Callable[..., WebSearchProvider]) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs) -> WebSearchProvider:
        if name not in self._factories:
            raise ValueError(f"Unsupported web search provider: {name}")
        return self._factories[name](**kwargs)

    def list(self) -> list[str]:
        return sorted(self._factories.keys())


@dataclass(slots=True)
class TavilyWebSearchProvider:
    """Tavily provider implementation."""

    name: str = "tavily"
    api_key: str | None = None
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    _client: object = field(init=False, repr=False)
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderAuthError("TAVILY_API_KEY is required")
        try:
            from tavily import TavilyClient  # type: ignore
        except ImportError as exc:
            raise DependencyMissingError(
                "Missing optional dependency 'tavily-python'. Install: pip install 'houyi[websearch-tavily]'"
            ) from exc
        self._client = TavilyClient(api_key=self.api_key)

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        """Search Tavily and normalize results."""

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
        normalized: list[WebSearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("content") or item.get("snippet") or title).strip()
            if not title or not url:
                continue
            normalized.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=item.get("content"),
                    score=item.get("score"),
                    source=self.name,
                    citations=None,
                )
            )
        return normalized


@dataclass(slots=True)
class SearxNGWebSearchProvider:
    """SearxNG provider implementation."""

    name: str = "searxng"
    base_url: str | None = None
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ProviderAuthError("SEARXNG_BASE_URL is required")

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        """Search SearxNG and normalize results."""

        base_url = self.base_url.rstrip("/")  # type: ignore[union-attr]
        params = urlencode({"q": query, "format": "json"})

        def _request() -> dict:
            url = f"{base_url}/search?{params}"
            headers = {
                "User-Agent": "houyi/console-web-search",
                "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
            }
            req = request.Request(url, headers=headers, method="GET")
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 429:
                    raise ProviderRateLimitError("SearxNG rate limited") from exc
                if exc.code in (401, 403):
                    raise ProviderAuthError("SearxNG unauthorized") from exc
                if 400 <= exc.code < 500:
                    raise ProviderInvalidResponse(f"SearxNG request failed: {exc}") from exc
                raise ProviderTimeoutError(f"SearxNG request failed: {exc}") from exc
            except (RemoteDisconnected, URLError, TimeoutError) as exc:
                raise ProviderTimeoutError(f"SearxNG request failed: {exc}") from exc

        response = await asyncio.to_thread(_request)
        if isinstance(response, dict):
            self.last_raw_payload = response
        results = response.get("results") if isinstance(response, dict) else None
        if results is None:
            raise ProviderInvalidResponse("SearxNG response missing results")

        normalized: list[WebSearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("content") or title).strip()
            if not title or not url:
                continue
            normalized.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=item.get("content") or None,
                    score=item.get("score"),
                    source=self.name,
                    citations=None,
                )
            )
        return normalized[:max_results]


@dataclass(slots=True)
class DuckDuckGoWebSearchProvider:
    """DuckDuckGo provider implementation (no key required)."""

    name: str = "ddg"
    endpoint: str = "https://api.duckduckgo.com/"
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        """Search DuckDuckGo and normalize results."""

        params = urlencode(
            {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "t": "houyi",
            }
        )

        def _request() -> dict:
            url = f"{self.endpoint}?{params}"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "close",
                "Cache-Control": "no-cache",
            }
            req = request.Request(url, headers=headers, method="GET")
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 429:
                    raise ProviderRateLimitError("DDG rate limited") from exc
                if exc.code in (401, 403):
                    raise ProviderAuthError("DDG unauthorized") from exc
                if 400 <= exc.code < 500:
                    raise ProviderInvalidResponse(f"DDG request failed: {exc}") from exc
                raise ProviderTimeoutError(f"DDG request failed: {exc}") from exc
            except (
                RemoteDisconnected,
                URLError,
                TimeoutError,
                ConnectionResetError,
                ssl.SSLError,
                OSError,
            ) as exc:
                raise ProviderTimeoutError(f"DDG request failed: {exc}") from exc

        response = await asyncio.to_thread(_request)
        if isinstance(response, dict):
            self.last_raw_payload = response
        topics = response.get("RelatedTopics") if isinstance(response, dict) else None
        if topics is None:
            raise ProviderInvalidResponse("DDG response missing RelatedTopics")

        results: list[WebSearchResult] = []
        seen: set[str] = set()

        def _extract(items: list[dict[str, Any]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                nested = item.get("Topics")
                if isinstance(nested, list):
                    _extract(nested)
                    continue
                text = (item.get("Text") or "").strip()
                url = (item.get("FirstURL") or "").strip()
                if not text or not url:
                    continue
                title = text.split(" - ", 1)[0].strip()
                dedupe_key = f"{url.lower()}|{title.lower()}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                results.append(
                    WebSearchResult(
                        title=title,
                        url=url,
                        snippet=text,
                        content=None,
                        score=None,
                        source=self.name,
                        citations=None,
                    )
                )

        _extract(topics)
        return results[:max_results]


@dataclass(slots=True)
class SerperWebSearchProvider:
    """Serper provider implementation."""

    name: str = "serper"
    api_key: str | None = None
    endpoint: str = "https://google.serper.dev/search"
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
    last_raw_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderAuthError("SERPER_API_KEY is required")

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        """Search Serper and normalize results."""

        payload = json.dumps({"q": query, "num": max_results}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
            "User-Agent": "houyi/console-web-search",
            "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
        }

        def _request() -> dict:
            req = request.Request(self.endpoint, data=payload, headers=headers, method="POST")  # type: ignore[arg-type]
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 429:
                    raise ProviderRateLimitError("Serper rate limited") from exc
                if exc.code in (401, 403):
                    raise ProviderAuthError("Serper unauthorized") from exc
                if 400 <= exc.code < 500:
                    raise ProviderInvalidResponse(f"Serper request failed: {exc}") from exc
                raise ProviderTimeoutError(f"Serper request failed: {exc}") from exc
            except (RemoteDisconnected, URLError, TimeoutError) as exc:
                raise ProviderTimeoutError(f"Serper request failed: {exc}") from exc

        response = await asyncio.to_thread(_request)
        if isinstance(response, dict):
            self.last_raw_payload = response
        results = response.get("organic") if isinstance(response, dict) else None
        if results is None:
            raise ProviderInvalidResponse("Serper response missing organic results")
        normalized: list[WebSearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("link") or "").strip()
            snippet = (item.get("snippet") or title).strip()
            if not title or not url:
                continue
            normalized.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=item.get("snippet") or None,
                    score=None,
                    source=self.name,
                    published_at=item.get("date"),
                    citations=None,
                )
            )
        return normalized


def build_default_provider_registry() -> WebSearchProviderRegistry:
    registry = WebSearchProviderRegistry()
    registry.register("ddg", DuckDuckGoWebSearchProvider)
    registry.register("searxng", SearxNGWebSearchProvider)
    registry.register("tavily", TavilyWebSearchProvider)
    registry.register("serper", SerperWebSearchProvider)
    return registry
