"""Unit tests for web search providers."""

from __future__ import annotations

import json
import sys

import pytest

from houyi.web_search.errors import (
    DependencyMissingError,
    ProviderAuthError,
    ProviderInvalidResponse,
)
from houyi.web_search.providers import (
    DuckDuckGoWebSearchProvider,
    SearxNGWebSearchProvider,
    SerperWebSearchProvider,
    TavilyWebSearchProvider,
)


def test_tavily_provider_requires_key() -> None:
    """Provider should fail without API key."""

    with pytest.raises(ProviderAuthError):
        TavilyWebSearchProvider(api_key=None)


def test_serper_provider_requires_key() -> None:
    """Provider should fail without API key."""

    with pytest.raises(ProviderAuthError):
        SerperWebSearchProvider(api_key=None)


def test_tavily_provider_missing_dependency(monkeypatch) -> None:
    """Provider should raise when dependency is missing."""

    monkeypatch.setitem(sys.modules, "tavily", None)
    with pytest.raises(DependencyMissingError):
        TavilyWebSearchProvider(api_key="test")


@pytest.mark.asyncio
async def test_tavily_provider_search_normalizes(monkeypatch) -> None:
    """Provider should normalize Tavily response into WebSearchResult."""

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args

        def search(self, *_args, **_kwargs):
            return {"results": [{"title": "t", "url": "u", "content": "c", "score": 0.5}]}

    monkeypatch.setitem(
        sys.modules,
        "tavily",
        type("_mod", (), {"TavilyClient": _Client}),
    )

    provider = TavilyWebSearchProvider(api_key="test")
    results = await provider.search("query", max_results=1)
    assert results[0].title == "t"
    assert results[0].snippet == "c"


@pytest.mark.asyncio
async def test_tavily_provider_invalid_response(monkeypatch) -> None:
    """Provider should raise on invalid response payload."""

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def search(self, *_args, **_kwargs):
            return {"unexpected": []}

    monkeypatch.setitem(
        sys.modules,
        "tavily",
        type("_mod", (), {"TavilyClient": _Client}),
    )

    provider = TavilyWebSearchProvider(api_key="test")
    with pytest.raises(ProviderInvalidResponse):
        await provider.search("query", max_results=1)


def test_searxng_provider_requires_base_url() -> None:
    """Provider should fail without base URL."""

    with pytest.raises(ProviderAuthError):
        SearxNGWebSearchProvider(base_url=None)


@pytest.mark.asyncio
async def test_searxng_provider_search_normalizes(monkeypatch) -> None:
    """Provider should normalize SearxNG response into WebSearchResult."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "content": "snippet",
                            "score": 0.5,
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "houyi.web_search.providers.request.urlopen", lambda *_args, **_kwargs: _Response()
    )

    provider = SearxNGWebSearchProvider(base_url="https://searx.local")
    results = await provider.search("query", max_results=1)
    assert results[0].title == "t"
    assert results[0].url == "https://example.com"
    assert results[0].snippet == "snippet"
    assert results[0].source == "searxng"


@pytest.mark.asyncio
async def test_searxng_provider_invalid_response(monkeypatch) -> None:
    """Provider should raise on invalid response payload."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"unexpected": []}).encode("utf-8")

    monkeypatch.setattr(
        "houyi.web_search.providers.request.urlopen", lambda *_args, **_kwargs: _Response()
    )

    provider = SearxNGWebSearchProvider(base_url="https://searx.local")
    with pytest.raises(ProviderInvalidResponse):
        await provider.search("query", max_results=1)


@pytest.mark.asyncio
async def test_ddg_provider_search_normalizes(monkeypatch) -> None:
    """Provider should normalize DDG response into WebSearchResult."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "RelatedTopics": [
                        {
                            "Text": "Example Result - Example",
                            "FirstURL": "https://example.com",
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "houyi.web_search.providers.request.urlopen", lambda *_args, **_kwargs: _Response()
    )

    provider = DuckDuckGoWebSearchProvider()
    results = await provider.search("query", max_results=1)
    assert results[0].title == "Example Result"
    assert results[0].url == "https://example.com"
    assert results[0].source == "ddg"


@pytest.mark.asyncio
async def test_ddg_provider_invalid_response(monkeypatch) -> None:
    """Provider should raise on invalid response payload."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"unexpected": []}).encode("utf-8")

    monkeypatch.setattr(
        "houyi.web_search.providers.request.urlopen", lambda *_args, **_kwargs: _Response()
    )

    provider = DuckDuckGoWebSearchProvider()
    with pytest.raises(ProviderInvalidResponse):
        await provider.search("query", max_results=1)


@pytest.mark.asyncio
async def test_ddg_provider_http_unauthorized_maps_to_auth_error(monkeypatch) -> None:
    """DDG 401/403 should surface as ProviderAuthError (non-retriable)."""

    from urllib.error import HTTPError

    def _urlopen(*_args, **_kwargs):
        raise HTTPError(
            url="https://api.duckduckgo.com/", code=403, msg="Forbidden", hdrs=None, fp=None
        )

    monkeypatch.setattr("houyi.web_search.providers.request.urlopen", _urlopen)

    provider = DuckDuckGoWebSearchProvider()
    with pytest.raises(ProviderAuthError):
        await provider.search("q", max_results=1)


@pytest.mark.asyncio
async def test_ddg_provider_http_4xx_maps_to_invalid_response(monkeypatch) -> None:
    """DDG 4xx (except 401/403/429) should surface as ProviderInvalidResponse (non-retriable)."""

    from urllib.error import HTTPError

    def _urlopen(*_args, **_kwargs):
        raise HTTPError(
            url="https://api.duckduckgo.com/", code=400, msg="Bad Request", hdrs=None, fp=None
        )

    monkeypatch.setattr("houyi.web_search.providers.request.urlopen", _urlopen)

    provider = DuckDuckGoWebSearchProvider()
    with pytest.raises(ProviderInvalidResponse):
        await provider.search("q", max_results=1)


@pytest.mark.asyncio
async def test_serper_provider_search_normalizes(monkeypatch) -> None:
    """Provider should normalize Serper response into WebSearchResult."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "organic": [
                        {
                            "title": "t",
                            "link": "u",
                            "snippet": "s",
                            "date": "2025-01-01",
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "houyi.web_search.providers.request.urlopen", lambda *_args, **_kwargs: _Response()
    )

    provider = SerperWebSearchProvider(api_key="test")
    results = await provider.search("query", max_results=1)
    assert results[0].title == "t"
    assert results[0].snippet == "s"
    assert results[0].source == "serper"


@pytest.mark.asyncio
async def test_serper_provider_invalid_response(monkeypatch) -> None:
    """Provider should raise on invalid response payload."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"unexpected": []}).encode("utf-8")

    monkeypatch.setattr(
        "houyi.web_search.providers.request.urlopen", lambda *_args, **_kwargs: _Response()
    )

    provider = SerperWebSearchProvider(api_key="test")
    with pytest.raises(ProviderInvalidResponse):
        await provider.search("query", max_results=1)
