"""Unit tests for web search providers."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest

from houyi.skills.web_search import providers as providers_module
from houyi.skills.web_search.errors import (
    DependencyMissingError,
    ProviderAuthError,
    ProviderInvalidResponse,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from houyi.skills.web_search.providers import (
    BochaWebSearchProvider,
    DuckDuckGoWebSearchProvider,
    SearxNGWebSearchProvider,
    SerperWebSearchProvider,
    TavilyWebSearchProvider,
    _http_json_request,
)

# ---------------------------------------------------------------------------
# Shared HTTP infrastructure
# ---------------------------------------------------------------------------


class TestHttpJsonRequest:
    """Tests for the shared _http_json_request function."""

    def test_success_get(self, monkeypatch) -> None:
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"ok": True}).encode()

        monkeypatch.setattr(
            providers_module.request,
            "urlopen",
            lambda *a, **k: _Resp(),
        )
        result = _http_json_request("https://example.com", label="test")
        assert result == {"ok": True}

    def test_429_maps_limit(self, monkeypatch) -> None:
        from urllib.error import HTTPError

        def _urlopen(*a, **k):
            raise HTTPError("", 429, "Rate Limited", {}, None)

        monkeypatch.setattr(providers_module.request, "urlopen", _urlopen)
        with pytest.raises(ProviderRateLimitError):
            _http_json_request("https://example.com", label="test")

    def test_403_maps_auth(self, monkeypatch) -> None:
        from urllib.error import HTTPError

        def _urlopen(*a, **k):
            raise HTTPError("", 403, "Forbidden", {}, None)

        monkeypatch.setattr(providers_module.request, "urlopen", _urlopen)
        with pytest.raises(ProviderAuthError):
            _http_json_request("https://example.com", label="test")

    def test_400_maps_invalid(self, monkeypatch) -> None:
        from urllib.error import HTTPError

        def _urlopen(*a, **k):
            raise HTTPError("", 400, "Bad Request", {}, None)

        monkeypatch.setattr(providers_module.request, "urlopen", _urlopen)
        with pytest.raises(ProviderInvalidResponse):
            _http_json_request("https://example.com", label="test")

    def test_timeout_maps_error(self, monkeypatch) -> None:
        def _urlopen(*a, **k):
            raise TimeoutError("timed out")

        monkeypatch.setattr(providers_module.request, "urlopen", _urlopen)
        with pytest.raises(ProviderTimeoutError):
            _http_json_request("https://example.com", label="test")

    def test_proxy_uses_opener(self, monkeypatch) -> None:
        opened_urls: list[str] = []

        class _FakeOpener:
            def open(self, req, *, timeout=None):
                opened_urls.append(req.full_url)

                class _Resp:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def read(self):
                        return b'{"proxied": true}'

                return _Resp()

        monkeypatch.setattr(
            providers_module.request,
            "build_opener",
            lambda *a: _FakeOpener(),
        )
        result = _http_json_request(
            "https://example.com",
            proxy_url="http://127.0.0.1:7890",
            label="test",
        )
        assert result == {"proxied": True}
        assert opened_urls == ["https://example.com"]


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


def test_tavily_requires_key() -> None:
    with pytest.raises(ProviderAuthError):
        TavilyWebSearchProvider(api_key=None)


def test_tavily_missing_dep(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tavily", None)
    with pytest.raises(DependencyMissingError):
        TavilyWebSearchProvider(api_key="test")


@pytest.mark.asyncio
async def test_tavily_normalizes(monkeypatch) -> None:
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
async def test_tavily_invalid(monkeypatch) -> None:
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


# ---------------------------------------------------------------------------
# Serper
# ---------------------------------------------------------------------------


def test_serper_requires_key() -> None:
    with pytest.raises(ProviderAuthError):
        SerperWebSearchProvider(api_key=None)


@pytest.mark.asyncio
async def test_serper_normalizes(monkeypatch) -> None:
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

    monkeypatch.setattr(providers_module.request, "urlopen", lambda *_args, **_kwargs: _Response())

    provider = SerperWebSearchProvider(api_key="test")
    results = await provider.search("query", max_results=1)
    assert results[0].title == "t"
    assert results[0].snippet == "s"
    assert results[0].source == "serper"


@pytest.mark.asyncio
async def test_serper_invalid(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"unexpected": []}).encode("utf-8")

    monkeypatch.setattr(providers_module.request, "urlopen", lambda *_args, **_kwargs: _Response())

    provider = SerperWebSearchProvider(api_key="test")
    with pytest.raises(ProviderInvalidResponse):
        await provider.search("query", max_results=1)


# ---------------------------------------------------------------------------
# SearxNG
# ---------------------------------------------------------------------------


def test_searxng_requires_url() -> None:
    with pytest.raises(ProviderAuthError):
        SearxNGWebSearchProvider(base_url=None)


@pytest.mark.asyncio
async def test_searxng_normalizes(monkeypatch) -> None:
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

    monkeypatch.setattr(providers_module.request, "urlopen", lambda *_args, **_kwargs: _Response())

    provider = SearxNGWebSearchProvider(base_url="https://searx.local")
    results = await provider.search("query", max_results=1)
    assert results[0].title == "t"
    assert results[0].url == "https://example.com"
    assert results[0].snippet == "snippet"
    assert results[0].source == "searxng"


@pytest.mark.asyncio
async def test_searxng_invalid(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"unexpected": []}).encode("utf-8")

    monkeypatch.setattr(providers_module.request, "urlopen", lambda *_args, **_kwargs: _Response())

    provider = SearxNGWebSearchProvider(base_url="https://searx.local")
    with pytest.raises(ProviderInvalidResponse):
        await provider.search("query", max_results=1)


# ---------------------------------------------------------------------------
# DuckDuckGo (duckduckgo-search library)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ddg_missing_dep(monkeypatch) -> None:
    provider = DuckDuckGoWebSearchProvider()

    real_import = __import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "ddgs":
            raise ImportError("No module named 'ddgs'")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_import):
        with pytest.raises(DependencyMissingError, match="ddgs"):
            await provider.search("query", max_results=3)


@pytest.mark.asyncio
async def test_ddg_normalizes(monkeypatch) -> None:
    ddgs_mod = types.SimpleNamespace(DDGS=object)
    with (
        patch.dict(sys.modules, {"ddgs": ddgs_mod}),
        patch.object(
            DuckDuckGoWebSearchProvider,
            "_do_search",
            new=AsyncMock(
                return_value=[
                    {"title": "Result 1", "href": "https://example.com/1", "body": "Body 1"},
                    {"title": "Result 2", "href": "https://example.com/2", "body": "Body 2"},
                ]
            ),
        ),
    ):
        provider = DuckDuckGoWebSearchProvider()
        results = await provider.search("test query", max_results=3)

    assert len(results) == 2
    assert results[0].title == "Result 1"
    assert results[0].url == "https://example.com/1"
    assert results[0].snippet == "Body 1"
    assert results[0].source == "ddg"


@pytest.mark.asyncio
async def test_ddg_empty_results(monkeypatch) -> None:
    ddgs_mod = types.SimpleNamespace(DDGS=object)
    with (
        patch.dict(sys.modules, {"ddgs": ddgs_mod}),
        patch.object(DuckDuckGoWebSearchProvider, "_do_search", new=AsyncMock(return_value=[])),
    ):
        provider = DuckDuckGoWebSearchProvider()
        results = await provider.search("obscure query", max_results=3)

    assert results == []


@pytest.mark.asyncio
async def test_ddg_rate_limit(monkeypatch) -> None:
    ddgs_mod = types.SimpleNamespace(DDGS=object)

    class RatelimitException(Exception):
        pass

    with (
        patch.dict(sys.modules, {"ddgs": ddgs_mod}),
        patch.object(
            DuckDuckGoWebSearchProvider,
            "_do_search",
            new=AsyncMock(side_effect=RatelimitException("429")),
        ),
    ):
        provider = DuckDuckGoWebSearchProvider()
        with pytest.raises(ProviderRateLimitError):
            await provider.search("q", max_results=1)


@pytest.mark.asyncio
async def test_ddg_maps_timeout(monkeypatch) -> None:
    ddgs_mod = types.SimpleNamespace(DDGS=object)
    with (
        patch.dict(sys.modules, {"ddgs": ddgs_mod}),
        patch.object(
            DuckDuckGoWebSearchProvider,
            "_do_search",
            new=AsyncMock(side_effect=TimeoutError("timed out")),
        ),
    ):
        provider = DuckDuckGoWebSearchProvider()
        with pytest.raises(ProviderTimeoutError):
            await provider.search("q", max_results=1)


@pytest.mark.asyncio
async def test_ddg_passes_proxy(monkeypatch) -> None:
    class _DDGS:
        def __init__(self, *, proxy: str | None, timeout: int):
            self.proxy = proxy
            self.timeout = timeout

        def text(self, _query: str, *, max_results: int, region: str):
            return []

    ddgs_mod = types.SimpleNamespace(DDGS=_DDGS)
    with patch.dict(sys.modules, {"ddgs": ddgs_mod}):
        provider = DuckDuckGoWebSearchProvider(proxy_url="http://127.0.0.1:7890")
        await provider.search("q", max_results=1)

    assert provider.last_raw_payload == {"results": []}


# ---------------------------------------------------------------------------
# Bocha (博查)
# ---------------------------------------------------------------------------


def test_bocha_requires_key() -> None:
    with pytest.raises(ProviderAuthError):
        BochaWebSearchProvider(api_key=None)


@pytest.mark.asyncio
async def test_bocha_normalizes(monkeypatch) -> None:
    response_data = {
        "code": 200,
        "data": {
            "webPages": {
                "value": [
                    {
                        "name": "Bocha Result",
                        "url": "https://example.cn/1",
                        "snippet": "Chinese result snippet",
                        "summary": "Full summary text",
                        "datePublished": "2025-01-01",
                    }
                ]
            }
        },
    }

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_data).encode("utf-8")

    monkeypatch.setattr(providers_module.request, "urlopen", lambda *a, **k: _Response())

    provider = BochaWebSearchProvider(api_key="test-key")
    results = await provider.search("测试查询", max_results=5)

    assert len(results) == 1
    assert results[0].title == "Bocha Result"
    assert results[0].url == "https://example.cn/1"
    assert results[0].snippet == "Chinese result snippet"
    assert results[0].content == "Full summary text"
    assert results[0].source == "bocha"
    assert results[0].published_at == "2025-01-01"


@pytest.mark.asyncio
async def test_bocha_invalid(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"code": 200, "data": {"unexpected": True}}).encode("utf-8")

    monkeypatch.setattr(providers_module.request, "urlopen", lambda *a, **k: _Response())

    provider = BochaWebSearchProvider(api_key="test-key")
    results = await provider.search("q", max_results=1)
    assert results == []


@pytest.mark.asyncio
async def test_bocha_passes_proxy(monkeypatch) -> None:
    """Proxy URL should be forwarded to the opener."""
    opened_with_proxy = {"called": False}

    class _FakeOpener:
        def open(self, req, *, timeout=None):
            opened_with_proxy["called"] = True

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps(
                        {
                            "code": 200,
                            "data": {"webPages": {"value": []}},
                        }
                    ).encode()

            return _Resp()

    monkeypatch.setattr(
        providers_module.request,
        "build_opener",
        lambda *a: _FakeOpener(),
    )

    provider = BochaWebSearchProvider(api_key="k", proxy_url="http://proxy:8080")
    await provider.search("q", max_results=1)
    assert opened_with_proxy["called"]
