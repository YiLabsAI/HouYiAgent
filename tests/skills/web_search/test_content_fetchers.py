"""Unit tests for web search content fetchers."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

import pytest

from houyi.skills.web_search.content_fetchers import JinaContentFetcher, ReadabilityContentFetcher
from houyi.skills.web_search.errors import ContentFetchError, DependencyMissingError


def test_readability_fetcher_missing_dependency(monkeypatch) -> None:
    """ReadabilityContentFetcher should fail without dependencies."""

    monkeypatch.setitem(__import__("sys").modules, "readability", None)
    monkeypatch.setitem(__import__("sys").modules, "bs4", None)
    with pytest.raises(DependencyMissingError):
        ReadabilityContentFetcher()


@pytest.mark.asyncio
async def test_readability_fetcher_success(monkeypatch) -> None:
    """ReadabilityContentFetcher should extract text content."""

    class _Doc:
        def __init__(self, html: str) -> None:
            self._html = html

        def summary(self) -> str:
            return "<div>Hello</div>"

    class _Soup:
        def __init__(self, html: str, _parser: str) -> None:
            self._html = html

        def get_text(self, _sep: str, strip: bool = True) -> str:
            return "Hello"

    monkeypatch.setitem(sys.modules, "readability", type("_mod", (), {"Document": _Doc}))
    monkeypatch.setitem(sys.modules, "bs4", type("_mod", (), {"BeautifulSoup": _Soup}))

    class _Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def _fake_open(*_args, **_kwargs):
        return _Response(b"<html></html>")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_open)
    fetcher = ReadabilityContentFetcher()
    result = await fetcher.fetch(["https://example.com"])
    assert result["https://example.com"] == "Hello"


@pytest.mark.asyncio
async def test_readability_fetcher_error(monkeypatch) -> None:
    """ReadabilityContentFetcher should raise on URL errors."""

    monkeypatch.setitem(sys.modules, "readability", type("_mod", (), {"Document": object}))
    monkeypatch.setitem(sys.modules, "bs4", type("_mod", (), {"BeautifulSoup": object}))

    def _fail(*_args, **_kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    fetcher = ReadabilityContentFetcher()
    with pytest.raises(ContentFetchError):
        await fetcher.fetch(["https://example.com"])


@pytest.mark.asyncio
async def test_jina_fetcher_reads_content(monkeypatch) -> None:
    """JinaContentFetcher should return content for URL."""

    class _Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def _fake_open(*_args, **_kwargs):
        return _Response(b"content")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_open)
    fetcher = JinaContentFetcher()
    result = await fetcher.fetch(["example.com"])
    assert result["example.com"] == "content"


@pytest.mark.asyncio
async def test_jina_fetcher_error(monkeypatch) -> None:
    """JinaContentFetcher should raise ContentFetchError on URL error."""

    def _fail(*_args, **_kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    fetcher = JinaContentFetcher()
    with pytest.raises(ContentFetchError):
        await fetcher.fetch(["example.com"])
