from __future__ import annotations

import asyncio
import importlib
import re
from urllib import request
from urllib.error import URLError
from urllib.parse import quote, urlparse, urlunparse

from houyi.skills.web_search.errors import ContentFetchError, DependencyMissingError


def _encode_url_for_http(url: str) -> str:
    """Percent-encode non-ASCII characters in a URL for safe HTTP transmission."""
    parsed = urlparse(url)
    encoded_path = quote(parsed.path, safe="/:@!$&'()*+,;=-._~")
    encoded_query = quote(parsed.query, safe="/:@!$&'()*+,;=-._~?=")
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            encoded_path,
            parsed.params,
            encoded_query,
            parsed.fragment,
        )
    )


class JinaContentFetcher:
    def __init__(self, *, api_key: str | None = None) -> None:
        from houyi.infrastructure.config.env_config import EnvConfig

        self._api_key = api_key or EnvConfig.get().jina_api_key or ""
        if self._api_key:
            self._endpoint = "https://r.jina.ai/"
        else:
            self._endpoint = "https://r.jina.ai/http/"

    async def fetch(self, urls: list[str]) -> dict[str, str]:
        def _fetch_one(url: str) -> tuple[str, str]:
            target = url.strip()
            if not target:
                return (url, "")
            if not re.match(r"^https?://", target):
                target = f"https://{target}"
            if self._api_key:
                jina_url = _encode_url_for_http(f"{self._endpoint}{target}")
                req = request.Request(
                    jina_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            else:
                jina_url = _encode_url_for_http(f"{self._endpoint}{target}")
                req = request.Request(jina_url)
            try:
                with request.urlopen(req, timeout=30) as response:
                    payload = response.read().decode("utf-8", errors="ignore")
            except (URLError, TimeoutError, OSError) as exc:
                raise ContentFetchError(f"Jina fetch failed: {exc}") from exc
            return (url, payload)

        results = await asyncio.gather(*[asyncio.to_thread(_fetch_one, url) for url in urls])
        return dict(results)


class ReadabilityContentFetcher:
    def __init__(self) -> None:
        try:
            importlib.import_module("bs4")
            importlib.import_module("readability")
        except ImportError as exc:
            raise DependencyMissingError(
                "Missing optional dependency 'readability-lxml' or 'beautifulsoup4'. Install: pip install 'houyi[websearch-readability]'"
            ) from exc

    async def fetch(self, urls: list[str]) -> dict[str, str]:
        from bs4 import BeautifulSoup
        from readability import Document

        def _fetch_one(url: str) -> tuple[str, str]:
            try:
                encoded_url = _encode_url_for_http(url)
                with request.urlopen(encoded_url, timeout=30) as response:
                    html = response.read().decode("utf-8", errors="ignore")
            except (URLError, TimeoutError, OSError) as exc:
                raise ContentFetchError(f"Readability fetch failed: {exc}") from exc
            doc = Document(html)
            summary = doc.summary()
            try:
                from markdownify import markdownify as md

                return (url, md(summary, strip=["img"]))
            except ImportError:
                soup = BeautifulSoup(summary, "html.parser")
                return (url, soup.get_text("\n", strip=True))

        results = await asyncio.gather(*[asyncio.to_thread(_fetch_one, url) for url in urls])
        return dict(results)
