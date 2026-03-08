from __future__ import annotations

import asyncio
import importlib
import re
from urllib import request
from urllib.error import URLError

from houyi.skills.web_search.errors import ContentFetchError, DependencyMissingError


class JinaContentFetcher:
    def __init__(self, *, endpoint: str = "https://r.jina.ai/http/") -> None:
        self._endpoint = endpoint

    async def fetch(self, urls: list[str]) -> dict[str, str]:
        def _fetch_one(url: str) -> tuple[str, str]:
            target = url.strip()
            if not target:
                return (url, "")
            if not re.match(r"^https?://", target):
                target = f"https://{target}"
            jina_url = f"{self._endpoint}{target}"
            try:
                with request.urlopen(jina_url, timeout=30) as response:
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
                with request.urlopen(url, timeout=30) as response:
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
