"""Integration test for L2 browse pipeline with real URL fetch."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from houyi.web_search.service import WebSearchService
from houyi.web_search.types import WebSearchResult


class _Provider:
    name = "ddg"

    def __init__(self, url: str) -> None:
        self._url = url

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        return [WebSearchResult(title="Browse Target", url=self._url, snippet="s")]


@pytest.mark.asyncio
async def test_web_search_browse_real_url() -> None:
    """browse mode should fetch real content via Jina/Readability."""

    load_dotenv()
    if os.getenv("BROWSE_INTEGRATION_TEST") != "1":
        pytest.skip("BROWSE_INTEGRATION_TEST not enabled; set to 1 to run L2 browse test")

    target_url = os.getenv("WEB_SEARCH_BROWSE_TEST_URL")
    if not target_url:
        pytest.skip("WEB_SEARCH_BROWSE_TEST_URL not set; requires real URL")

    service = WebSearchService(provider=_Provider(target_url))
    response = await service.search("browse", max_results=1, include_content=True)
    assert response.results
    assert response.results[0].content
