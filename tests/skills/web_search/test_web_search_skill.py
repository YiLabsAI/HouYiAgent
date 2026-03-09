"""Unit tests for web search skill builder."""

from __future__ import annotations

import pytest

from houyi.domain.skill.policy import ExecutionMode
from houyi.skills.web_search import skill as web_search_skill_module
from houyi.skills.web_search.skill import build_web_search_skill
from houyi.skills.web_search.types import WebSearchMetadata, WebSearchResponse, WebSearchResult


class _Service:
    def __init__(self) -> None:
        self.called = False
        self.include_content: bool | None = None
        self.use_cache: bool | None = None

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        include_content: bool = False,
        use_cache: bool = True,
    ) -> WebSearchResponse:
        self.called = True
        self.include_content = include_content
        self.use_cache = use_cache
        return WebSearchResponse(
            query=query,
            provider="ddg",
            results=[WebSearchResult(title="t", url="u", snippet="s")],
            metadata=WebSearchMetadata(
                cached=False,
                cache_hit=False,
                latency_ms=1,
                provider="ddg",
            ),
        )


class _ServiceWithoutUseCache:
    def __init__(self) -> None:
        self.called = False
        self.include_content: bool | None = None

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        include_content: bool = False,
    ) -> WebSearchResponse:
        self.called = True
        self.include_content = include_content
        return WebSearchResponse(
            query=query,
            provider="ddg",
            results=[WebSearchResult(title="t", url="u", snippet="s")],
            metadata=WebSearchMetadata(
                cached=False,
                cache_hit=False,
                latency_ms=1,
                provider="ddg",
            ),
        )


@pytest.mark.asyncio
async def test_build_web_search_skill_executes(monkeypatch) -> None:
    """build_web_search_skill should bind executor and return normalized output."""

    service = _Service()

    def _from_env(provider: str | None = None) -> _Service:
        assert provider is None
        return service

    monkeypatch.setattr(web_search_skill_module.WebSearchService, "from_env", _from_env)

    skill = build_web_search_skill()
    assert skill.execution_mode == ExecutionMode.PLUGIN
    result = await skill.executor(query="hi", max_results=1)
    assert service.called is True
    assert service.include_content is False
    assert result["provider"] == "ddg"
    assert result["metadata"]["cache_hit"] is False
    assert result["metadata"]["provider"] == "ddg"


@pytest.mark.asyncio
async def test_build_web_search_skill_browse_mode(monkeypatch) -> None:
    """browse mode should enable content extraction."""

    service = _Service()

    def _from_env(provider: str | None = None) -> _Service:
        assert provider is None
        return service

    monkeypatch.setattr(web_search_skill_module.WebSearchService, "from_env", _from_env)

    skill = build_web_search_skill()
    await skill.executor(query="hi", max_results=1, mode="browse")
    assert service.called is True
    assert service.include_content is True


@pytest.mark.asyncio
async def test_build_web_search_skill_provider_override(monkeypatch) -> None:
    """provider override should flow into service factory."""

    service = _Service()

    def _from_env(provider: str | None = None) -> _Service:
        assert provider == "searxng"
        return service

    monkeypatch.setattr(web_search_skill_module.WebSearchService, "from_env", _from_env)

    skill = build_web_search_skill()
    await skill.executor(query="hi", max_results=1, provider="searxng")
    assert service.called is True


@pytest.mark.asyncio
async def test_build_web_search_skill_use_cache_false(monkeypatch) -> None:
    service = _Service()

    def _from_env(provider: str | None = None) -> _Service:
        assert provider is None
        return service

    monkeypatch.setattr(web_search_skill_module.WebSearchService, "from_env", _from_env)

    skill = build_web_search_skill()
    await skill.executor(query="hi", max_results=1, use_cache=False)

    assert service.called is True
    assert service.use_cache is False


@pytest.mark.asyncio
async def test_build_web_search_skill_without_use_cache_param(monkeypatch) -> None:
    service = _ServiceWithoutUseCache()

    def _from_env(provider: str | None = None) -> _ServiceWithoutUseCache:
        assert provider is None
        return service

    monkeypatch.setattr(web_search_skill_module.WebSearchService, "from_env", _from_env)

    skill = build_web_search_skill()
    result = await skill.executor(query="hi", max_results=1, use_cache=False)

    assert service.called is True
    assert service.include_content is False
    assert result["provider"] == "ddg"
