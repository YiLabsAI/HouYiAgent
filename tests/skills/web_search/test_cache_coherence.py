"""Cache coherence tests for WebSearchService.

Covers:
* Provider-agnostic cache keying (fallback responses remain reachable).
* Minimum-results PUT gate (thin responses do not poison the cache).
* Public stats snapshot surface for benchmark observability.
"""

from __future__ import annotations

import pytest

from houyi.assurance.verification.cache import LRUCache
from houyi.infrastructure.config.env_config import (
    ENV_WEB_SEARCH_CACHE_MIN_RESULTS,
)
from houyi.skills.web_search.service import (
    WebSearchService,
    _reset_global_cache_for_tests,
    get_global_cache_stats,
)


class _Provider:
    def __init__(self, *, results=None):
        self.name = "tavily"
        self._results = results or []

    async def search(self, query, *, max_results):
        return self._results


def _sample_results(count: int) -> list[dict]:
    return [{"title": f"t{idx}", "url": f"https://example.com/{idx}"} for idx in range(count)]


class TestCacheKey:
    def test_key_omits_provider(self):
        svc = WebSearchService(provider=_Provider())
        key = svc._cache_key("q", max_results=5, include_content=True)
        assert "tavily" not in key
        assert key == "q|5|1"

    def test_key_shape(self):
        svc = WebSearchService(provider=_Provider())
        assert svc._cache_key("q", max_results=3, include_content=False) == "q|3|0"

    def test_key_varies_max(self):
        svc = WebSearchService(provider=_Provider())
        key_a = svc._cache_key("q", max_results=1, include_content=False)
        key_b = svc._cache_key("q", max_results=2, include_content=False)
        assert key_a != key_b


class TestMinResultsGate:
    @pytest.mark.asyncio
    async def test_blocks_thin_response(self, monkeypatch):
        monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_MIN_RESULTS, "3")
        provider = _Provider(results=_sample_results(2))
        cache = LRUCache(max_size=10, default_ttl=60)
        service = WebSearchService(provider=provider, cache=cache, cache_ttl=60)

        first = await service.search("q", max_results=10)
        second = await service.search("q", max_results=10)
        assert first.metadata.cache_hit is False
        assert second.metadata.cache_hit is False

    @pytest.mark.asyncio
    async def test_min_results_gate(self, monkeypatch):
        monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_MIN_RESULTS, "3")
        provider = _Provider(results=_sample_results(3))
        cache = LRUCache(max_size=10, default_ttl=60)
        service = WebSearchService(provider=provider, cache=cache, cache_ttl=60)

        await service.search("q", max_results=10)
        second = await service.search("q", max_results=10)
        assert second.metadata.cache_hit is True

    @pytest.mark.asyncio
    async def test_env_disables_gate(self, monkeypatch):
        monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_MIN_RESULTS, "1")
        provider = _Provider(results=_sample_results(1))
        cache = LRUCache(max_size=10, default_ttl=60)
        service = WebSearchService(provider=provider, cache=cache, cache_ttl=60)

        await service.search("q", max_results=10)
        second = await service.search("q", max_results=10)
        assert second.metadata.cache_hit is True

    def test_default_is_three(self, monkeypatch):
        monkeypatch.delenv(ENV_WEB_SEARCH_CACHE_MIN_RESULTS, raising=False)
        assert WebSearchService._resolve_cache_min_results() == 3

    def test_invalid_env_ignored(self, monkeypatch):
        monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_MIN_RESULTS, "not-a-number")
        assert WebSearchService._resolve_cache_min_results() == 3

    def test_zero_env_ignored(self, monkeypatch):
        monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_MIN_RESULTS, "0")
        assert WebSearchService._resolve_cache_min_results() == 3


class TestCacheStatsSnapshot:
    def test_none_without_cache(self, monkeypatch):
        _reset_global_cache_for_tests()
        assert get_global_cache_stats() is None

    def test_none_when_cold(self, monkeypatch):
        _reset_global_cache_for_tests()
        # Building a service via constructor does not materialize the
        # global cache; snapshot should stay None until the first use.
        WebSearchService(provider=_Provider())
        assert get_global_cache_stats() is None

    def test_reflects_hits_misses(self, monkeypatch):
        monkeypatch.setenv(ENV_WEB_SEARCH_CACHE_MIN_RESULTS, "1")
        # Attach a custom cache to the global slot so the public snapshot
        # reads the same object the service writes to.
        from houyi.skills.web_search import service as service_module

        _reset_global_cache_for_tests()
        custom = LRUCache(max_size=8, default_ttl=60)
        service_module._GLOBAL_CACHE = custom

        custom.put("k1", {"payload": True}, ttl=60)
        custom.get("k1")
        custom.get("missing")

        snapshot = get_global_cache_stats()
        assert snapshot is not None
        assert snapshot["hits"] == 1
        assert snapshot["misses"] == 1
        assert snapshot["hit_rate"] == 0.5
        assert snapshot["entries"] == 1
        assert snapshot["max_size"] == 8

        _reset_global_cache_for_tests()
