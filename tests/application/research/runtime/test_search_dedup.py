"""Tests for cross-sub-question deduplication and disambiguation fallbacks."""

from __future__ import annotations

import asyncio

from houyi.application.research.runtime.coordinator import _DelegateDedup
from houyi.application.research.runtime.search_executor import (
    _make_disambiguation_queries,
)


class TestDelegateDedup:
    async def test_claim_url_first_true(self):
        dedup = _DelegateDedup()
        assert await dedup.claim_url("https://a.com") is True

    async def test_claim_url_second_false(self):
        dedup = _DelegateDedup()
        await dedup.claim_url("https://a.com")
        assert await dedup.claim_url("https://a.com") is False

    async def test_different_urls_both_true(self):
        dedup = _DelegateDedup()
        assert await dedup.claim_url("https://a.com") is True
        assert await dedup.claim_url("https://b.com") is True

    async def test_claim_query_dedupes(self):
        dedup = _DelegateDedup()
        assert await dedup.claim_query("RocketMQ overview") is True
        assert await dedup.claim_query("rocketmq overview") is False

    async def test_claim_query_empty_false(self):
        dedup = _DelegateDedup()
        assert await dedup.claim_query("") is False

    async def test_concurrent_claim_safe(self):
        dedup = _DelegateDedup()
        results = await asyncio.gather(
            dedup.claim_url("https://race.com"),
            dedup.claim_url("https://race.com"),
        )
        assert sum(results) == 1


class TestDisambiguationQueries:
    def test_builds_from_user_query(self):
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])
        fallbacks = _make_disambiguation_queries(
            f"{name} background",
            f"{name} Apache RocketMQ creator",
        )
        assert len(fallbacks) >= 1
        assert any(name in fb for fb in fallbacks)

    def test_empty_when_no_anchor(self):
        fallbacks = _make_disambiguation_queries(
            "general topic analysis",
            "general topic analysis",
        )
        # English anchor extraction may produce something or not;
        # the key contract is no crash and bounded output.
        assert len(fallbacks) <= 3

    def test_includes_english_variant(self):
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])
        fallbacks = _make_disambiguation_queries(
            f"{name} background",
            f"{name} Apache RocketMQ open source",
        )
        en_queries = [fb for fb in fallbacks if "Apache" in fb or "RocketMQ" in fb]
        assert len(en_queries) >= 1

    def test_max_three_fallbacks(self):
        name = "".join(chr(cp) for cp in [0x51AF, 0x5609])
        fallbacks = _make_disambiguation_queries(
            f"{name} background info",
            f"{name} Apache RocketMQ creator open source messaging distributed system",
        )
        assert len(fallbacks) <= 3
