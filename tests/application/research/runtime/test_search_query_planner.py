from __future__ import annotations

import json
from unittest.mock import AsyncMock

from houyi.application.research.runtime.search_query_planner import (
    QueryPlanner,
    _ensure_bilingual_queries,
)

from ..conftest import MockLLM


def _planner(**kwargs) -> QueryPlanner:
    return QueryPlanner(llm=MockLLM(), max_rounds=3, llm_kwargs={}, **kwargs)


def _non_english_query() -> str:
    return "".join(
        chr(codepoint) for codepoint in [0x667A, 0x80FD, 0x4F53, 0x6846, 0x67B6, 0x5BF9, 0x6BD4]
    )


class TestQueryPlanner:
    async def test_queries_bilingual(self):
        non_english_query = _non_english_query()
        planner = QueryPlanner(
            llm=MockLLM(responses=[json.dumps([f"{non_english_query} 2026"], ensure_ascii=False)]),
            max_rounds=3,
            llm_kwargs={},
        )
        queries, metadata = await planner.generate_queries(
            f"{non_english_query} HouYi LangChain agent framework",
            f"{non_english_query} HouYi LangChain agent framework",
            [],
            0,
            {},
        )
        assert any(any(ch.isascii() and ch.isalpha() for ch in query) for query in queries)
        assert any(any("\u4e00" <= ch <= "\u9fff" for ch in query) for query in queries)
        assert metadata["bilingual_expected"] is True
        assert "query_role_mix" in metadata
        assert any(role == "english_official" for role in metadata["query_role_mix"])

    async def test_strengthens_english_seed(self):
        non_english_query = _non_english_query()

        queries, metadata = _ensure_bilingual_queries(
            [non_english_query],
            f"{non_english_query} HouYi LangChain",
            f"{non_english_query} HouYi LangChain",
        )

        assert any("official report" in query.lower() for query in queries if query.isascii())
        assert metadata["bilingual_fallback_applied"] is True
        assert "english_official" in metadata["query_role_mix"]

    async def test_claim_dedupes_normalized(self):
        planner = _planner(claim_query=AsyncMock(side_effect=[True, False]))
        claimed, skipped = await planner.claim_queries(["Q1", "q1", "Q2"], set())
        assert claimed == ["Q1"]
        assert skipped == 2

    async def test_snapshot_normalizes_non_dict(self):
        async def _bad(_round):
            return [_round]

        planner = _planner(get_collaboration_snapshot=_bad)
        result = await planner.read_collaboration_snapshot(1)
        assert result == {}

    async def test_snapshot_returns_empty(self):
        planner = _planner()
        result = await planner.read_collaboration_snapshot(1)
        assert result == {}
