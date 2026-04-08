from __future__ import annotations

from unittest.mock import AsyncMock

from houyi.application.research.runtime.search_query_planner import QueryPlanner

from ..conftest import MockLLM


def _planner(**kwargs) -> QueryPlanner:
    return QueryPlanner(llm=MockLLM(), max_rounds=3, llm_kwargs={}, **kwargs)


class TestQueryPlanner:
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
