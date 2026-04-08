from __future__ import annotations

from houyi.application.research.runtime.search_budget import BudgetPolicy
from houyi.application.research.types import SearchContext, SubQuestion


def _ctx(**kwargs) -> SearchContext:
    return SearchContext(run_id="r1", plan_id="p1", user_query="q", **kwargs)


class TestBudgetPolicy:
    def test_source_target_uses_expected(self):
        policy = BudgetPolicy()
        target = policy.resolve_total_source_target(
            SubQuestion(question="Q?", expected_sources=3),
            _ctx(max_total_sources=10),
        )
        assert target == 3

    def test_query_budget_min_floor(self):
        policy = BudgetPolicy()
        budget = policy.resolve_query_budget_ms(
            _ctx(max_query_budget_ms=120),
            round_budget_ms=200,
            query_count=2,
        )
        assert budget == 250

    def test_budget_derived_from_remaining(self):
        policy = BudgetPolicy()
        budget = policy.resolve_round_budget_ms(
            _ctx(),
            remaining_sub_question_ms=600,
            rounds_remaining=3,
        )
        assert budget == 250

    def test_parallelism_capped_by_count(self):
        policy = BudgetPolicy()
        result = policy.resolve_query_parallelism(_ctx(), query_count=1, default_parallelism=4)
        assert result == 1

    def test_sub_question_budget_default(self):
        policy = BudgetPolicy()
        budget = policy.resolve_sub_question_budget_ms(_ctx(), max_rounds=3)
        assert budget == 3 * 90_000
