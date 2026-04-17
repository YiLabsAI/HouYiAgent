from __future__ import annotations

from dataclasses import dataclass

from houyi.application.research.types import SearchContext, SubQuestion

_DEFAULT_QUERY_BUDGET_MS = 45_000
_DEFAULT_ROUND_BUDGET_MS = 90_000
_MIN_BUDGET_MS = 250
_DEFAULT_TOTAL_SOURCE_CAP = 100
# Small expected-source tasks only need a little over-collection headroom,
# while broader tasks benefit from one extra slot to avoid brittle under-fetch.
_SMALL_SOURCE_TARGET_THRESHOLD = 4
_SMALL_SOURCE_TARGET_HEADROOM = 2
_LARGE_SOURCE_TARGET_HEADROOM = 3
# Do not keep expanding the per-sub-question source target indefinitely; beyond
# this point latency rises faster than evidence quality in current benchmarks.
_MAX_TOTAL_SOURCE_TARGET = 12


@dataclass(slots=True, frozen=True)
class BudgetPolicy:
    default_query_budget_ms: int = _DEFAULT_QUERY_BUDGET_MS
    default_round_budget_ms: int = _DEFAULT_ROUND_BUDGET_MS
    min_budget_ms: int = _MIN_BUDGET_MS

    def resolve_max_results_per_query(
        self,
        context: SearchContext,
        default_max_results_per_query: int,
    ) -> int:
        if context.max_results_per_query > 0:
            return context.max_results_per_query
        return default_max_results_per_query

    def resolve_query_parallelism(
        self,
        context: SearchContext,
        query_count: int,
        default_parallelism: int,
    ) -> int:
        if query_count <= 0:
            return 1
        configured = context.max_query_parallelism or default_parallelism
        return max(1, min(configured, default_parallelism, query_count))

    def resolve_total_source_target(self, sub_question: SubQuestion, context: SearchContext) -> int:
        configured_cap = (
            context.max_total_sources
            if context.max_total_sources > 0
            else _DEFAULT_TOTAL_SOURCE_CAP
        )
        expected = max(sub_question.expected_sources, 1)
        headroom = (
            _SMALL_SOURCE_TARGET_HEADROOM
            if expected <= _SMALL_SOURCE_TARGET_THRESHOLD
            else _LARGE_SOURCE_TARGET_HEADROOM
        )
        target = max(expected, min(expected + headroom, _MAX_TOTAL_SOURCE_TARGET))
        return max(1, min(configured_cap, target))

    def resolve_query_budget_ms(
        self,
        context: SearchContext,
        round_budget_ms: int,
        query_count: int,
    ) -> int:
        configured = context.max_query_budget_ms
        if configured > 0:
            return max(self.min_budget_ms, min(configured, round_budget_ms or configured))
        if round_budget_ms <= 0:
            return self.default_query_budget_ms
        derived = max(self.min_budget_ms, round_budget_ms // max(query_count, 1))
        return min(self.default_query_budget_ms, derived)

    def resolve_round_budget_ms(
        self,
        context: SearchContext,
        remaining_sub_question_ms: int,
        rounds_remaining: int,
    ) -> int:
        configured = context.max_round_budget_ms
        if configured > 0:
            return max(self.min_budget_ms, min(configured, remaining_sub_question_ms))
        if remaining_sub_question_ms <= 0:
            return 0
        derived = max(self.min_budget_ms, remaining_sub_question_ms // max(rounds_remaining, 1))
        return min(self.default_round_budget_ms, derived)

    def resolve_sub_question_budget_ms(self, context: SearchContext, max_rounds: int) -> int:
        if context.max_sub_question_budget_ms > 0:
            return context.max_sub_question_budget_ms
        return max_rounds * self.default_round_budget_ms
