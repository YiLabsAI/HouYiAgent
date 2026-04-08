"""SearchExecutor — multi-round search orchestration for a single sub-question."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.runtime.search_budget import BudgetPolicy
from houyi.application.research.runtime.search_query_planner import (
    CollaborationSnapshotCallback,
    QueryPlanner,
    _collaboration_stop_reason,
)
from houyi.application.research.runtime.search_round_runner import RoundRequest, RoundRunner
from houyi.application.research.runtime.search_sufficiency import SufficiencyEvaluator
from houyi.application.research.runtime.search_telemetry import (
    SearchEventCallback,
    TelemetryEmitter,
)
from houyi.application.research.types import (
    SearchContext,
    SearchResult,
    SearchRound,
    SourceReference,
    SubQuestion,
    SufficiencyDecision,
    SufficiencyFeatures,
)
from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)


class SearchExecutor:
    """Coordinates multi-round search for a single sub-question.

    Uses existing ``WebSearchService`` with ``include_content=True`` (browse
    mode) to get both URLs and full content in a single call.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        web_search: WebSearchService,
        max_search_rounds: int = 3,
        max_results_per_query: int = 8,
        max_query_parallelism: int = 2,
        on_event: SearchEventCallback | None = None,
        claim_query: Callable[[str], Awaitable[bool]] | None = None,
        claim_url: Callable[[str], Awaitable[bool]] | None = None,
        check_cancelled: Callable[[], None] | None = None,
        get_collaboration_snapshot: CollaborationSnapshotCallback | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self._max_rounds = max_search_rounds
        self._max_results_per_query = max_results_per_query
        self._max_query_parallelism = max(1, max_query_parallelism)
        self._on_event = on_event
        self._budget_policy = BudgetPolicy()
        self._query_planner = QueryPlanner(
            llm=llm_adapter,
            max_rounds=max_search_rounds,
            llm_kwargs=llm_kwargs,
            claim_query=claim_query,
            get_collaboration_snapshot=get_collaboration_snapshot,
        )
        self._telemetry = TelemetryEmitter(self._notify)
        self._round_runner = RoundRunner(
            web_search=web_search,
            telemetry=self._telemetry,
            claim_url=claim_url,
            check_cancelled=check_cancelled,
        )
        self._sufficiency_evaluator = SufficiencyEvaluator(llm=llm_adapter, llm_kwargs=llm_kwargs)
        self._check_cancelled = check_cancelled

    async def search(
        self,
        sub_question: SubQuestion,
        context: SearchContext,
    ) -> SearchResult:
        """Execute multi-round search until information is sufficient.

        Returns a ``SearchResult`` containing all rounds, deduplicated
        sources, and a coverage score.
        """
        rounds: list[SearchRound] = []
        all_sources: dict[str, SourceReference] = {}
        seen_urls: set[str] = set(context.excluded_urls)
        prior = list(context.prior_findings)
        max_results_per_query = self._budget_policy.resolve_max_results_per_query(
            context,
            self._max_results_per_query,
        )
        total_source_target = self._budget_policy.resolve_total_source_target(sub_question, context)
        search_started = time.perf_counter()
        sub_question_budget_ms = self._budget_policy.resolve_sub_question_budget_ms(
            context,
            self._max_rounds,
        )

        for round_idx in range(self._max_rounds):
            self._run_cancel_check()
            round_number = round_idx + 1
            remaining_sub_question_ms = self._remaining_budget_ms(
                search_started,
                sub_question_budget_ms,
            )
            if remaining_sub_question_ms <= 0:
                decision = SufficiencyDecision(
                    sufficient=False,
                    rationale="Sub-question budget exhausted before next round",
                    decision_by="guardrail",
                    reason_code="sub_question_budget_exhausted",
                )
                rounds.append(
                    SearchRound(
                        round_index=round_idx,
                        queries=[],
                        hits=[],
                        sufficient=False,
                        rationale=decision.rationale,
                        elapsed_ms=0.0,
                        decision_by=decision.decision_by,
                        reason_code=decision.reason_code,
                        stop_layer="sub_question",
                        missing_dimensions=decision.missing_dimensions,
                        features=decision.features,
                    )
                )
                await self._telemetry.budget_consumed(
                    question_id=sub_question.question_id,
                    round_index=round_number,
                    layer="sub_question",
                    reason_code=decision.reason_code,
                    budget_ms=sub_question_budget_ms,
                    remaining_ms=0,
                )
                await self._telemetry.round_timing(
                    question_id=sub_question.question_id,
                    round_number=round_number,
                    elapsed_ms=0.0,
                    query_count=0,
                    skipped_queries=0,
                    cancelled_queries=0,
                    hit_count=0,
                    source_count=len(all_sources),
                    decision=decision,
                    stop_layer="sub_question",
                )
                break
            collaboration = await self._query_planner.read_collaboration_snapshot(round_number)
            stop_reason = _collaboration_stop_reason(collaboration)
            if stop_reason:
                decision = SufficiencyDecision(
                    sufficient=True,
                    rationale=stop_reason,
                    decision_by="collaboration",
                    reason_code="collaboration_stop",
                )
                rounds.append(
                    SearchRound(
                        round_index=round_idx,
                        queries=[],
                        hits=[],
                        sufficient=decision.sufficient,
                        rationale=decision.rationale,
                        elapsed_ms=0.0,
                        decision_by=decision.decision_by,
                        reason_code=decision.reason_code,
                        stop_layer="collaboration",
                        missing_dimensions=decision.missing_dimensions,
                        features=decision.features,
                    )
                )
                await self._telemetry.round_timing(
                    question_id=sub_question.question_id,
                    round_number=round_number,
                    elapsed_ms=0.0,
                    query_count=0,
                    skipped_queries=0,
                    cancelled_queries=0,
                    hit_count=0,
                    source_count=len(all_sources),
                    decision=decision,
                    stop_layer="collaboration",
                )
                break
            raw_queries = await self._generate_queries(
                sub_question.question,
                context.user_query,
                prior,
                round_idx,
                collaboration,
            )
            queries, skipped_queries = await self._query_planner.claim_queries(raw_queries, set())

            await self._notify(
                "search.queries_generated",
                {
                    "question_id": sub_question.question_id,
                    "round": round_number,
                    "queries": queries,
                },
            )

            round_started = time.perf_counter()
            round_budget_ms = self._budget_policy.resolve_round_budget_ms(
                context,
                remaining_sub_question_ms,
                self._max_rounds - round_idx,
            )
            round_result = await self._round_runner.run(
                RoundRequest(
                    question_id=sub_question.question_id,
                    round_index=round_number,
                    queries=queries,
                    seen_urls=seen_urls,
                    all_sources=all_sources,
                    max_results_per_query=max_results_per_query,
                    query_parallelism=self._budget_policy.resolve_query_parallelism(
                        context,
                        len(queries),
                        self._max_query_parallelism,
                    ),
                    target_total_sources=total_source_target,
                    query_budget_ms=self._budget_policy.resolve_query_budget_ms(
                        context,
                        round_budget_ms,
                        len(queries),
                    ),
                    round_budget_ms=round_budget_ms,
                )
            )
            hits = round_result.hits

            summary = "; ".join(h.title for h in hits[:5])
            features = self._sufficiency_evaluator.build_features(
                list(all_sources.values()),
                sub_question.question,
                context.user_query,
            )
            await self._telemetry.sufficiency_features(
                question_id=sub_question.question_id,
                round_index=round_number,
                features=features,
            )
            if round_result.stop_layer:
                decision = SufficiencyDecision(
                    sufficient=False,
                    rationale=round_result.stop_reason,
                    decision_by="guardrail",
                    reason_code=round_result.reason_code,
                    missing_dimensions=list(features.missing_dimensions),
                    features=features,
                )
            else:
                decision = await self._evaluate_sufficiency(
                    question=sub_question.question,
                    user_query=context.user_query,
                    summary=summary,
                    sources=list(all_sources.values()),
                    collaboration=collaboration,
                    features=features,
                    expected_sources=sub_question.expected_sources,
                )
            round_elapsed_ms = (time.perf_counter() - round_started) * 1000.0
            await self._telemetry.sufficiency_decision(
                question_id=sub_question.question_id,
                round_index=round_number,
                decision=decision,
            )

            rounds.append(
                SearchRound(
                    round_index=round_idx,
                    queries=queries,
                    hits=hits,
                    sufficient=decision.sufficient,
                    rationale=decision.rationale,
                    elapsed_ms=round(round_elapsed_ms, 1),
                    skipped_queries=skipped_queries + round_result.skipped_queries,
                    cancelled_queries=round_result.cancelled_queries,
                    decision_by=decision.decision_by,
                    reason_code=decision.reason_code,
                    stop_layer=round_result.stop_layer,
                    missing_dimensions=decision.missing_dimensions,
                    features=decision.features,
                )
            )

            await self._telemetry.round_timing(
                question_id=sub_question.question_id,
                round_number=round_number,
                elapsed_ms=round(round_elapsed_ms, 1),
                query_count=len(queries),
                skipped_queries=skipped_queries + round_result.skipped_queries,
                cancelled_queries=round_result.cancelled_queries,
                hit_count=len(hits),
                source_count=len(all_sources),
                decision=decision,
                stop_layer=round_result.stop_layer,
            )

            if hits:
                prior.append(summary)

            if decision.sufficient or not queries or round_result.stop_layer:
                break

        sources = list(all_sources.values())
        coverage = min(1.0, len(sources) / max(sub_question.expected_sources, 1))

        return SearchResult(
            question_id=sub_question.question_id,
            rounds=rounds,
            sources=sources,
            summary=prior[-1] if prior else "",
            coverage_score=coverage,
            exhausted=len(rounds) >= self._max_rounds and not rounds[-1].sufficient,
        )

    async def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        if self._on_event:
            try:
                await self._on_event(event_type, data)
            except Exception:
                logger.debug("Search event callback failed for %s", event_type, exc_info=True)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _generate_queries(
        self,
        question: str,
        user_query: str,
        prior: list[str],
        round_idx: int,
        collaboration: dict[str, Any],
    ) -> list[str]:
        return await self._query_planner.generate_queries(
            question,
            user_query,
            prior,
            round_idx,
            collaboration,
        )

    async def _evaluate_sufficiency(
        self,
        *,
        question: str,
        user_query: str,
        summary: str,
        sources: list[SourceReference],
        collaboration: dict[str, Any],
        features: SufficiencyFeatures,
        expected_sources: int,
    ) -> SufficiencyDecision:
        return await self._sufficiency_evaluator.evaluate(
            question=question,
            user_query=user_query,
            summary=summary,
            sources=sources,
            collaboration=collaboration,
            features=features,
            expected_sources=expected_sources,
        )

    def _run_cancel_check(self) -> None:
        if self._check_cancelled is not None:
            self._check_cancelled()

    def _remaining_budget_ms(self, started: float, budget_ms: int) -> int:
        if budget_ms <= 0:
            return 0
        elapsed_ms = int((time.perf_counter() - started) * 1000.0)
        return max(0, budget_ms - elapsed_ms)
