"""SearchExecutor — multi-round search orchestration for a single sub-question."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any
from urllib.parse import urlparse

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.runtime.search_budget import BudgetPolicy
from houyi.application.research.runtime.search_query_planner import (
    CollaborationSnapshotCallback,
    QueryPlanner,
    _collaboration_stop_reason,
    _looks_entity_query,
)
from houyi.application.research.runtime.search_round_runner import RoundRequest, RoundRunner
from houyi.application.research.runtime.search_sufficiency import SufficiencyEvaluator
from houyi.application.research.runtime.search_telemetry import (
    SearchEventCallback,
    TelemetryEmitter,
)
from houyi.application.research.taxonomy import (
    QUERY_HYGIENE_CJK_STOPWORDS,
    QUERY_HYGIENE_EN_STOPWORDS,
    QUERY_HYGIENE_FILLER_TOKENS,
)
from houyi.application.research.types import (
    AnswerCoverageContract,
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

# Follow-up query budgeting keeps the first round broad enough to establish
# coverage, then narrows quickly to control search latency. The current values
# preserve the pre-existing heuristic shape: 3 queries to start, 2 when breadth
# is still required, 1 when signals indicate focused deepening is cheaper.
_INITIAL_QUERY_CAP = 3
_BREADTH_QUERY_CAP = 2
_FOCUSED_QUERY_CAP = 1
# Once we already have ~4 accumulated sources or 2 shared-source signals, the
# executor can usually switch from exploration to focused follow-up without
# hurting coverage; these thresholds are kept behavior-compatible with the
# earlier inline heuristic and should be revisited together with benchmark data.
_FOCUSED_SOURCE_THRESHOLD = 4
_SHARED_SIGNAL_THRESHOLD = 2
# When remaining sub-question budget drops below this fraction, allow early
# termination even if critical gaps (authority / identity / facet) are still
# open.  This prevents search-phase latency blowout on hard-to-resolve
# sub-questions while preserving thorough search when budget allows.
_EARLY_STOP_BUDGET_FLOOR = 0.20


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
        seen_queries: set[str] = set()
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
        completed_rounds: list[SearchRound] = []
        completed_sources: dict[str, SourceReference] = {}
        completed_summary = ""

        try:
            for round_idx in range(self._max_rounds):
                self._run_cancel_check()
                round_number = round_idx + 1
                remaining_sub_question_ms = self._remaining_budget_ms(
                    search_started,
                    sub_question_budget_ms,
                )
                if remaining_sub_question_ms <= 0:
                    await self._record_budget_exhausted_round(
                        rounds=rounds,
                        sub_question=sub_question,
                        round_idx=round_idx,
                        round_number=round_number,
                        all_sources=all_sources,
                        sub_question_budget_ms=sub_question_budget_ms,
                    )
                    break
                collaboration = await self._query_planner.read_collaboration_snapshot(round_number)
                stop_reason = _collaboration_stop_reason(collaboration)
                if stop_reason:
                    await self._record_collaboration_stop_round(
                        rounds=rounds,
                        sub_question=sub_question,
                        round_idx=round_idx,
                        round_number=round_number,
                        all_sources=all_sources,
                        stop_reason=stop_reason,
                    )
                    break
                round_state = await self._execute_round(
                    sub_question=sub_question,
                    context=context,
                    round_idx=round_idx,
                    round_number=round_number,
                    collaboration=collaboration,
                    previous_round=rounds[-1] if rounds else None,
                    prior=prior,
                    all_sources=all_sources,
                    seen_queries=seen_queries,
                    seen_urls=seen_urls,
                    max_results_per_query=max_results_per_query,
                    total_source_target=total_source_target,
                    remaining_sub_question_ms=remaining_sub_question_ms,
                )
                rounds.append(round_state["round"])
                if round_state["hits"]:
                    prior.append(round_state["summary"])
                completed_rounds = list(rounds)
                completed_sources = dict(all_sources)
                completed_summary = prior[-1] if prior else ""

                remaining_ratio = self._remaining_budget_ms(
                    search_started, sub_question_budget_ms
                ) / max(sub_question_budget_ms, 1)
                if _can_terminate_early(rounds, remaining_budget_ratio=remaining_ratio):
                    rounds[-1].stop_layer = "early_termination"
                    await self._notify(
                        "search.early_termination",
                        {
                            "question_id": sub_question.question_id,
                            "round": round_number,
                            "new_unique_urls": len(rounds[-1].hits),
                            "new_domains": _count_domains(rounds[-1].hits),
                            "missing_dimensions_count": len(rounds[-1].missing_dimensions),
                        },
                    )
                    break

                if (
                    round_state["decision"].sufficient
                    or not round_state["queries"]
                    or round_state["stop_layer"]
                ):
                    break
        except asyncio.CancelledError:
            if not context.salvage_on_cancel:
                raise
            await self._telemetry.partial_result_returned(
                question_id=sub_question.question_id,
                reason="timeout",
                completed_rounds=len(completed_rounds),
                source_count=len(completed_sources),
            )
            return self._build_search_result(
                sub_question=sub_question,
                rounds=completed_rounds,
                sources=list(completed_sources.values()),
                summary=completed_summary or "Search timed out",
                error="search_timeout_partial",
            )

        sources = list(all_sources.values())
        coverage = min(1.0, len(sources) / max(sub_question.expected_sources, 1))
        exhausted = (
            bool(rounds)
            and not any(round_.sufficient for round_ in rounds)
            and (
                len(rounds) >= self._max_rounds
                or not rounds[-1].queries
                or bool(rounds[-1].stop_layer)
            )
        )

        return self._build_search_result(
            sub_question=sub_question,
            rounds=rounds,
            sources=sources,
            summary=prior[-1] if prior else "",
            coverage_score=coverage,
            exhausted=exhausted,
        )

    def _build_search_result(
        self,
        *,
        sub_question: SubQuestion,
        rounds: list[SearchRound],
        sources: list[SourceReference],
        summary: str,
        error: str | None = None,
        coverage_score: float | None = None,
        exhausted: bool | None = None,
    ) -> SearchResult:
        if coverage_score is None:
            coverage_score = min(1.0, len(sources) / max(sub_question.expected_sources, 1))
        if exhausted is None:
            exhausted = (
                bool(rounds)
                and not any(round_.sufficient for round_ in rounds)
                and (
                    len(rounds) >= self._max_rounds
                    or not rounds[-1].queries
                    or bool(rounds[-1].stop_layer)
                )
            )
        return SearchResult(
            question_id=sub_question.question_id,
            rounds=rounds,
            sources=sources,
            summary=summary,
            coverage_score=coverage_score,
            exhausted=exhausted,
            error=error,
        )

    async def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        if self._on_event:
            try:
                await self._on_event(event_type, data)
            except Exception:
                logger.debug("Search event callback failed for %s", event_type, exc_info=True)

    async def _record_budget_exhausted_round(
        self,
        *,
        rounds: list[SearchRound],
        sub_question: SubQuestion,
        round_idx: int,
        round_number: int,
        all_sources: dict[str, SourceReference],
        sub_question_budget_ms: int,
    ) -> None:
        decision = SufficiencyDecision(
            sufficient=False,
            rationale="Sub-question budget exhausted before next round",
            decision_by="guardrail",
            reason_code="sub_question_budget_exhausted",
        )
        rounds.append(
            self._build_round_record(
                round_idx=round_idx,
                queries=[],
                hits=[],
                decision=decision,
                elapsed_ms=0.0,
                skipped_queries=0,
                cancelled_queries=0,
                stop_layer="sub_question",
                new_unique_urls=0,
                new_domains=0,
                zero_hit_query_count=0,
                duplicate_url_rate=0.0,
                missing_dimensions_count=len(decision.missing_dimensions),
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
        await self._emit_round_timing(
            sub_question=sub_question,
            round_number=round_number,
            all_sources=all_sources,
            decision=decision,
            queries=[],
            hits=[],
            elapsed_ms=0.0,
            skipped_queries=0,
            cancelled_queries=0,
            stop_layer="sub_question",
            new_unique_urls=0,
            new_domains=0,
            zero_hit_query_count=0,
            duplicate_url_rate=0.0,
            missing_dimensions_count=len(decision.missing_dimensions),
        )

    async def _record_collaboration_stop_round(
        self,
        *,
        rounds: list[SearchRound],
        sub_question: SubQuestion,
        round_idx: int,
        round_number: int,
        all_sources: dict[str, SourceReference],
        stop_reason: str,
    ) -> None:
        decision = SufficiencyDecision(
            sufficient=True,
            rationale=stop_reason,
            decision_by="collaboration",
            reason_code="collaboration_stop",
        )
        rounds.append(
            self._build_round_record(
                round_idx=round_idx,
                queries=[],
                hits=[],
                decision=decision,
                elapsed_ms=0.0,
                skipped_queries=0,
                cancelled_queries=0,
                stop_layer="collaboration",
                new_unique_urls=0,
                new_domains=0,
                zero_hit_query_count=0,
                duplicate_url_rate=0.0,
                missing_dimensions_count=len(decision.missing_dimensions),
            )
        )
        await self._emit_round_timing(
            sub_question=sub_question,
            round_number=round_number,
            all_sources=all_sources,
            decision=decision,
            queries=[],
            hits=[],
            elapsed_ms=0.0,
            skipped_queries=0,
            cancelled_queries=0,
            stop_layer="collaboration",
            new_unique_urls=0,
            new_domains=0,
            zero_hit_query_count=0,
            duplicate_url_rate=0.0,
            missing_dimensions_count=len(decision.missing_dimensions),
        )

    async def _execute_round(
        self,
        *,
        sub_question: SubQuestion,
        context: SearchContext,
        round_idx: int,
        round_number: int,
        collaboration: dict[str, Any],
        previous_round: SearchRound | None,
        prior: list[str],
        all_sources: dict[str, SourceReference],
        seen_queries: set[str],
        seen_urls: set[str],
        max_results_per_query: int,
        total_source_target: int,
        remaining_sub_question_ms: int,
    ) -> dict[str, Any]:
        queries, skipped_queries = await self._prepare_round_queries(
            sub_question=sub_question,
            context=context,
            round_idx=round_idx,
            round_number=round_number,
            prior=prior,
            collaboration=collaboration,
            all_sources=all_sources,
            seen_queries=seen_queries,
            previous_round=previous_round,
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
        summary = "; ".join(hit.title for hit in hits[:5])
        decision = await self._decide_round_sufficiency(
            sub_question=sub_question,
            context=context,
            round_number=round_number,
            summary=summary,
            all_sources=all_sources,
            collaboration=collaboration,
            round_result=round_result,
        )
        round_elapsed_ms = (time.perf_counter() - round_started) * 1000.0
        skipped_total = skipped_queries + round_result.skipped_queries
        raw_hit_count = sum(len(execution.hits) for execution in round_result.executions)
        zero_hit_query_count = sum(1 for execution in round_result.executions if not execution.hits)
        duplicate_url_rate = 0.0
        if raw_hit_count > 0:
            duplicate_url_rate = max(0.0, (raw_hit_count - len(hits)) / raw_hit_count)
        await self._telemetry.sufficiency_decision(
            question_id=sub_question.question_id,
            round_index=round_number,
            decision=decision,
        )
        round_record = self._build_round_record(
            round_idx=round_idx,
            queries=queries,
            hits=hits,
            decision=decision,
            elapsed_ms=round(round_elapsed_ms, 1),
            skipped_queries=skipped_total,
            cancelled_queries=round_result.cancelled_queries,
            stop_layer=round_result.stop_layer,
            new_unique_urls=len(hits),
            new_domains=_count_domains(hits),
            zero_hit_query_count=zero_hit_query_count,
            duplicate_url_rate=round(duplicate_url_rate, 3),
            missing_dimensions_count=len(decision.missing_dimensions),
        )
        await self._emit_round_timing(
            sub_question=sub_question,
            round_number=round_number,
            all_sources=all_sources,
            decision=decision,
            queries=queries,
            hits=hits,
            elapsed_ms=round(round_elapsed_ms, 1),
            skipped_queries=skipped_total,
            cancelled_queries=round_result.cancelled_queries,
            stop_layer=round_result.stop_layer,
            new_unique_urls=len(hits),
            new_domains=_count_domains(hits),
            zero_hit_query_count=zero_hit_query_count,
            duplicate_url_rate=round(duplicate_url_rate, 3),
            missing_dimensions_count=len(decision.missing_dimensions),
        )
        return {
            "round": round_record,
            "decision": decision,
            "queries": queries,
            "summary": summary,
            "hits": hits,
            "stop_layer": round_result.stop_layer,
        }

    async def _prepare_round_queries(
        self,
        *,
        sub_question: SubQuestion,
        context: SearchContext,
        round_idx: int,
        round_number: int,
        prior: list[str],
        collaboration: dict[str, Any],
        all_sources: dict[str, SourceReference],
        seen_queries: set[str],
        previous_round: SearchRound | None,
    ) -> tuple[list[str], int]:
        raw_queries, query_metadata = await self._generate_queries(
            sub_question.question,
            context.user_query,
            prior,
            round_idx,
            collaboration,
            sub_question.coverage_contract,
            query_type=sub_question.query_type,
            disambiguation_needed=sub_question.disambiguation_needed,
        )
        # Drop noisy LLM-emitted queries (repeated tokens, filler-only,
        # stopword-only) before they consume dedup / budget / provider slots.
        # Hygiene is purely reductive and never rewrites query text so
        # downstream attribution remains clean.
        raw_queries, hygiene_dropped = _apply_query_hygiene(raw_queries)
        if hygiene_dropped:
            query_metadata["hygiene_dropped"] = hygiene_dropped
        queries, skipped_queries = await self._query_planner.claim_queries(
            self._trim_queries_for_round(
                raw_queries,
                round_idx=round_idx,
                accumulated_source_count=len(all_sources),
                prior_count=len(prior),
                collaboration=collaboration,
                previous_round=previous_round,
                sub_question=sub_question,
                user_query=context.user_query,
            ),
            seen_queries,
        )
        # Disambiguation enrichment: when entity_identity gaps persist after
        # round 0 AND the LLM's queries were all deduplicated (leaving no
        # new queries), append forced disambiguation queries as a fallback.
        # Kept narrow to avoid flooding the search API with extra queries
        # when the LLM already produced viable ones.
        if (
            not queries
            and round_idx > 0
            and previous_round is not None
            and "entity_identity" in (previous_round.missing_dimensions or [])
        ):
            fallbacks = _make_disambiguation_queries(
                sub_question.question,
                context.user_query,
            )
            disam_queries, extra_skipped = await self._query_planner.claim_queries(
                fallbacks,
                seen_queries,
            )
            skipped_queries += extra_skipped
            if disam_queries:
                queries.extend(disam_queries)
                query_metadata["disambiguation_fallback_applied"] = True
        await self._notify(
            "search.queries_generated",
            {
                "question_id": sub_question.question_id,
                "round": round_number,
                "queries": queries,
                **query_metadata,
            },
        )
        return queries, skipped_queries

    async def _decide_round_sufficiency(
        self,
        *,
        sub_question: SubQuestion,
        context: SearchContext,
        round_number: int,
        summary: str,
        all_sources: dict[str, SourceReference],
        collaboration: dict[str, Any],
        round_result: Any,
    ) -> SufficiencyDecision:
        features = self._sufficiency_evaluator.build_features(
            list(all_sources.values()),
            sub_question.question,
            context.user_query,
            sub_question.coverage_contract,
        )
        await self._telemetry.sufficiency_features(
            question_id=sub_question.question_id,
            round_index=round_number,
            features=features,
        )
        if round_result.stop_layer:
            return SufficiencyDecision(
                sufficient=False,
                rationale=round_result.stop_reason,
                decision_by="guardrail",
                reason_code=round_result.reason_code,
                missing_dimensions=list(features.missing_dimensions),
                features=features,
            )
        # Note: _guardrail_sufficiency_decision inside
        # SufficiencyEvaluator.evaluate() already short-circuits the LLM
        # call when source_count >= expected_sources AND quality criteria
        # (relevance, diversity, authority, recency) are met.  We rely on
        # that structured guardrail rather than a raw count check here.
        return await self._evaluate_sufficiency(
            question=sub_question.question,
            user_query=context.user_query,
            summary=summary,
            sources=list(all_sources.values()),
            collaboration=collaboration,
            features=features,
            expected_sources=sub_question.expected_sources,
            coverage_contract=sub_question.coverage_contract,
        )

    def _build_round_record(
        self,
        *,
        round_idx: int,
        queries: list[str],
        hits: list[Any],
        decision: SufficiencyDecision,
        elapsed_ms: float,
        skipped_queries: int,
        cancelled_queries: int,
        stop_layer: str | None,
        new_unique_urls: int,
        new_domains: int,
        zero_hit_query_count: int,
        duplicate_url_rate: float,
        missing_dimensions_count: int,
    ) -> SearchRound:
        normalized_stop_layer = stop_layer or ""
        return SearchRound(
            round_index=round_idx,
            queries=queries,
            hits=hits,
            sufficient=decision.sufficient,
            rationale=decision.rationale,
            elapsed_ms=elapsed_ms,
            skipped_queries=skipped_queries,
            cancelled_queries=cancelled_queries,
            decision_by=decision.decision_by,
            reason_code=decision.reason_code,
            stop_layer=normalized_stop_layer,
            missing_dimensions=decision.missing_dimensions,
            features=decision.features,
            new_unique_urls=new_unique_urls,
            new_domains=new_domains,
            zero_hit_query_count=zero_hit_query_count,
            duplicate_url_rate=duplicate_url_rate,
            missing_dimensions_count=missing_dimensions_count,
        )

    async def _emit_round_timing(
        self,
        *,
        sub_question: SubQuestion,
        round_number: int,
        all_sources: dict[str, SourceReference],
        decision: SufficiencyDecision,
        queries: list[str],
        hits: list[Any],
        elapsed_ms: float,
        skipped_queries: int,
        cancelled_queries: int,
        stop_layer: str | None,
        new_unique_urls: int,
        new_domains: int,
        zero_hit_query_count: int,
        duplicate_url_rate: float,
        missing_dimensions_count: int,
    ) -> None:
        normalized_stop_layer = stop_layer or ""
        await self._telemetry.round_timing(
            question_id=sub_question.question_id,
            round_number=round_number,
            elapsed_ms=elapsed_ms,
            query_count=len(queries),
            skipped_queries=skipped_queries,
            cancelled_queries=cancelled_queries,
            hit_count=len(hits),
            source_count=len(all_sources),
            decision=decision,
            stop_layer=normalized_stop_layer,
            new_unique_urls=new_unique_urls,
            new_domains=new_domains,
            zero_hit_query_count=zero_hit_query_count,
            duplicate_url_rate=duplicate_url_rate,
            missing_dimensions_count=missing_dimensions_count,
        )

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
        coverage_contract: AnswerCoverageContract,
        *,
        query_type: str = "factual",
        disambiguation_needed: bool = False,
    ) -> tuple[list[str], dict[str, Any]]:
        return await self._query_planner.generate_queries(
            question,
            user_query,
            prior,
            round_idx,
            collaboration,
            coverage_contract=coverage_contract,
            query_type=query_type,
            disambiguation_needed=disambiguation_needed,
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
        coverage_contract: Any,
    ) -> SufficiencyDecision:
        return await self._sufficiency_evaluator.evaluate(
            question=question,
            user_query=user_query,
            summary=summary,
            sources=sources,
            collaboration=collaboration,
            features=features,
            expected_sources=expected_sources,
            coverage_contract=coverage_contract,
        )

    def _run_cancel_check(self) -> None:
        if self._check_cancelled is not None:
            self._check_cancelled()

    def _remaining_budget_ms(self, started: float, budget_ms: int) -> int:
        if budget_ms <= 0:
            return 0
        elapsed_ms = int((time.perf_counter() - started) * 1000.0)
        return max(0, budget_ms - elapsed_ms)

    def _trim_queries_for_round(
        self,
        queries: list[str],
        *,
        round_idx: int,
        accumulated_source_count: int,
        prior_count: int,
        collaboration: dict[str, Any],
        previous_round: SearchRound | None,
        sub_question: SubQuestion | None = None,
        user_query: str = "",
    ) -> list[str]:
        if round_idx <= 0:
            return queries[:_INITIAL_QUERY_CAP]
        gap_count = _gap_count(previous_round)
        if gap_count == 0:
            return []
        features = previous_round.features if previous_round is not None else None
        ambiguous_entity = bool(
            sub_question
            and _looks_entity_query(
                sub_question.question,
                user_query,
                sub_question.coverage_contract,
            )
        )
        needs_breadth = bool(
            ambiguous_entity
            or gap_count is None
            or gap_count >= 2
            or (features is not None and features.authority_source_count == 0)
            or (
                features is not None
                and features.noisy_source_count > max(features.source_count // 2, 1)
            )
        )
        if gap_count == 1:
            if needs_breadth:
                return queries[:_BREADTH_QUERY_CAP]
            return queries[:_FOCUSED_QUERY_CAP]
        shared_source_count = collaboration.get("shared_source_count", 0)
        if round_idx == 1:
            cap = (
                _BREADTH_QUERY_CAP
                if needs_breadth
                else (
                    _FOCUSED_QUERY_CAP
                    if accumulated_source_count >= _FOCUSED_SOURCE_THRESHOLD
                    or shared_source_count >= _SHARED_SIGNAL_THRESHOLD
                    else _BREADTH_QUERY_CAP
                )
            )
            return queries[:cap]
        cap = (
            _BREADTH_QUERY_CAP
            if needs_breadth
            else (
                _FOCUSED_QUERY_CAP
                if (accumulated_source_count > 0 or prior_count > 0 or shared_source_count > 0)
                else _BREADTH_QUERY_CAP
            )
        )
        return queries[:cap]


def _gap_count(round_: SearchRound | None) -> int | None:
    if round_ is None:
        return None
    return len(round_.missing_dimensions)


def _count_domains(hits: list[Any]) -> int:
    domains = {
        (getattr(hit, "domain", None) or urlparse(getattr(hit, "url", "")).netloc).lower()
        for hit in hits
        if getattr(hit, "url", None)
    }
    return len({domain for domain in domains if domain})


def _can_terminate_early(
    rounds: list[SearchRound],
    *,
    remaining_budget_ratio: float = 1.0,
) -> bool:
    """Determine whether additional search rounds can be skipped.

    Returns ``True`` when the search shows diminishing returns:

    - **Yield decay**: the current round produced ≤ 50% of the new
      unique URLs that the previous round found (or zero hits), AND
    - **Gap stall**: the number of missing coverage dimensions did not
      decrease between rounds.

    Critical-gap guard: keeps searching when core authority / identity /
    facet-fit gaps are still open — *unless* the sub-question time budget
    is almost exhausted (``remaining_budget_ratio < _EARLY_STOP_BUDGET_FLOOR``),
    in which case early termination is allowed to prevent latency blowout.
    """
    if len(rounds) < 2:
        return False
    previous_round = rounds[-2]
    current_round = rounds[-1]
    if current_round.sufficient:
        return False
    previous_gap_count = _gap_count(previous_round)
    current_gap_count = _gap_count(current_round)
    if previous_gap_count is None or current_gap_count is None:
        return False
    # When both rounds produced zero new URLs, further rounds are futile
    # (e.g. all search providers are failing).  Continuing only wastes
    # budget and amplifies rate-limit pressure.
    if previous_round.new_unique_urls == 0 and current_round.new_unique_urls == 0:
        return True
    current_features = current_round.features
    # Only block early termination for critical gaps when budget still allows
    # another round.  When budget is nearly exhausted, let it stop gracefully.
    budget_allows_guard = remaining_budget_ratio >= _EARLY_STOP_BUDGET_FLOOR
    if budget_allows_guard and current_gap_count > 0 and current_features is not None:
        if any(
            dim in {"authority", "entity_identity", "facet_coverage", "task_fit"}
            for dim in current_features.missing_dimensions
        ):
            return False
        if current_features.authority_source_count == 0:
            return False
        if current_features.noisy_source_count > max(current_features.source_count // 2, 1):
            return False

    prev_yield = previous_round.new_unique_urls
    curr_yield = current_round.new_unique_urls
    yield_decayed = curr_yield == 0 or (prev_yield > 0 and curr_yield <= prev_yield * 0.5)
    gap_stalled = current_gap_count >= previous_gap_count
    return yield_decayed and gap_stalled


# Disambiguation query budget: max queries returned to the caller.
_DISAMBIGUATION_QUERY_BUDGET = 3
# Max context terms composed with the entity anchor per query.
_DISAMBIGUATION_CONTEXT_TERMS = 4
# Min token length to be considered a meaningful context term.
# Single-character tokens (articles, CJK particles) add noise.
_MIN_CONTEXT_TOKEN_LEN = 2
# Min token length for English-only extraction (skip 2-letter prepositions).
_MIN_ASCII_TOKEN_LEN = 3


def _is_cjk_char(ch: str) -> bool:
    """Return True when ``ch`` is a CJK unified ideograph or common extension.

    Scope is intentionally narrow to the ranges actually seen in planner
    output (Basic Multilingual Plane CJK + Extension A + Compatibility).
    Symbols, punctuation, and Latin characters all return False so the
    hygiene logic can distinguish mixed-script queries from CJK-only ones.
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF


def _apply_query_hygiene(queries: Iterable[str]) -> tuple[list[str], int]:
    """Drop search queries that cannot contribute useful retrieval signal.

    Applied rules, in order, all case-insensitive:

    1. **Repeated token**: queries with two or more tokens where any token
       appears more than once (e.g. ``"X X"``).  These are almost always
       planner bugs and waste provider quota.
    2. **Filler/stopword-only**: queries whose every token is drawn from
       the shared taxonomy vocabularies (filler fragments, CJK
       interrogatives, English WH-words).  Nothing informative remains.
    3. **CJK collapse**: queries that contain CJK characters where removing
       noise tokens leaves fewer than two content tokens *and* at least one
       token was removed.  Pure-content short queries (e.g. a lone domain
       term) are preserved because they may still be useful.

    Hygiene is purely reductive: each kept query is returned unchanged, in
    its original order, so downstream dedup and budget logic observes a
    strict subset of what the planner proposed.

    Args:
        queries: iterable of raw query strings.

    Returns:
        Tuple ``(kept_queries, dropped_count)``.  ``dropped_count`` also
        counts blank / whitespace-only entries.
    """
    filler_lower = {token.lower() for token in QUERY_HYGIENE_FILLER_TOKENS}
    en_stopwords = {token.lower() for token in QUERY_HYGIENE_EN_STOPWORDS}
    cjk_stopwords = set(QUERY_HYGIENE_CJK_STOPWORDS)

    def _is_noise_token(raw: str) -> bool:
        lowered = raw.lower()
        return lowered in filler_lower or lowered in en_stopwords or raw in cjk_stopwords

    kept: list[str] = []
    dropped = 0
    for query in queries:
        if not query or not query.strip():
            dropped += 1
            continue
        stripped = query.strip()
        tokens = stripped.split()
        if not tokens:
            dropped += 1
            continue
        lowered_tokens = [token.lower() for token in tokens]
        if len(tokens) >= 2 and len(set(lowered_tokens)) < len(lowered_tokens):
            dropped += 1
            continue
        content_tokens = [token for token in tokens if not _is_noise_token(token)]
        if not content_tokens:
            dropped += 1
            continue
        has_cjk = any(_is_cjk_char(ch) for ch in stripped)
        if has_cjk and len(content_tokens) < 2 and len(content_tokens) < len(tokens):
            dropped += 1
            continue
        kept.append(stripped)
    return kept, dropped


def _make_disambiguation_queries(
    question: str,
    user_query: str,
) -> list[str]:
    """Generate deterministic disambiguation queries from user_query context.

    Called when LLM-generated queries are all deduped in a later round.
    Composes the entity anchor with non-anchor context tokens from the
    original user query, plus the raw sub-question and an English variant.

    Limitation: whitespace tokenisation is ineffective for unsegmented CJK
    queries.  A future improvement should use a lightweight tokeniser.
    """
    from houyi.application.research.runtime.search_query_planner import (
        _extract_entity_anchor,
    )

    anchor = _extract_entity_anchor(question, user_query)
    if not anchor:
        return []
    # Extract context terms from user_query that are NOT just the anchor.
    context_tokens = [
        tok
        for tok in user_query.split()
        if tok.strip() and tok.strip().lower() != anchor.lower() and anchor not in tok
    ]
    # Filter trivially short tokens (articles, particles).
    context_terms = [tok for tok in context_tokens if len(tok) >= _MIN_CONTEXT_TOKEN_LEN]
    queries: list[str] = []
    if context_terms:
        context_chunk = " ".join(context_terms[:_DISAMBIGUATION_CONTEXT_TERMS])
        queries.append(f"{anchor} {context_chunk}".strip())
    # Raw sub-question carries the richest context from the planner.
    if question and question not in queries:
        queries.append(question.strip())
    # English variant: extract ASCII terms from user_query + anchor.
    ascii_terms = [
        tok
        for tok in user_query.split()
        if tok.isascii() and any(ch.isalpha() for ch in tok) and len(tok) >= _MIN_ASCII_TOKEN_LEN
    ]
    if ascii_terms:
        en_query = f"{anchor} {' '.join(ascii_terms[:_DISAMBIGUATION_CONTEXT_TERMS])}".strip()
        if en_query not in queries:
            queries.append(en_query)
    return queries[:_DISAMBIGUATION_QUERY_BUDGET]
