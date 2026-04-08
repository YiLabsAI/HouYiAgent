from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from houyi.application.research.types import SufficiencyDecision, SufficiencyFeatures

SearchEventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class TelemetryEmitter:
    notify: SearchEventCallback

    async def budget_consumed(
        self,
        *,
        question_id: str,
        round_index: int,
        layer: str,
        reason_code: str,
        budget_ms: int,
        remaining_ms: int,
    ) -> None:
        await self.notify(
            "search.budget_consumed",
            {
                "question_id": question_id,
                "round": round_index,
                "layer": layer,
                "reason_code": reason_code,
                "budget_ms": budget_ms,
                "remaining_ms": remaining_ms,
            },
        )

    async def sufficiency_features(
        self,
        *,
        question_id: str,
        round_index: int,
        features: SufficiencyFeatures,
    ) -> None:
        await self.notify(
            "search.sufficiency_features",
            {
                "question_id": question_id,
                "round": round_index,
                "source_count": features.source_count,
                "relevant_source_count": features.relevant_source_count,
                "domain_count": features.domain_count,
                "provider_count": features.provider_count,
                "authority_source_count": features.authority_source_count,
                "recent_source_count": features.recent_source_count,
                "relevance_score": features.relevance_score,
                "diversity_score": features.diversity_score,
                "authority_score": features.authority_score,
                "recency_score": features.recency_score,
                "missing_dimensions": list(features.missing_dimensions),
            },
        )

    async def sufficiency_decision(
        self,
        *,
        question_id: str,
        round_index: int,
        decision: SufficiencyDecision,
    ) -> None:
        await self.notify(
            "search.sufficiency_decision",
            {
                "question_id": question_id,
                "round": round_index,
                "sufficient": decision.sufficient,
                "rationale": decision.rationale,
                "decision_by": decision.decision_by,
                "reason_code": decision.reason_code,
                "missing_dimensions": list(decision.missing_dimensions),
            },
        )

    async def round_timing(
        self,
        *,
        question_id: str,
        round_number: int,
        elapsed_ms: float,
        query_count: int,
        skipped_queries: int,
        cancelled_queries: int,
        hit_count: int,
        source_count: int,
        decision: SufficiencyDecision,
        stop_layer: str,
    ) -> None:
        await self.notify(
            "search.round_timing",
            {
                "question_id": question_id,
                "round": round_number,
                "elapsed_ms": elapsed_ms,
                "query_count": query_count,
                "skipped_queries": skipped_queries,
                "cancelled_queries": cancelled_queries,
                "hit_count": hit_count,
                "source_count": source_count,
                "sufficient": decision.sufficient,
                "rationale": decision.rationale,
                "decision_by": decision.decision_by,
                "reason_code": decision.reason_code,
                "stop_layer": stop_layer,
                "missing_dimensions": list(decision.missing_dimensions),
            },
        )

    async def query_timing(
        self,
        *,
        question_id: str,
        round_index: int,
        query: str,
        elapsed_ms: float,
        hit_count: int,
        provider: str,
        reason_code: str,
    ) -> None:
        await self.notify(
            "search.query_timing",
            {
                "question_id": question_id,
                "round": round_index,
                "query": query,
                "elapsed_ms": round(elapsed_ms, 1),
                "hit_count": hit_count,
                "cancelled": False,
                "provider": provider,
                "reason_code": reason_code,
            },
        )

    async def query_cancelled(
        self,
        question_id: str,
        round_index: int,
        queries: list[str],
        *,
        reason: str,
    ) -> None:
        for pending_query in queries:
            await self.notify(
                "search.query_cancelled",
                {
                    "question_id": question_id,
                    "round": round_index,
                    "query": pending_query,
                    "reason": reason,
                },
            )
