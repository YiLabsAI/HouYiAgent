from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from houyi.application.evolution.artifacts import CandidateVariant, EvolutionArtifact
from houyi.application.evolution.dataset import EvolutionDataset
from houyi.application.evolution.policy_store import EvolutionPolicyStore
from houyi.application.evolution.shadow import (
    SHADOW_VERDICT_HOLD,
    SHADOW_VERDICT_PROMOTE,
    SHADOW_VERDICT_REJECT,
    ShadowEvaluator,
    ShadowReport,
    candidate_with_score,
)


class PromotionLevel(str, Enum):
    REJECTED = "rejected"
    SHADOW = "shadow"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    candidate: CandidateVariant | None
    level: PromotionLevel
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    shadow_report: ShadowReport | None = None


@dataclass(slots=True)
class PromotionManager:
    min_score: float = 0.01

    def stage_shadow(
        self,
        candidates: list[CandidateVariant],
        active: EvolutionArtifact,
        shadow_evaluator: ShadowEvaluator,
        dataset: EvolutionDataset,
        *,
        policy_store: EvolutionPolicyStore | None = None,
    ) -> PromotionDecision:
        if not candidates:
            return PromotionDecision(None, PromotionLevel.REJECTED, "no_candidates")
        best = max(candidates, key=lambda candidate: candidate.score)
        if best.score < self.min_score:
            return PromotionDecision(
                best,
                PromotionLevel.REJECTED,
                "score_below_threshold",
            )
        report = shadow_evaluator.compare(active, best, dataset)
        promoted = candidate_with_score(best, report.shadow_score)
        if report.verdict == SHADOW_VERDICT_PROMOTE:
            if policy_store is not None:
                policy_store.set_active(promoted.artifact)
            return PromotionDecision(
                promoted,
                PromotionLevel.ACTIVE,
                report.reason,
                metrics=dict(report.metrics),
                shadow_report=report,
            )
        if report.verdict == SHADOW_VERDICT_HOLD:
            if policy_store is not None:
                policy_store.set_shadow(promoted.artifact)
            return PromotionDecision(
                promoted,
                PromotionLevel.SHADOW,
                report.reason,
                metrics=dict(report.metrics),
                shadow_report=report,
            )
        if report.verdict == SHADOW_VERDICT_REJECT:
            return PromotionDecision(
                promoted,
                PromotionLevel.REJECTED,
                report.reason,
                metrics=dict(report.metrics),
                shadow_report=report,
            )
        return PromotionDecision(
            promoted,
            PromotionLevel.REJECTED,
            f"unknown_shadow_verdict:{report.verdict}",
            metrics=dict(report.metrics),
            shadow_report=report,
        )
