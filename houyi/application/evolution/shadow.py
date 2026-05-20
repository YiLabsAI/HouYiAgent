"""Shadow evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from houyi.application.evolution.artifacts import CandidateVariant, EvolutionArtifact
from houyi.application.evolution.dataset import EvolutionDataset
from houyi.application.evolution.evaluation import EvolutionEvaluator

SHADOW_VERDICT_PROMOTE = "promote"
SHADOW_VERDICT_HOLD = "hold"
SHADOW_VERDICT_REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ShadowReport:
    active_score: float
    shadow_score: float
    delta: float
    sample_size: int
    holdout_size: int
    verdict: str
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


class ShadowEvaluator(Protocol):
    def compare(
        self,
        active: EvolutionArtifact,
        candidate: CandidateVariant,
        dataset: EvolutionDataset,
    ) -> ShadowReport: ...


@dataclass(slots=True)
class DatasetShadowEvaluator:
    evaluator: EvolutionEvaluator
    min_delta: float = 0.01
    regression_tolerance: float = 0.0

    def compare(
        self,
        active: EvolutionArtifact,
        candidate: CandidateVariant,
        dataset: EvolutionDataset,
    ) -> ShadowReport:
        active_variant = CandidateVariant(artifact=active, score=0.0)
        evaluations = self.evaluator.evaluate([active_variant, candidate], dataset)
        if len(evaluations) < 2:
            return ShadowReport(
                active_score=0.0,
                shadow_score=0.0,
                delta=0.0,
                sample_size=len(dataset.train) + len(dataset.holdout),
                holdout_size=len(dataset.holdout),
                verdict=SHADOW_VERDICT_REJECT,
                reason="evaluator_returned_insufficient_results",
            )
        active_score = float(evaluations[0].score)
        shadow_score = float(evaluations[1].score)
        delta = shadow_score - active_score
        sample = len(dataset.train) + len(dataset.holdout)
        metrics = {
            "active_score": active_score,
            "shadow_score": shadow_score,
            "delta": delta,
            "min_delta": self.min_delta,
        }
        if delta < -self.regression_tolerance:
            return ShadowReport(
                active_score=active_score,
                shadow_score=shadow_score,
                delta=delta,
                sample_size=sample,
                holdout_size=len(dataset.holdout),
                verdict=SHADOW_VERDICT_REJECT,
                reason="shadow_regressed_against_active",
                metrics=metrics,
            )
        if delta >= self.min_delta:
            return ShadowReport(
                active_score=active_score,
                shadow_score=shadow_score,
                delta=delta,
                sample_size=sample,
                holdout_size=len(dataset.holdout),
                verdict=SHADOW_VERDICT_PROMOTE,
                reason="shadow_beat_active",
                metrics=metrics,
            )
        return ShadowReport(
            active_score=active_score,
            shadow_score=shadow_score,
            delta=delta,
            sample_size=sample,
            holdout_size=len(dataset.holdout),
            verdict=SHADOW_VERDICT_HOLD,
            reason="shadow_non_regressive_below_margin",
            metrics=metrics,
        )


def candidate_with_score(candidate: CandidateVariant, new_score: float) -> CandidateVariant:
    return replace(candidate, score=new_score)


__all__ = [
    "SHADOW_VERDICT_HOLD",
    "SHADOW_VERDICT_PROMOTE",
    "SHADOW_VERDICT_REJECT",
    "DatasetShadowEvaluator",
    "ShadowEvaluator",
    "ShadowReport",
    "candidate_with_score",
]
