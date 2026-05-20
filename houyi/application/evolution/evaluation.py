from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from houyi.application.evolution.artifacts import CandidateVariant
from houyi.application.evolution.dataset import EvolutionDataset


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate: CandidateVariant
    score: float
    passed: bool
    reason: str


class EvolutionEvaluator(Protocol):
    def evaluate(
        self,
        candidates: list[CandidateVariant],
        dataset: EvolutionDataset,
    ) -> list[CandidateEvaluation]: ...


class HeuristicEvolutionEvaluator:
    def evaluate(
        self,
        candidates: list[CandidateVariant],
        dataset: EvolutionDataset,
    ) -> list[CandidateEvaluation]:
        holdout_bonus = 0.01 if dataset.holdout else 0.0
        return [
            CandidateEvaluation(
                candidate=candidate,
                score=round(candidate.score + holdout_bonus, 4),
                passed=candidate.score > 0.0,
                reason="heuristic_score_positive"
                if candidate.score > 0.0
                else "heuristic_score_non_positive",
            )
            for candidate in candidates
        ]
