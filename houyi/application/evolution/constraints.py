from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from houyi.application.evolution.artifacts import CandidateVariant


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    passed: bool
    constraint_name: str
    message: str


class EvolutionConstraintGate(Protocol):
    def validate(self, candidate: CandidateVariant) -> list[ConstraintResult]: ...


@dataclass(frozen=True, slots=True)
class BasicConstraintGate:
    max_content_size: int = 15_000
    min_score: float = 0.0

    def validate(self, candidate: CandidateVariant) -> list[ConstraintResult]:
        content_size = len(candidate.artifact.content)
        return [
            ConstraintResult(
                passed=content_size <= self.max_content_size,
                constraint_name="content_size",
                message=f"content size {content_size}/{self.max_content_size}",
            ),
            ConstraintResult(
                passed=candidate.score >= self.min_score,
                constraint_name="candidate_score",
                message=f"candidate score {candidate.score:.4f} >= {self.min_score:.4f}",
            ),
        ]

    def filter_valid(self, candidates: list[CandidateVariant]) -> list[CandidateVariant]:
        return [
            candidate
            for candidate in candidates
            if all(result.passed for result in self.validate(candidate))
        ]
