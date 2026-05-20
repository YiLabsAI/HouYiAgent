"""Deterministic memory evolution framework."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from houyi.adapters.memory.types import MemoryRecord, MemoryScope, MemoryType


class EvolutionStage(str, Enum):
    """Stage names emitted by one memory evolution run."""

    SAMPLE = "sample"
    REFLECT = "reflect"
    MUTATE = "mutate"
    EVALUATE = "evaluate"
    PROMOTE = "promote"
    CALIBRATE = "calibrate"


@dataclass(frozen=True, slots=True)
class EvolutionBudget:
    """Limits for one bounded evolution run."""

    max_units: int = 500
    min_quality_delta: float = 0.01


@dataclass(frozen=True, slots=True)
class EvolutionSample:
    """A sampled cluster selected for evolution."""

    sample_id: str
    scope: MemoryScope
    memory_type: MemoryType
    records: tuple[MemoryRecord, ...]
    priority: float


@dataclass(frozen=True, slots=True)
class ReflectionProposal:
    """A proposed higher-level memory derived from one sample."""

    proposal_id: str
    sample_id: str
    unit_kind: str
    content: str
    confidence: float
    source_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryMutation:
    """Append-only mutation candidate generated from a proposal."""

    mutation_id: str
    proposal_id: str
    record: MemoryRecord


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Quality assessment for one mutation candidate."""

    mutation_id: str
    quality_before: float
    quality_after: float
    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Promotion outcome for one evaluated mutation."""

    mutation_id: str
    promoted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class CalibrationUpdate:
    """Policy feedback produced after promotion decisions."""

    accepted_count: int = 0
    rejected_count: int = 0
    next_min_quality_delta: float = 0.01


@dataclass(frozen=True, slots=True)
class EvolutionRunReport:
    """Result summary for one deterministic evolution run."""

    total_units_scanned: int = 0
    observations_created: int = 0
    reflections_created: int = 0
    duration_ms: float = 0.0
    quality_score_before: float = 0.0
    quality_score_after: float = 0.0
    samples: tuple[EvolutionSample, ...] = ()
    proposals: tuple[ReflectionProposal, ...] = ()
    mutations: tuple[MemoryMutation, ...] = ()
    evaluations: tuple[EvaluationResult, ...] = ()
    promotions: tuple[PromotionDecision, ...] = ()
    calibration: CalibrationUpdate = CalibrationUpdate()
    created_records: tuple[MemoryRecord, ...] = ()
    truncated: bool = False
    stages: tuple[EvolutionStage, ...] = (
        EvolutionStage.SAMPLE,
        EvolutionStage.REFLECT,
        EvolutionStage.MUTATE,
        EvolutionStage.EVALUATE,
        EvolutionStage.PROMOTE,
        EvolutionStage.CALIBRATE,
    )


class MemoryDreamer:
    """Run append-only memory evolution without mutating source records."""

    def run(
        self,
        records: list[MemoryRecord],
        *,
        budget: EvolutionBudget | None = None,
    ) -> EvolutionRunReport:
        started = time.perf_counter()
        active_budget = budget or EvolutionBudget()
        if active_budget.max_units <= 0:
            raise ValueError("EvolutionBudget.max_units must be > 0")

        selected = records[: active_budget.max_units]
        truncated = len(records) > len(selected)
        before = _quality_score(selected)
        samples = _sample(selected)
        proposals = _reflect(samples)
        mutations = _mutate(proposals)
        evaluations = _evaluate(
            selected,
            mutations,
            min_quality_delta=active_budget.min_quality_delta,
        )
        promotions = _promote(evaluations)
        created = _promoted_records(mutations, promotions)
        calibration = _calibrate(promotions, active_budget)
        after = _quality_score([*selected, *created])
        duration_ms = (time.perf_counter() - started) * 1000
        return EvolutionRunReport(
            total_units_scanned=len(selected),
            observations_created=sum(
                1 for record in created if record.metadata.get("unit_kind") == "observation"
            ),
            reflections_created=sum(
                1 for record in created if record.metadata.get("unit_kind") == "reflection"
            ),
            duration_ms=duration_ms,
            quality_score_before=before,
            quality_score_after=after,
            samples=samples,
            proposals=proposals,
            mutations=mutations,
            evaluations=evaluations,
            promotions=promotions,
            calibration=calibration,
            created_records=created,
            truncated=truncated,
        )


def _sample(records: list[MemoryRecord]) -> tuple[EvolutionSample, ...]:
    grouped: dict[tuple[MemoryScope, MemoryType], list[MemoryRecord]] = defaultdict(list)
    for record in records:
        if record.valid_to is not None:
            continue
        grouped[(record.scope, record.memory_type)].append(record)

    samples: list[EvolutionSample] = []
    for (scope, memory_type), group in grouped.items():
        if len(group) < 2:
            continue
        priority = len(group) * (sum(item.confidence for item in group) / len(group))
        samples.append(
            EvolutionSample(
                sample_id=f"{scope.value}:{memory_type.value}",
                scope=scope,
                memory_type=memory_type,
                records=tuple(group),
                priority=round(priority, 4),
            )
        )
    return tuple(sorted(samples, key=lambda sample: sample.priority, reverse=True))


def _reflect(samples: tuple[EvolutionSample, ...]) -> tuple[ReflectionProposal, ...]:
    proposals: list[ReflectionProposal] = []
    for sample in samples:
        content = _join_contents(list(sample.records), limit=3)
        proposals.append(
            ReflectionProposal(
                proposal_id=f"observation:{sample.sample_id}",
                sample_id=sample.sample_id,
                unit_kind="observation",
                content=f"Observation across {len(sample.records)} memories: {content}",
                confidence=_average_confidence(sample.records),
                source_record_ids=tuple(record.record_id for record in sample.records),
            )
        )
    if len(proposals) >= 2:
        content = "; ".join(proposal.content for proposal in proposals[:3])
        proposals.append(
            ReflectionProposal(
                proposal_id="reflection:summary",
                sample_id=",".join(proposal.sample_id for proposal in proposals),
                unit_kind="reflection",
                content=f"Reflection from {len(proposals)} observations: {content}",
                confidence=sum(proposal.confidence for proposal in proposals) / len(proposals),
                source_record_ids=tuple(
                    record_id for proposal in proposals for record_id in proposal.source_record_ids
                ),
            )
        )
    return tuple(proposals)


def _mutate(proposals: tuple[ReflectionProposal, ...]) -> tuple[MemoryMutation, ...]:
    mutations: list[MemoryMutation] = []
    for proposal in proposals:
        first_scope, first_type = _proposal_target(proposal)
        if proposal.unit_kind == "reflection":
            first_type = MemoryType.STRATEGY
        record = MemoryRecord(
            scope=first_scope,
            key=proposal.proposal_id,
            content=proposal.content,
            memory_type=first_type,
            confidence=proposal.confidence,
            tags=[proposal.unit_kind],
            metadata={
                "unit_kind": proposal.unit_kind,
                "source_record_ids": ",".join(proposal.source_record_ids),
            },
        )
        mutations.append(
            MemoryMutation(
                mutation_id=f"mutation:{proposal.proposal_id}",
                proposal_id=proposal.proposal_id,
                record=record,
            )
        )
    return tuple(mutations)


def _evaluate(
    baseline: list[MemoryRecord],
    mutations: tuple[MemoryMutation, ...],
    *,
    min_quality_delta: float,
) -> tuple[EvaluationResult, ...]:
    evaluations: list[EvaluationResult] = []
    before = _quality_score(baseline)
    for mutation in mutations:
        after = _quality_score([*baseline, mutation.record])
        delta = after - before
        accepted = delta >= min_quality_delta
        evaluations.append(
            EvaluationResult(
                mutation_id=mutation.mutation_id,
                quality_before=before,
                quality_after=after,
                accepted=accepted,
                reason="quality_delta_met" if accepted else "quality_delta_too_low",
            )
        )
    return tuple(evaluations)


def _promote(evaluations: tuple[EvaluationResult, ...]) -> tuple[PromotionDecision, ...]:
    return tuple(
        PromotionDecision(
            mutation_id=evaluation.mutation_id,
            promoted=evaluation.accepted,
            reason=evaluation.reason,
        )
        for evaluation in evaluations
    )


def _promoted_records(
    mutations: tuple[MemoryMutation, ...],
    promotions: tuple[PromotionDecision, ...],
) -> tuple[MemoryRecord, ...]:
    promoted_ids = {promotion.mutation_id for promotion in promotions if promotion.promoted}
    return tuple(mutation.record for mutation in mutations if mutation.mutation_id in promoted_ids)


def _calibrate(
    promotions: tuple[PromotionDecision, ...],
    budget: EvolutionBudget,
) -> CalibrationUpdate:
    accepted = sum(1 for promotion in promotions if promotion.promoted)
    rejected = len(promotions) - accepted
    next_delta = budget.min_quality_delta
    if rejected > accepted:
        next_delta = min(0.2, budget.min_quality_delta + 0.01)
    elif accepted > rejected and budget.min_quality_delta > 0.01:
        next_delta = max(0.01, budget.min_quality_delta - 0.01)
    return CalibrationUpdate(
        accepted_count=accepted,
        rejected_count=rejected,
        next_min_quality_delta=round(next_delta, 4),
    )


def _proposal_target(proposal: ReflectionProposal) -> tuple[MemoryScope, MemoryType]:
    sample_id = proposal.sample_id.split(",", maxsplit=1)[0]
    scope_value, memory_type_value = sample_id.split(":", maxsplit=1)
    return MemoryScope(scope_value), MemoryType(memory_type_value)


def _average_confidence(records: tuple[MemoryRecord, ...]) -> float:
    return min(1.0, sum(item.confidence for item in records) / len(records))


def _join_contents(records: list[MemoryRecord], *, limit: int) -> str:
    return "; ".join(record.content for record in records[:limit])


def _quality_score(records: list[MemoryRecord]) -> float:
    if not records:
        return 0.0
    active = [record for record in records if record.valid_to is None]
    source_rich = [record for record in active if record.provenance is not None or record.metadata]
    structured = [
        record
        for record in active
        if record.metadata.get("unit_kind") in {"observation", "reflection"}
    ]
    active_ratio = len(active) / len(records)
    source_ratio = len(source_rich) / len(active) if active else 0.0
    structure_ratio = min(1.0, len(structured) / max(1, len(active)))
    return round((0.5 * active_ratio) + (0.25 * source_ratio) + (0.25 * structure_ratio), 4)


__all__ = [
    "CalibrationUpdate",
    "EvaluationResult",
    "EvolutionBudget",
    "EvolutionRunReport",
    "EvolutionSample",
    "EvolutionStage",
    "MemoryDreamer",
    "MemoryMutation",
    "PromotionDecision",
    "ReflectionProposal",
]
