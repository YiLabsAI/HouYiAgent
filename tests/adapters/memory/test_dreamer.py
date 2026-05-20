from __future__ import annotations

import pytest

from houyi.adapters.memory.dreamer import EvolutionBudget, EvolutionStage, MemoryDreamer
from houyi.adapters.memory.types import MemoryRecord, MemoryScope, MemoryType


def make_record(
    key: str,
    content: str,
    *,
    memory_type: MemoryType = MemoryType.FACT,
) -> MemoryRecord:
    return MemoryRecord(
        scope=MemoryScope.USER,
        key=key,
        content=content,
        memory_type=memory_type,
        confidence=0.8,
    )


def test_dreamer_observes() -> None:
    report = MemoryDreamer().run(
        [
            make_record("a", "user likes coffee", memory_type=MemoryType.PREFERENCE),
            make_record("b", "user likes tea", memory_type=MemoryType.PREFERENCE),
        ]
    )

    assert report.total_units_scanned == 2
    assert report.observations_created == 1
    assert report.created_records[0].metadata["unit_kind"] == "observation"


def test_dreamer_reflects() -> None:
    report = MemoryDreamer().run(
        [
            make_record("a", "user likes coffee", memory_type=MemoryType.PREFERENCE),
            make_record("b", "user likes tea", memory_type=MemoryType.PREFERENCE),
            make_record("c", "user works remotely", memory_type=MemoryType.PROFILE),
            make_record("d", "user lives in Berlin", memory_type=MemoryType.PROFILE),
        ]
    )

    assert report.observations_created == 2
    assert report.reflections_created == 1
    assert report.created_records[-1].metadata["unit_kind"] == "reflection"


def test_dreamer_budget() -> None:
    records = [make_record(str(i), f"memory {i}") for i in range(3)]

    report = MemoryDreamer().run(records, budget=EvolutionBudget(max_units=2))

    assert report.total_units_scanned == 2
    assert report.truncated is True


def test_dreamer_rejects_budget() -> None:
    with pytest.raises(ValueError):
        MemoryDreamer().run([], budget=EvolutionBudget(max_units=0))


def test_dreamer_improves_quality() -> None:
    records = [
        make_record("a", "user likes coffee", memory_type=MemoryType.PREFERENCE),
        make_record("b", "user likes tea", memory_type=MemoryType.PREFERENCE),
    ]

    report = MemoryDreamer().run(records)

    assert report.quality_score_after > report.quality_score_before


def test_dreamer_stages() -> None:
    report = MemoryDreamer().run(
        [
            make_record("a", "user likes coffee", memory_type=MemoryType.PREFERENCE),
            make_record("b", "user likes tea", memory_type=MemoryType.PREFERENCE),
        ]
    )

    assert report.stages == (
        EvolutionStage.SAMPLE,
        EvolutionStage.REFLECT,
        EvolutionStage.MUTATE,
        EvolutionStage.EVALUATE,
        EvolutionStage.PROMOTE,
        EvolutionStage.CALIBRATE,
    )


def test_dreamer_promotes() -> None:
    report = MemoryDreamer().run(
        [
            make_record("a", "user likes coffee", memory_type=MemoryType.PREFERENCE),
            make_record("b", "user likes tea", memory_type=MemoryType.PREFERENCE),
        ]
    )

    assert report.samples
    assert report.proposals
    assert report.mutations
    assert report.evaluations
    assert all(decision.promoted for decision in report.promotions)
    assert report.calibration.accepted_count == len(report.promotions)


def test_dreamer_rejects() -> None:
    report = MemoryDreamer().run(
        [
            make_record("a", "user likes coffee", memory_type=MemoryType.PREFERENCE),
            make_record("b", "user likes tea", memory_type=MemoryType.PREFERENCE),
        ],
        budget=EvolutionBudget(min_quality_delta=1.0),
    )

    assert report.created_records == ()
    assert all(not decision.promoted for decision in report.promotions)
    assert report.calibration.rejected_count == len(report.promotions)
