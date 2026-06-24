from __future__ import annotations

from pathlib import Path

from houyi.adapters.memory.dreamer import EvolutionRunReport
from houyi.adapters.memory.dreamer_report import build_before_after, write_evolution_report
from houyi.adapters.memory.types import MemoryRecord, MemoryScope, MemoryType


def _records() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            scope=MemoryScope.USER,
            key=key,
            content=content,
            memory_type=MemoryType.PREFERENCE,
            confidence=0.8,
        )
        for key, content in (("a", "likes coffee"), ("b", "likes tea"))
    ]


def _report() -> EvolutionRunReport:
    records = _records()
    return EvolutionRunReport(
        total_units_scanned=len(records),
        quality_score_before=0.3,
        quality_score_after=0.6,
        created_records=tuple(records),
    )


def test_report_promotes_gain() -> None:
    before_after = build_before_after(_report(), run_id="r1")

    assert before_after.verdict == "promote"
    assert before_after.delta > 0
    assert before_after.optimizer == "memory_dreamer"


def test_write_report_emits_files(tmp_path: Path) -> None:
    md_path = write_evolution_report(_report(), tmp_path, run_id="r2")

    assert md_path.exists()
    assert (tmp_path / "before_after.json").exists()
    assert "memory_dreamer" in md_path.read_text(encoding="utf-8")
