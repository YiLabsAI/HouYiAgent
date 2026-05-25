"""17-cell HouYi memory benchmark matrix + formal runner ().

This module owns three concerns:

1. Cell metadata — the canonical 17-cell matrix with each cell's
 dataset (which fixture / benchmark family is the source of
 truth) and evidence (test path or external dataset note).
2. CellRunner — a CI-friendly driver that takes a check callable
 per cell and produces a deterministic CellReport. Without any
 custom checks the runner falls back to the static evidence
 already bundled with each cell, which lets make check print a
 stable matrix even when external benchmarks (LoCoMo / HaluMem)
 have not been run.
3. Report writers — write_cells_report emits {cells.json,
 summary.md} under benchmark/output/memory/{run_id}/.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Literal

CellStatus = Literal["passed", "partial", "pending", "failed"]


class CellDataset(str, Enum):
    """Source-of-truth dataset family for a cell.

    The cell-to-dataset mapping lets a status report say something
    like "R1 passed against LoCoMo SH" instead of "R1 passed against
    an internal smoke test".
    """

    INTERNAL = "internal"
    """Pure-unit-test smoke (default for cells whose external dataset
 has not been wired yet). Internal cells fall back to the static
 evidence string; the runner never marks them failed on its
 own."""
    LOCOMO = "locomo"
    HALUMEM = "halumem"
    ADVERSARIAL = "adversarial"
    DREAMER = "dreamer"


@dataclass(frozen=True, slots=True)
class MemoryBenchCell:
    """One benchmark cell and its current local evidence status."""

    cell_id: str
    title: str
    status: CellStatus
    evidence: str
    dataset: CellDataset = CellDataset.INTERNAL


_CELL_MATRIX: tuple[MemoryBenchCell, ...] = (
    MemoryBenchCell("E1", "atomic extraction", "passed", "test_ingestor.py"),
    MemoryBenchCell("E2", "source anchoring", "passed", "test_ingestor.py"),
    MemoryBenchCell("E3", "sourceless routing", "passed", "test_ingestor.py"),
    MemoryBenchCell("E4", "vague candidate routing", "passed", "test_resolver.py"),
    MemoryBenchCell("E5", "retraction handling", "passed", "test_retraction.py"),
    MemoryBenchCell("U1", "entity update", "passed", "test_entity_state_view.py"),
    MemoryBenchCell("U2", "conflict invalidation", "passed", "test_resolver.py"),
    MemoryBenchCell(
        "U3",
        "update benchmark smoke",
        "partial",
        "test_memory_halumem.py",
        CellDataset.HALUMEM,
    ),
    MemoryBenchCell(
        "R1",
        "single-hop recall",
        "passed",
        "test_memory.py::test_single_hop",
        CellDataset.LOCOMO,
    ),
    MemoryBenchCell(
        "R2",
        "multi-hop recall",
        "passed",
        "test_memory.py::test_multi_hop",
        CellDataset.LOCOMO,
    ),
    MemoryBenchCell(
        "R3",
        "temporal recall",
        "passed",
        "test_memory.py::test_temporal_as_of",
        CellDataset.LOCOMO,
    ),
    MemoryBenchCell("R4", "unknown abstention", "passed", "test_memory.py::test_unknown_idk"),
    MemoryBenchCell("R5", "source fallback", "passed", "test_memory.py::test_source_fallback"),
    MemoryBenchCell(
        "R6",
        "adversarial recall",
        "partial",
        "test_adversarial.py",
        CellDataset.ADVERSARIAL,
    ),
    MemoryBenchCell(
        "Q1",
        "HaluMem QA",
        "pending",
        "real HaluMem QA run",
        CellDataset.HALUMEM,
    ),
    MemoryBenchCell("D1", "UEA observation", "passed", "test_dreamer.py", CellDataset.DREAMER),
    MemoryBenchCell("D2", "UEA reflection", "passed", "test_dreamer.py", CellDataset.DREAMER),
)


def cell_matrix() -> tuple[MemoryBenchCell, ...]:
    """Return the fixed 17-cell local matrix."""
    return _CELL_MATRIX


def cell_pass_rate() -> float:
    """Return passed-cell ratio over the local 17-cell matrix."""
    cells = cell_matrix()
    passed = sum(1 for cell in cells if cell.status == "passed")
    return passed / len(cells)


# ---------------------------------------------------------------------------
# CellRunner — formal driver + JSON / Markdown report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellOutcome:
    """Result of running one cell's check."""

    cell: MemoryBenchCell
    status: CellStatus
    detail: str
    duration_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell.cell_id,
            "title": self.cell.title,
            "dataset": self.cell.dataset.value,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.cell.evidence,
            "duration_s": round(self.duration_s, 4),
        }


@dataclass
class CellReport:
    """Aggregate of a 17-cell run."""

    run_id: str
    outcomes: list[CellOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "passed")

    @property
    def pass_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return self.passed / self.total

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "by_status": _count_by_status(self.outcomes),
            "by_dataset": _count_by_dataset(self.outcomes),
            "cells": [o.to_dict() for o in self.outcomes],
        }


CellCheck = Callable[[MemoryBenchCell], "CellCheckResult"]
"""Per-cell pluggable check — returns (status, detail)."""


@dataclass(frozen=True)
class CellCheckResult:
    status: CellStatus
    detail: str = ""


def _static_check(cell: MemoryBenchCell) -> CellCheckResult:
    """Default check — surfaces the cell's static evidence verbatim."""
    return CellCheckResult(status=cell.status, detail=cell.evidence)


def _count_by_status(outcomes: list[CellOutcome]) -> dict[str, int]:
    out: dict[str, int] = {}
    for o in outcomes:
        out[o.status] = out.get(o.status, 0) + 1
        return out


def _count_by_dataset(outcomes: list[CellOutcome]) -> dict[str, int]:
    out: dict[str, int] = {}
    for o in outcomes:
        out[o.cell.dataset.value] = out.get(o.cell.dataset.value, 0) + 1
        return out


class CellRunner:
    """Formal runner for the 17-cell matrix.

    The runner is intentionally synchronous: each cell's check is a
    deterministic callable so the matrix can run inside make check
    without an event loop. External-benchmark cells (LoCoMo / HaluMem)
    wire their async harness behind a sync wrapper before handing the
    closure here.
    """

    def __init__(
        self,
        cells: tuple[MemoryBenchCell, ...] | None = None,
        *,
        checks: Mapping[str, CellCheck] | None = None,
    ) -> None:
        self._cells = cells if cells is not None else cell_matrix()
        self._checks: dict[str, CellCheck] = dict(checks or {})

    def run(self, *, run_id: str | None = None) -> CellReport:
        rid = run_id or _make_run_id()
        report = CellReport(run_id=rid)
        for cell in self._cells:
            check = self._checks.get(cell.cell_id, _static_check)
            t0 = time.perf_counter()
            try:
                result = check(cell)
            except Exception as exc:  # surface failure as failed cell
                result = CellCheckResult(
                    status="failed",
                    detail=f"check raised: {type(exc).__name__}: {exc}"[:300],
                )
            outcome = CellOutcome(
                cell=cell,
                status=result.status,
                detail=result.detail or cell.evidence,
                duration_s=time.perf_counter() - t0,
            )
            report.outcomes.append(outcome)
        return report


def _make_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_cells_report(
    report: CellReport,
    output_root: str | Path,
) -> Path:
    """Write {run_id}/cells.json and {run_id}/summary.md.

    Returns the directory the files landed in. The path layout is
    benchmark/output/memory/{run_id}/.
    """
    root = Path(output_root) / report.run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "cells.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "summary.md").write_text(_render_markdown(report), encoding="utf-8")
    return root


def _render_markdown(report: CellReport) -> str:
    lines: list[str] = []
    lines.append(f"# HouYi Memory Bench — 17-cell report `{report.run_id}`")
    lines.append("")
    lines.append(f"**Pass rate**: {report.passed}/{report.total} = {report.pass_rate * 100:.1f}%")
    lines.append("")
    by_status = _count_by_status(report.outcomes)
    if by_status:
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items()))
        lines.append(f"**By status**: {parts}")
        lines.append("")
        lines.append("| Cell | Title | Dataset | Status | Detail |")
        lines.append("|------|-------|---------|--------|--------|")
        for o in report.outcomes:
            detail = o.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {o.cell.cell_id} | {o.cell.title} | {o.cell.dataset.value} "
                f"| {o.status} | {detail} |"
            )
    lines.append("")
    return "\n".join(lines)


def override_status(
    cell: MemoryBenchCell, *, status: CellStatus, evidence: str | None = None
) -> MemoryBenchCell:
    """Return a copy of cell with a new status / evidence.

    Helpers like the LoCoMo runner produce a fresh status per cell;
    this keeps the underlying matrix immutable while letting callers
    feed updated rows into CellRunner for downstream JSON / Markdown
    rendering.
    """
    if evidence is None:
        return replace(cell, status=status)
    return replace(cell, status=status, evidence=evidence)


__all__ = [
    "CellCheck",
    "CellCheckResult",
    "CellDataset",
    "CellOutcome",
    "CellReport",
    "CellRunner",
    "CellStatus",
    "MemoryBenchCell",
    "cell_matrix",
    "cell_pass_rate",
    "override_status",
    "write_cells_report",
]
