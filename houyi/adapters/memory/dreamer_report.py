"""Render a memory evolution run as a before/after report.

The control plane already ships a BeforeAfterReport for prompt-artifact
optimization. A memory evolution pass produces the same shape of evidence —
a quality score before and after, plus the notes that were promoted — so this
module maps an EvolutionRunReport onto that report and reuses its markdown /
JSON writer. This is how an evolution pass leaves an auditable, comparable
artifact on disk.
"""

from __future__ import annotations

from pathlib import Path

from houyi.adapters.memory.dreamer import EvolutionRunReport
from houyi.application.evolution.before_after import (
    BeforeAfterReport,
    make_run_id,
    write_report,
)

_OPTIMIZER = "memory_dreamer"
_ARTIFACT_TYPE = "memory_records"


def build_before_after(
    report: EvolutionRunReport,
    *,
    run_id: str | None = None,
) -> BeforeAfterReport:
    """Convert an evolution run into a BeforeAfterReport."""
    delta = report.quality_score_after - report.quality_score_before
    promoted = "\n".join(record.content for record in report.created_records) or "(none)"
    return BeforeAfterReport(
        run_id=run_id or make_run_id(),
        optimizer=_OPTIMIZER,
        artifact_type=_ARTIFACT_TYPE,
        baseline_content=f"{report.total_units_scanned} source memories",
        optimized_content=promoted,
        baseline_score=report.quality_score_before,
        optimized_score=report.quality_score_after,
        delta=round(delta, 4),
        sample_size=report.total_units_scanned,
        signal_count=len(report.created_records),
        verdict="promote" if delta > 0 else "hold",
        reason="quality_gain" if delta > 0 else "no_gain",
        metrics={
            "observations_created": float(report.observations_created),
            "reflections_created": float(report.reflections_created),
            "duration_ms": round(report.duration_ms, 4),
        },
    )


def write_evolution_report(
    report: EvolutionRunReport,
    output_dir: Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Write before_after.md / .json for an evolution run; return the md path."""
    return write_report(build_before_after(report, run_id=run_id), output_dir)


__all__ = ["build_before_after", "write_evolution_report"]
