"""Evolution run budget and report.

The memory evolution subsystem has two off-hot-path arms that run inside
MemoryEngine.evolve:

- the deterministic consolidator (dreamer_consolidate) -- zero-LLM
  bi-temporal supersession of contradicted single-valued attributes.
- the failure-anchored reflector (dreamer_reflect) -- LLM re-extraction of
  query-answering facts from source turns for queries recall failed to answer.

Both return their own report (ConsolidationReport / ReflectionReport),
which EvolutionRunReport aggregates alongside the persisted records. The
legacy six-stage MemoryDreamer pipeline (sample / reflect / mutate /
evaluate / promote / calibrate over already-extracted records) was removed: it
operated on flattened facts and could not recover semantics the generic
extractor lost, and its lexical-coverage evaluator was tautological with its
own echo-the-query reflector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from houyi.adapters.memory.types import MemoryRecord

if TYPE_CHECKING:
    from houyi.adapters.memory.dreamer_consolidate import ConsolidationReport
    from houyi.adapters.memory.dreamer_reflect import ReflectionReport


@dataclass(frozen=True, slots=True)
class EvolutionBudget:
    """Limits for one bounded evolution run."""

    max_units: int = 500
    min_quality_delta: float = 0.01


@dataclass(frozen=True, slots=True)
class EvolutionRunReport:
    """Aggregated result of one evolution run.

    consolidation and reflection carry the two arms' reports (None when
    that arm was disabled or had no view / failing queries). created_records
    holds the facts persisted by the reflector.
    """

    total_units_scanned: int = 0
    observations_created: int = 0
    reflections_created: int = 0
    duration_ms: float = 0.0
    quality_score_before: float = 0.0
    quality_score_after: float = 0.0
    created_records: tuple[MemoryRecord, ...] = ()
    consolidation: ConsolidationReport | None = None
    reflection: ReflectionReport | None = None


__all__ = ["EvolutionBudget", "EvolutionRunReport"]
