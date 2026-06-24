"""Deterministic entity-state conflict resolution (the consolidator).

The write path is deliberately append-only: upsert never closes the prior
active row of a single-valued attribute, so a superseded value
(job: banker -> designer) leaves two active rows that recall would surface as
contradictory candidates. The consolidator repairs the materialized current-
view contract off the hot path: it scans triples with >=2 active rows, keeps
the newest (max valid_from) for single-valued attributes, and closes the rest
by setting valid_to to the successor's valid_from (bi-temporal supersession,
preserving as-of queries).

This is the deterministic, zero-LLM counterpart to the LLM reflector. It runs
before reflection inside MemoryEngine.evolve so every evolution run first
repairs structural contradictions, then reflects. It is idempotent (the
valid_to IS NULL guard), costs no LLM tokens, and is bounded by the partial
index, so it is cheap to run every extraction batch.

Design references: Graphiti/Zep bi-temporal supersession lineage. The
consolidator is its deterministic core. Policy (accumulate detection,
keep-newest) lives here; storage (row close + memories propagation) lives in
the EntityStateView.supersede method, keeping the consolidator backend-
agnostic and free of any retriever dependency (no circular coupling with
recall).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from houyi.adapters.memory.backends.base import EntityStateView


@dataclass(frozen=True, slots=True)
class ConsolidationReport:
    """Result of one entity-state consolidation run."""

    triples_scanned: int = 0
    triples_resolved: int = 0
    rows_closed: int = 0
    rows_propagated: int = 0
    skipped_accumulate: int = 0
    duration_ms: float = 0.0


class Consolidator(Protocol):
    """Repair structural contradictions in stored memory state."""

    def consolidate(self, *, namespace: str | None = None) -> ConsolidationReport: ...


def _is_accumulate(record) -> bool:
    """A row is part of an open set when its qualifiers tag accumulate=true."""
    qualifiers = getattr(record, "qualifiers", None)
    return isinstance(qualifiers, dict) and qualifiers.get("accumulate") == "true"


class EntityStateConsolidator:
    """Close superseded active rows so single-valued attributes converge.

    For each triple with >=2 active rows:
    - If any active row is tagged accumulate, the attribute is an open set
      (multiple concurrent values are expected); skip it to avoid data loss.
    - Otherwise the attribute is single-valued: keep the row with the greatest
      valid_from (the successor) and close every other active row, propagating
      the closure to the backing memories row so FTS and vector recall agree.

    valid_from is unique per triple (UNIQUE constraint), so the successor is
    unambiguous; there is no tie to break.
    """

    def __init__(self, view: EntityStateView) -> None:
        if view is None:
            raise ValueError("view is required")
        self._view = view

    def consolidate(self, *, namespace: str | None = None) -> ConsolidationReport:
        started = time.perf_counter()
        triples = self._view.list_conflicted_triples(namespace)
        rows_closed = 0
        rows_propagated = 0
        resolved = 0
        skipped_accumulate = 0

        for ns, entity, attribute in triples:
            active = self._view.get_active(ns, entity, attribute)
            if len(active) < 2:
                # Raced with a concurrent write that already resolved it.
                continue
            if any(_is_accumulate(record) for record in active):
                skipped_accumulate += 1
                continue
            keeper = max(active, key=lambda record: record.valid_from)
            closed, propagated = self._view.supersede(
                ns,
                entity,
                attribute,
                keep_state_id=keeper.state_id,
                valid_to=keeper.valid_from,
            )
            rows_closed += closed
            rows_propagated += propagated
            if closed > 0:
                resolved += 1

        return ConsolidationReport(
            triples_scanned=len(triples),
            triples_resolved=resolved,
            rows_closed=rows_closed,
            rows_propagated=rows_propagated,
            skipped_accumulate=skipped_accumulate,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )


__all__ = [
    "ConsolidationReport",
    "Consolidator",
    "EntityStateConsolidator",
]
