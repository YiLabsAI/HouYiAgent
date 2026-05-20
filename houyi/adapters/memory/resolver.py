"""Writer-side resolver facade used by the LLM-as-Writer agent loop.

Exposes four primitive operations that an LLM agent can invoke when it
decides how to absorb a freshly extracted AtomicFact:

- read_entity_state: inspect what is already known about an entity.
- write_unit: record a new active value (only when no active row
 exists for the triple).
- update_unit: supersede the currently active row with a new value.
- invalidate_unit: close the currently active row without a successor
 (used to honour explicit retraction signals).

The resolver is deliberately backend-agnostic: it depends only on the
EntityStateView and CandidateInbox abstract protocols, so any
storage backend (SQLite, Postgres, in-memory) can be plugged in without
touching the business logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from houyi.adapters.memory.backends.base import CandidateInbox, EntityStateView
from houyi.adapters.memory.types import AtomicFact, Certainty, EntityStateRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors and result types
# ---------------------------------------------------------------------------


class WriterToolError(RuntimeError):
    """Base class for writer-tool errors that the LLM agent can recover from."""


class ConflictError(WriterToolError):
    """Raised when write_unit is called but an active row already exists."""


class MissingActiveError(WriterToolError):
    """Raised when update_unit / invalidate_unit finds no active row."""


@dataclass(frozen=True)
class IngestDecision:
    """Outcome of routing an AtomicFact through the writer pipeline.

    The shape is tuned to be readable inside an LLM trace: the
    decision enum value is the discriminator, and exactly one of
    state / candidate_id is populated.
    """

    decision: str  # "admitted" | "deferred_vague" | "duplicate"
    state: EntityStateRecord | None = None
    candidate_id: str | None = None


# ---------------------------------------------------------------------------
# Writer tool facade
# ---------------------------------------------------------------------------


class MemoryWriterTools:
    """Bundle of four writer primitives over a fixed namespace.

    Each MemoryWriterTools instance is bound to a single namespace so
    that the agent loop cannot accidentally write to the wrong tenant by
    forgetting to thread a context argument. Cross-namespace agents
    instantiate one tool bundle per target namespace.
    """

    def __init__(
        self,
        view: EntityStateView,
        inbox: CandidateInbox,
        namespace: str,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must be non-empty")
        self._view = view
        self._inbox = inbox
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        return self._namespace

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_entity_state(
        self,
        entity: str,
        attribute: str | None = None,
        as_of: float | None = None,
    ) -> list[EntityStateRecord]:
        """Inspect the current (or historical) state of an entity.

        as_of=None returns the active rows; passing an epoch second
        instead returns the rows that were active at that instant. The
        agent uses this to decide whether write_unit or
        update_unit is the correct follow-up call.
        """
        if as_of is None:
            return self._view.get_active(self._namespace, entity, attribute)
        return self._view.get_as_of(self._namespace, entity, as_of, attribute)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_unit(self, fact: AtomicFact) -> IngestDecision:
        """Record a fresh fact for which no active row exists yet.

        Vague facts are intercepted and routed to the candidate inbox so
        the main store keeps its high-confidence invariant. If an active
        row already exists for the triple a ConflictError is raised
        and the agent is expected to retry via update_unit.
        """
        if fact.certainty is Certainty.VAGUE:
            return self._defer_vague(fact)

        existing = self._view.get_active(self._namespace, fact.subject, fact.predicate)
        if existing:
            raise ConflictError(
                f"active row already exists for "
                f"({fact.subject!r}, {fact.predicate!r}); use update_unit"
            )

        record = self._view.upsert(
            namespace=self._namespace,
            entity=fact.subject,
            attribute=fact.predicate,
            value=fact.object,
            certainty=fact.certainty,
            valid_from=fact.valid_from,
            source_unit_id=fact.source_anchor,
            qualifiers=fact.qualifiers,
        )
        return IngestDecision(decision="admitted", state=record)

    def update_unit(self, fact: AtomicFact) -> IngestDecision:
        """Supersede the active row for a triple with a newer value.

        Same vague-routing rule as write_unit. Raises
        MissingActiveError when there is no row to supersede so the
        agent fails fast instead of silently double-writing.
        """
        if fact.certainty is Certainty.VAGUE:
            return self._defer_vague(fact)

        existing = self._view.get_active(self._namespace, fact.subject, fact.predicate)
        if not existing:
            raise MissingActiveError(
                f"no active row to update for "
                f"({fact.subject!r}, {fact.predicate!r}); use write_unit"
            )

        record = self._view.upsert(
            namespace=self._namespace,
            entity=fact.subject,
            attribute=fact.predicate,
            value=fact.object,
            certainty=fact.certainty,
            valid_from=fact.valid_from,
            source_unit_id=fact.source_anchor,
            qualifiers=fact.qualifiers,
        )
        return IngestDecision(decision="admitted", state=record)

    def invalidate_unit(
        self,
        entity: str,
        attribute: str,
        valid_to: float | None = None,
    ) -> bool:
        """Close the currently active row without inserting a successor.

        Returns True when a row was actually closed, False when
        no active row was present (mirrors the EntityStateView API
        so the agent can branch without exception handling).
        """
        return self._view.invalidate(self._namespace, entity, attribute, valid_to=valid_to)

    # ------------------------------------------------------------------
    # Convenience: end-to-end ingestion
    # ------------------------------------------------------------------

    def ingest_fact(self, fact: AtomicFact) -> IngestDecision:
        """Route a fact through the full writer pipeline.

        Decision tree:
        1. certainty == VAGUE → park in vague inbox, do not touch
           the main store.
        2. accumulate == True → the fact is one item in an open-ended set.
           If an active row exists for (subject, predicate), append the new
           object value (comma-separated, deduped) via update_unit rather
           than replacing. First occurrence uses write_unit as normal.
        3. Active row already present for a single-valued predicate →
           update_unit (write-time supersession).
        4. Otherwise → write_unit.

        This is the convenience entrypoint used by callers that do not
        need fine-grained control; an LLM agent loop would instead call
        write_unit / update_unit / invalidate_unit directly
        so the chain-of-thought stays auditable.
        """
        if fact.certainty is Certainty.VAGUE:
            return self._defer_vague(fact)

        existing = self._view.get_active(self._namespace, fact.subject, fact.predicate)

        if fact.accumulate and existing:
            current_items = [v.strip() for v in existing[0].value.split(",") if v.strip()]
            new_item = fact.object.strip()
            if new_item in current_items:
                return IngestDecision(decision="duplicate", state=existing[0])
            merged_value = ", ".join([*current_items, new_item])
            merged_fact = fact.model_copy(update={"object": merged_value, "valid_from": None})
            return self.update_unit(merged_fact)

        if existing:
            return self.update_unit(fact)
        return self.write_unit(fact)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _defer_vague(self, fact: AtomicFact) -> IngestDecision:
        candidate_id = self._inbox.add(self._namespace, fact)
        return IngestDecision(decision="deferred_vague", candidate_id=candidate_id)
