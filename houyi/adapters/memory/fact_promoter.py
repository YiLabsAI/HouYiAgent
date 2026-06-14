"""Project accepted L1 atomic facts into the L2 MemoryRecord store.

When the L1 extractor writes an AtomicFact to the entity-state view,
the recall fast path can already find it through entity_state. To
make the same fact retrievable through the vector path, a row is also
written into the main memories table with embedding_pending=1 so the
embedding-backfill worker can fill it later.

This projection is exposed as the FactPromoter protocol so callers
can plug in any policy, or disable it by passing promoter=None to
the worker. The default MemoryRecordPromoter writes one MemoryRecord
per accepted fact with embedding=None; the SQLite backend then sets
embedding_pending=1 on the row automatically.
"""

from __future__ import annotations

import logging
from typing import Protocol

from houyi.adapters.memory.types import (
    AtomicFact,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RawTurn,
)

logger = logging.getLogger(__name__)


class FactPromoter(Protocol):
    """Hook the L1 worker calls for each successfully projected fact.

    The protocol is sync; the worker wraps the call in
    asyncio.to_thread when it needs to keep the event loop responsive,
    so implementations do not need to be async def.
    """

    # Protocol body has no real implementation, so coverage cannot
    # observe it; the pragma silences a false-positive miss.
    def promote(self, turn: RawTurn, fact: AtomicFact) -> None:  # pragma: no cover
        ...


class _RecordSink(Protocol):
    """Minimal slice of the memory backend the default promoter needs."""

    def put(self, record: MemoryRecord) -> None: ...


class MemoryRecordPromoter:
    """Write one MemoryRecord per accepted fact, with embedding deferred.

    Record shape is intentionally narrow:

    - key = f"{subject}.{predicate}" so repeat facts collapse onto the
      same recall row. Entity-state still owns the bi-temporal validity;
      the memories row is only a vector-search affordance.
    - content = str(object) — the embeddable text payload.
    - embedding = None — filled later by the backfill worker. The SQLite
      backend flips embedding_pending=1 on the new row automatically.
    - scope = USER and memory_type = FACT are the conservative defaults.
      Callers that need different scoping can subclass and override
      _make_record.
    """

    def __init__(
        self,
        backend: _RecordSink,
        *,
        scope: MemoryScope = MemoryScope.USER,
        memory_type: MemoryType = MemoryType.FACT,
        provider_label: str = "atomic_fact_extractor",
    ) -> None:
        if backend is None:
            raise ValueError("backend is required")
        self._backend = backend
        self._scope = scope
        self._memory_type = memory_type
        self._provider_label = provider_label

    def promote(self, turn: RawTurn, fact: AtomicFact) -> None:
        """Persist the fact as a deferred-embedding MemoryRecord.

        The fact is already committed to entity_state by the time this
        runs, so the L2 row is an optimization rather than a correctness
        guarantee. Failures are logged and swallowed so the worker's
        retry loop is not affected.
        """
        try:
            record = self._make_record(turn, fact)
            self._backend.put(record)
        except Exception:
            logger.warning(
                "fact promotion failed for %s.%s",
                fact.subject,
                fact.predicate,
                exc_info=True,
            )

    def _make_record(self, turn: RawTurn, fact: AtomicFact) -> MemoryRecord:
        import hashlib

        anchor = fact.source_anchor or ""
        plain = f"{fact.subject}|{fact.predicate}|{fact.object}|{anchor}"
        digest = hashlib.sha256(plain.encode()).hexdigest()[:24]
        record_id = f"fact:{digest}"

        # Build self-contained semantic content so both retrievers and reasoning policies
        # have full access to subject, predicate, object, and temporal event_time.
        if fact.predicate == "_compound":
            content = str(fact.object)
        else:
            pred = (fact.predicate or "").replace("_", " ")
            content = f"{fact.subject} {pred} {fact.object}"
        date_val = fact.event_time
        if not date_val and fact.qualifiers:
            for k in ("date", "time", "when", "since", "occurred", "year"):
                if fact.qualifiers.get(k):
                    date_val = fact.qualifiers[k]
                    break
        if date_val:
            content += f" (time: {date_val})"

        return MemoryRecord(
            record_id=record_id,
            key=f"{fact.subject}.{fact.predicate}.{digest}",
            content=content,
            scope=self._scope,
            memory_type=self._memory_type,
            confidence=_certainty_to_confidence(fact),
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            provenance=MemoryProvenance(
                source_type="atomic_fact",
                source_ids=[fact.source_anchor] if fact.source_anchor else [],
                extracted_by=self._provider_label,
            ),
            embedding=None,
            metadata={
                "session_id": turn.session_id,
                "turn_id": turn.turn_id,
                "fact_subject": fact.subject,
                "fact_predicate": fact.predicate,
                "fact_object": fact.object,
            },
        )


def _certainty_to_confidence(fact: AtomicFact) -> float:
    """Map the AtomicFact certainty enum onto a numeric confidence.

    Mirrors the heuristics used elsewhere in the memory layer so
    promoted records sort the same way under the lexical+semantic
    blend in MemoryRetriever.retrieve.
    """
    from houyi.adapters.memory.types import Certainty

    return {
        Certainty.CERTAIN: 0.9,
        Certainty.PROBABLE: 0.6,
        Certainty.VAGUE: 0.3,
    }.get(fact.certainty, 0.5)


__all__ = ["FactPromoter", "MemoryRecordPromoter"]
