"""Timeline retriever for historical and as-of entity-state lookup.

The retriever uses the same materialized entity-state view as the
single-hop retriever, but reads historical rows instead of only the
currently active row. It supports two modes:

- query.as_of is set: return rows active at that instant.
- query.as_of is absent: return row history newest-first.

Entity and attribute inference is intentionally shared with
EntityStateRetriever so factual and temporal paths agree on how a
query target is derived.
"""

from __future__ import annotations

import asyncio

from houyi.adapters.memory.backends.base import EntityStateView
from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.retrievers.entity_state import (
    EntityAttributeHint,
    _extract_event_time,
    _infer_entity_attribute,
    _is_identity_anchor,
)
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, EntityStateRecord


class TimelineRetriever(Retriever):
    """Retrieve historical entity-state rows.

    The class does not parse natural-language dates. Callers that know
    the target instant should set RecallQuery.as_of; otherwise the
    retriever returns full history for the inferred entity/attribute so
    later fusion can rank or trim by recency.
    """

    def __init__(self, view: EntityStateView) -> None:
        if view is None:
            raise ValueError("view is required")
        self._view = view

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        hint = _infer_entity_attribute(query)
        if hint is None:
            return []

        if query.as_of is not None:
            rows = await asyncio.to_thread(
                self._view.get_as_of,
                query.namespace,
                hint.entity,
                query.as_of,
                hint.attribute,
            )
            return [
                _candidate_from_row(row, self.name, hint, mode="as_of")
                for row in rows
                if not _is_identity_anchor(row)
            ]

        rows = await asyncio.to_thread(
            self._view.get_history,
            query.namespace,
            hint.entity,
            hint.attribute,
        )
        return [
            _candidate_from_row(row, self.name, hint, mode="history")
            for row in rows
            if not _is_identity_anchor(row)
        ]


def _candidate_from_row(
    row: EntityStateRecord,
    retriever_name: str,
    hint: EntityAttributeHint,
    *,
    mode: str,
) -> RecallCandidate:
    """Convert a historical row into a timeline recall candidate."""
    fact = AtomicFact(
        subject=row.entity,
        predicate=row.attribute,
        object=row.value,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        certainty=row.certainty,
        source_anchor=row.source_unit_id or row.state_id,
        qualifiers=row.qualifiers,
        event_time=_extract_event_time(row.qualifiers),
    )
    score = _timeline_score(row, hint, mode=mode)
    return RecallCandidate(
        fact=fact,
        score=score,
        matched_by=RetrieverKind.TIMELINE,
        retriever_name=retriever_name,
        signals={
            "entity": row.entity,
            "attribute": row.attribute,
            "hint_source": hint.source,
            "mode": mode,
            "valid_from": row.valid_from,
            "valid_to": row.valid_to,
        },
        explanation=f"{mode} entity-state row for {row.entity}.{row.attribute}",
    )


def _timeline_score(
    row: EntityStateRecord,
    hint: EntityAttributeHint,
    *,
    mode: str,
) -> float:
    """Score exact attribute and as-of matches above broad history rows."""
    score = 0.6
    if mode == "as_of":
        score += 1.0
        if hint.attribute is not None and hint.attribute == row.attribute:
            score += 2.0
            if row.valid_to is None:
                score += 0.2
    return score


__all__ = ["TimelineRetriever"]
