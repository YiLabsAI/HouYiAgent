"""Event retriever for temporal occurrence lookup.

Queries the events table by subject (and optionally action) to surface
first-class MemoryEvent records as RecallCandidates. Each event is
converted to an AtomicFact where:
- subject = event.subject
- predicate = event.action
- object = event.object with timestamp appended parenthetically
- event_time = event.timestamp (so the answerer can render (time: ...))

This retriever is wired into TEMPORAL_QUERY and FACTUAL_LOOKUP routes.
"""

from __future__ import annotations

import asyncio

from houyi.adapters.memory.backends.base import EventView
from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.retrievers.entity_state import _infer_entity_attribute
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, Certainty, MemoryEvent

# Certainty-based base scores -- CERTAIN events deserve a higher anchor
# than PROBABLE, which in turn outranks VAGUE.
_CERTAINTY_SCORE: dict[Certainty, float] = {
    Certainty.CERTAIN: 7.0,
    Certainty.PROBABLE: 5.0,
    Certainty.VAGUE: 3.0,
}


class EventRetriever(Retriever):
    """Direct lookup on the events table for entity-action queries."""

    def __init__(self, event_view: EventView) -> None:
        if event_view is None:
            raise ValueError("event_view is required")
        self._view = event_view

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        hint = _infer_entity_attribute(query)
        if hint is None or not hint.entity:
            return []

        entity = hint.entity.strip()
        namespace = query.namespace

        # If an attribute hint looks like an action verb, query by subject+action.
        # Otherwise query by subject only and let downstream rank by relevance.
        query_action = None
        if hint.attribute and hint.attribute.strip():
            query_action = hint.attribute.strip()

        if query_action:
            events = await asyncio.to_thread(
                self._view.get_events_by_subject_and_action,
                namespace,
                entity,
                query_action,
            )
        else:
            events = await asyncio.to_thread(
                self._view.get_events_by_subject,
                namespace,
                entity,
            )

        return [_candidate_from_event(e, self.name, query_action=query_action) for e in events]


def _candidate_from_event(
    event: MemoryEvent,
    retriever_name: str,
    query_action: str | None = None,
) -> RecallCandidate:
    """Convert a MemoryEvent into a RecallCandidate with event_time set.

    Scoring policy:
    - Base score from certainty (CERTAIN=7, PROBABLE=5, VAGUE=3).
    - +2.0 boost when the query attribute exactly matches the stored action
      verb, indicating a precise subject+action hit rather than a broad
      subject-only sweep.
    """
    fact = AtomicFact(
        subject=event.subject,
        predicate=event.action,
        object=f"{event.object} ({event.timestamp})",
        certainty=event.certainty,
        source_anchor=event.source_anchor,
        qualifiers=event.qualifiers,
        event_time=event.timestamp,
    )
    score = _CERTAINTY_SCORE.get(event.certainty, 5.0)
    # Boost for precise action verb match: the query explicitly asked for
    # this action, and the event's action matches exactly.
    if query_action and event.action.strip().lower() == query_action.lower():
        score += 2.0
    return RecallCandidate(
        fact=fact,
        score=score,
        matched_by=RetrieverKind.EVENT,
        retriever_name=retriever_name,
        signals={
            "event_id": event.event_id,
            "action": event.action,
            "timestamp": event.timestamp,
        },
        explanation=f"event: {event.subject} {event.action} {event.object}",
    )


__all__ = ["EventRetriever"]
