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

# Words that look like capitalized tokens but are not entity names.
_SKIP_WORDS: frozenset[str] = frozenset(
    {
        "when",
        "where",
        "what",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "which",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "did",
        "do",
        "does",
        "has",
        "have",
        "will",
        "would",
        "can",
        "could",
        "may",
        "might",
        "shall",
        "should",
        "must",
        "i",
        "he",
        "she",
        "it",
        "we",
        "they",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)


def _extract_all_entities(query_text: str, primary: str) -> list[str]:
    """Extract all capitalized proper nouns from query text.

    Returns a deduplicated list starting with the primary entity,
    followed by any additional entities found in the query. This
    mirrors the multi-entity scan in EntityStateRetriever so that
    queries mentioning multiple people (e.g. "John and James") query
    events for ALL of them, not just the first.
    """
    entities = [primary]
    seen = {primary.lower()}
    for word in query_text.split():
        clean = word.strip(".,!?:;'\"()[]{}")
        if not clean or not clean[0].isupper():
            continue
        low = clean.lower()
        if low in _SKIP_WORDS or low in seen:
            continue
        seen.add(low)
        entities.append(clean)
    return entities


class EventRetriever(Retriever):
    """Direct lookup on the events table for entity-action queries.

    Supports multi-entity queries: when the query mentions multiple
    people (e.g. "John and James"), events are queried for ALL of
    them, not just the first entity extracted by the regex.
    """

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

        primary_entity = hint.entity.strip()
        namespace = query.namespace

        # query_action is the hint.attribute, kept ONLY as a scoring signal for
        # _candidate_from_event (it boosts a candidate whose stored action
        # exactly matches the query verb). It is NOT used to filter the DB
        # query, because hint.attribute is a query property (noun phrase like
        # "places or events", "martial arts") while event actions are verbs
        # (plans_to_go, went, agreed_to) — they never match, so filtering by
        # the attribute yields zero events for almost every question.
        query_action = hint.attribute.strip() if hint.attribute else None

        # Multi-entity scan: extract all capitalized proper nouns from the
        # query text so multi-person queries (e.g. "John and James") retrieve
        # events for ALL mentioned entities.
        entities = _extract_all_entities(query.text, primary_entity)

        all_events: list[MemoryEvent] = []
        seen_event_ids: set[str] = set()
        for entity in entities:
            evts = await asyncio.to_thread(
                self._view.get_events_by_subject,
                namespace,
                entity,
            )
            for evt in evts:
                if evt.event_id not in seen_event_ids:
                    seen_event_ids.add(evt.event_id)
                    all_events.append(evt)
        return [_candidate_from_event(e, self.name, query_action=query_action) for e in all_events]


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
    quals = dict(event.qualifiers or {})
    if event.context:
        quals["context"] = event.context
    fact = AtomicFact(
        subject=event.subject,
        predicate=event.action,
        object=f"{event.object} ({event.timestamp})",
        certainty=event.certainty,
        source_anchor=event.source_anchor,
        qualifiers=quals or None,
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
