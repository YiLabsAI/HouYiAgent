"""Failure-query mining for the reflection pipeline.

The reflector (dreamer_reflect) is failure-anchored: it reflects only on
queries that recall failed to answer. This module mines those failing-query
texts from the RECALL_FAILURE events the recall pipeline emits, so the
reflector has something to reflect on without a golden answer set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from houyi.application.evolution.event_log import EvolutionEventLog


def mine_failure_queries(
    event_log: EvolutionEventLog,
    *,
    max_queries: int = 64,
) -> list[str]:
    """Extract replayable failing-query texts from RECALL_FAILURE events.

    Reads the full event log, keeps RECALL_FAILURE events that carry a
    query_preview payload, and returns de-duplicated query texts in first-
    seen order, capped at max_queries. Returns an empty list when no
    replayable signal exists, which lets callers skip reflection entirely
    instead of fabricating queries.
    """
    from houyi.application.evolution.events import EvolutionEventType

    events, _ = event_log.read_since(0)
    seen: set[str] = set()
    queries: list[str] = []
    for event in events:
        if event.event_type is not EvolutionEventType.RECALL_FAILURE:
            continue
        preview = event.payload.get("query_preview")
        if not isinstance(preview, str):
            continue
        text = preview.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append(text)
        if len(queries) >= max_queries:
            break
    return queries


__all__ = ["mine_failure_queries"]
