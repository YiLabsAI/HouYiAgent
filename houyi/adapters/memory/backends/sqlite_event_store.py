"""SQLite-backed store for first-class MemoryEvent records."""

from __future__ import annotations

import sqlite3

from houyi.adapters.memory.backends.base import EventView
from houyi.adapters.memory.types import Certainty, MemoryEvent


class SQLiteEventStore(EventView):
    """Persistent store for MemoryEvent rows in the events table.

    Implements the EventView ABC. Connection management delegates to
    the parent SQLiteMemoryBackend via _conn(), which returns the
    same thread-local connection used by backend.transaction() --
    so writes inside a transaction automatically participate in it.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def _conn(self) -> sqlite3.Connection:
        return self._backend._conn()

    def add_event(self, event: MemoryEvent) -> MemoryEvent:
        """INSERT OR REPLACE an event row. Returns the stored event."""
        qualifiers_json = _json_dumps(event.qualifiers) if event.qualifiers else None
        self._conn().execute(
            """
            INSERT OR REPLACE INTO events
            (event_id, namespace, subject, action, object, timestamp,
             context, certainty, qualifiers, source_anchor,
             valid_from, valid_to, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.namespace,
                event.subject,
                event.action,
                event.object,
                event.timestamp,
                event.context,
                event.certainty.value,
                qualifiers_json,
                event.source_anchor,
                event.valid_from,
                event.valid_to,
                event.created_at,
            ),
        )
        return event

    def get_event(self, event_id: str) -> MemoryEvent | None:
        """Retrieve a single event by event_id."""
        row = (
            self._conn()
            .execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return _row_to_event(row)

    def get_events_by_subject(self, namespace: str, subject: str) -> list[MemoryEvent]:
        """Return all active events (valid_to IS NULL) for a given entity."""
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM events WHERE namespace = ? AND subject = ? AND valid_to IS NULL",
                (namespace, subject),
            )
            .fetchall()
        )
        return [_row_to_event(r) for r in rows]

    def get_events_by_subject_and_action(
        self,
        namespace: str,
        subject: str,
        action: str,
    ) -> list[MemoryEvent]:
        """Return active events matching both subject and action."""
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM events WHERE namespace = ? AND subject = ? AND action = ? AND valid_to IS NULL",
                (namespace, subject, action),
            )
            .fetchall()
        )
        return [_row_to_event(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import json  # noqa: E402
from typing import Any  # noqa: E402


def _json_dumps(obj: dict[str, str]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
    d = dict(row)
    qualifiers = None
    if d.get("qualifiers"):
        with __import__("contextlib").suppress(Exception):
            qualifiers = json.loads(d["qualifiers"])
    return MemoryEvent(
        event_id=d["event_id"],
        namespace=d["namespace"],
        subject=d["subject"],
        action=d["action"],
        object=d["object"],
        timestamp=d["timestamp"],
        context=d.get("context", ""),
        certainty=Certainty(d.get("certainty", "certain")),
        qualifiers=qualifiers,
        source_anchor=d.get("source_anchor", ""),
        valid_from=d["valid_from"],
        valid_to=d.get("valid_to"),
        created_at=d["created_at"],
    )
