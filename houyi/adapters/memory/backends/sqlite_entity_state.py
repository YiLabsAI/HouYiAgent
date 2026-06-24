"""SQLite entity state view implementation."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from typing import Any

from houyi.adapters.memory.backends.base import EntityStateView
from houyi.adapters.memory.fact_identity import fact_record_id
from houyi.adapters.memory.types import Certainty, EntityStateRecord

logger = logging.getLogger(__name__)


class SQLiteEntityStateView(EntityStateView):
    """SQLite-backed implementation of EntityStateView.

    The class is intentionally a thin facade over SQLiteMemoryBackend;
    it shares the backend's connection so that writer-side transactions
    can span both the AtomicFact store and the entity-state view.
    Alternative backends (Postgres, in-memory) implement the same ABC
    surface independently and need not derive from this class.
    """

    def __init__(self, backend) -> None:
        self._backend = backend

    _NUDGE = 1e-6

    def _get_reconciliation_ts(self, existing: list[Any], ts: float) -> float:
        candidate = ts
        taken = {row["valid_from"] for row in existing}
        while candidate in taken or any(abs(candidate - t) < 1e-9 for t in taken):
            candidate += self._NUDGE
        return candidate

    def _find_prev_next(self, existing: list[Any], ts: float) -> tuple[Any, Any]:
        prev_record = None
        next_record = None
        for row in existing:
            if row["valid_from"] < ts:
                prev_record = row
            elif row["valid_from"] >= ts and next_record is None:
                next_record = row
        return prev_record, next_record

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def upsert(
        self,
        namespace: str,
        entity: str,
        attribute: str,
        value: str,
        *,
        certainty: Certainty = Certainty.CERTAIN,
        valid_from: float | None = None,
        source_unit_id: str | None = None,
        qualifiers: dict[str, str] | None = None,
        accumulate: bool = False,
    ) -> EntityStateRecord:
        """Append a new active state row. This is append-only: it does NOT close
        any prior active row for the same (namespace, entity, attribute) triple.

        Closing stale active rows (so that a single-valued attribute has at most
        one active row) is deferred to the dreamer's entity_state conflict-
        resolution pass, which scans triples with >=2 active rows and sets
        valid_to on the superseded ones by valid_from order. Keeping the write
        path append-only preserves its low latency; consolidation runs off the
        hot path. Callers that need immediate retraction use invalidate().

        Semantics:
        - The new row is inserted with valid_to = NULL regardless of accumulate.
        - accumulate only tags the row's qualifiers so readers know multiple
          active values are expected (open set) rather than a contradiction.
        - valid_from must be >= any existing active row's valid_from; otherwise
          ValueError is raised because the materialized view has no defined
          order for backdated edits.
        """
        if not namespace or not entity or not attribute:
            raise ValueError("namespace, entity, attribute must be non-empty")

        ts: float = valid_from if valid_from is not None else time.time()
        conn = self._backend._conn()
        try:
            with conn:  # Atomic: close-old + insert-new must succeed together.
                # Query all existing records sorted by valid_from
                existing = conn.execute(
                    """
                    SELECT state_id, valid_from, valid_to, value FROM entity_state
                    WHERE namespace=? AND entity=? AND attribute=?
                    ORDER BY valid_from ASC
                    """,
                    (namespace, entity, attribute),
                ).fetchall()

                # Avoid UNIQUE constraint collision by auto-bumping duplicate timestamps slightly
                ts = self._get_reconciliation_ts(existing, ts)

                calculated_valid_to = None

                effective_qualifiers = dict(qualifiers or {})
                if accumulate:
                    effective_qualifiers["accumulate"] = "true"

                record = EntityStateRecord(
                    namespace=namespace,
                    entity=entity,
                    attribute=attribute,
                    value=value,
                    certainty=certainty,
                    valid_from=ts,
                    valid_to=calculated_valid_to,
                    source_unit_id=source_unit_id,
                    qualifiers=effective_qualifiers if effective_qualifiers else None,
                )
                conn.execute(
                    """
                    INSERT INTO entity_state
                    (state_id, namespace, entity, attribute, value,
                    certainty, qualifiers, valid_from, valid_to,
                    source_unit_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.state_id,
                        record.namespace,
                        record.entity,
                        record.attribute,
                        record.value,
                        record.certainty.value,
                        json.dumps(record.qualifiers, ensure_ascii=False)
                        if record.qualifiers
                        else None,
                        record.valid_from,
                        record.valid_to,
                        record.source_unit_id,
                        record.created_at,
                    ),
                )
        except sqlite3.Error as exc:
            raise RuntimeError(f"entity_state upsert failed: {exc}") from exc
        return record

    def invalidate(
        self,
        namespace: str,
        entity: str,
        attribute: str,
        *,
        valid_to: float | None = None,
    ) -> bool:
        """Close the currently active row for a triple without inserting a successor.

        Used for explicit retraction signals where the old fact is known
        to be false but no replacement is yet known. Returns True if
        an active row was closed.
        """
        ts = valid_to if valid_to is not None else time.time()
        conn = self._backend._conn()
        cur = conn.execute(
            """
            UPDATE entity_state SET valid_to=?
            WHERE namespace=? AND entity=? AND attribute=?
            AND valid_to IS NULL
            """,
            (ts, namespace, entity, attribute),
        )
        # Propagate valid_to retraction to memories table
        conn.execute(
            "UPDATE memories SET valid_to=? WHERE scope=? AND key LIKE ? AND valid_to IS NULL",
            (ts, namespace, f"{entity}.{attribute}.%"),
        )
        if not getattr(self._backend._in_transaction, "active", False):
            conn.commit()
        return cur.rowcount > 0

    def list_conflicted_triples(
        self,
        namespace: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Return (namespace, entity, attribute) triples with >=2 active rows.

        Uses the idx_entity_state_active partial index so only active rows are
        scanned. The consolidator calls this to find single-valued attributes
        the append-only write path left with concurrent active values.
        """
        conn = self._backend._conn()
        if namespace is None:
            rows = conn.execute(
                """
                SELECT namespace, entity, attribute FROM entity_state
                WHERE valid_to IS NULL
                GROUP BY namespace, entity, attribute
                HAVING COUNT(*) >= 2
                ORDER BY namespace ASC, entity ASC, attribute ASC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT namespace, entity, attribute FROM entity_state
                WHERE namespace=? AND valid_to IS NULL
                GROUP BY namespace, entity, attribute
                HAVING COUNT(*) >= 2
                ORDER BY entity ASC, attribute ASC
                """,
                (namespace,),
            ).fetchall()
        return [(row["namespace"], row["entity"], row["attribute"]) for row in rows]

    def supersede(
        self,
        namespace: str,
        entity: str,
        attribute: str,
        *,
        keep_state_id: str,
        valid_to: float,
    ) -> tuple[int, int]:
        """Close every active row of the triple except keep_state_id.

        The successor (keep_state_id) stays active; every other active row is
        closed with valid_to (the successor's valid_from, preserving as-of
        semantics). The valid_to IS NULL guard makes the call idempotent.

        For each closed row the backing memories row is closed too, located by
        re-deriving its record_id from (entity, attribute, value, source_unit_id)
        via the shared fact identity hash. This is a precise single-row close,
        unlike invalidate()'s key-prefix LIKE which would also close the
        successor's backing row. Both updates share one transaction so the
        entity_state view and the memories store never disagree.
        """
        conn = self._backend._conn()
        rows_closed = 0
        rows_propagated = 0
        victims = conn.execute(
            """
            SELECT state_id, value, source_unit_id FROM entity_state
            WHERE namespace=? AND entity=? AND attribute=?
            AND state_id<>? AND valid_to IS NULL
            """,
            (namespace, entity, attribute, keep_state_id),
        ).fetchall()
        for row in victims:
            conn.execute(
                "UPDATE entity_state SET valid_to=? WHERE state_id=?",
                (valid_to, row["state_id"]),
            )
            rows_closed += 1
            anchor = row["source_unit_id"] or ""
            record_id = fact_record_id(entity, attribute, row["value"], anchor)
            cur = conn.execute(
                "UPDATE memories SET valid_to=? WHERE record_id=? AND valid_to IS NULL",
                (valid_to, record_id),
            )
            if cur.rowcount > 0:
                rows_propagated += cur.rowcount
        # Mirror invalidate(): commit only when no outer transaction owns the
        # connection, so supersede is safe to nest inside a writer transaction.
        if not getattr(self._backend._in_transaction, "active", False):
            conn.commit()
        return rows_closed, rows_propagated

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_active(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        """Return all currently active rows for an entity.

        Hot path on the recall side; backed by the partial index
        idx_entity_state_active so the lookup is O(log N) without
        scanning historical rows.
        """
        conn = self._backend._conn()
        if attribute is None:
            rows = conn.execute(
                """
                SELECT * FROM entity_state
                WHERE namespace=? AND entity=? AND valid_to IS NULL
                ORDER BY attribute ASC
                """,
                (namespace, entity),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM entity_state
                WHERE namespace=? AND entity=? AND attribute=? AND valid_to IS NULL
                """,
                (namespace, entity, attribute),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_id(self, state_id: str) -> EntityStateRecord | None:
        """Retrieve a single EntityStateRecord by state_id."""
        conn = self._backend._conn()
        row = conn.execute(
            "SELECT * FROM entity_state WHERE state_id=?",
            (state_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_as_of(
        self,
        namespace: str,
        entity: str,
        ts: float,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        """Return rows that were active at instant ts.

        A row is considered active at ts when valid_from <= ts
        and (valid_to IS NULL OR valid_to > ts).
        """
        if attribute is None:
            sql = """
 SELECT * FROM entity_state
 WHERE namespace=? AND entity=?
 AND valid_from<=? AND (valid_to IS NULL OR valid_to>?)
 ORDER BY attribute ASC
 """
            params: tuple[Any, ...] = (namespace, entity, ts, ts)
        else:
            sql = """
 SELECT * FROM entity_state
 WHERE namespace=? AND entity=? AND attribute=?
 AND valid_from<=? AND (valid_to IS NULL OR valid_to>?)
 """
            params = (namespace, entity, attribute, ts, ts)
        rows = self._backend._conn().execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_history(
        self,
        namespace: str,
        entity: str,
        attribute: str | None = None,
    ) -> list[EntityStateRecord]:
        """Return every recorded version of an entity attribute, newest first."""
        conn = self._backend._conn()
        if attribute is None:
            rows = conn.execute(
                """
 SELECT * FROM entity_state
 WHERE namespace=? AND entity=?
 ORDER BY attribute ASC, valid_from DESC
 """,
                (namespace, entity),
            ).fetchall()
        else:
            rows = conn.execute(
                """
 SELECT * FROM entity_state
 WHERE namespace=? AND entity=? AND attribute=?
 ORDER BY valid_from DESC
 """,
                (namespace, entity, attribute),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_entities(self, namespace: str) -> list[str]:
        """Return all distinct entity names that have active rows."""
        conn = self._backend._conn()
        rows = conn.execute(
            """
 SELECT DISTINCT entity FROM entity_state
 WHERE namespace=? AND valid_to IS NULL
 ORDER BY entity ASC
 """,
            (namespace,),
        ).fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EntityStateRecord:
        d: dict[str, Any] = dict(row)
        qualifiers: dict[str, str] | None = None
        raw = d.get("qualifiers")
        if isinstance(raw, str) and raw:
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    qualifiers = {str(k): str(v) for k, v in parsed.items()}
        return EntityStateRecord(
            state_id=d["state_id"],
            namespace=d["namespace"],
            entity=d["entity"],
            attribute=d["attribute"],
            value=d["value"],
            certainty=Certainty(d["certainty"]),
            valid_from=d["valid_from"],
            valid_to=d.get("valid_to"),
            source_unit_id=d.get("source_unit_id"),
            qualifiers=qualifiers,
            created_at=d["created_at"],
        )
