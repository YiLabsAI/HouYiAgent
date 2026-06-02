"""SQLite entity state view implementation."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from typing import Any

from houyi.adapters.memory.backends.base import EntityStateView
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
        """Insert a new active state, closing any prior active row first (unless accumulate is True).

        Semantics:
        - If accumulate is False, and a row with valid_to IS NULL already exists for the
          (namespace, entity, attribute) triple, its valid_to is
          set to the new valid_from (closed-open interval contract).
        - The new row is inserted with valid_to = NULL.
        - valid_from must be >= any existing active row's
          valid_from; otherwise ValueError is raised because the
          materialized view has no defined order for backdated edits.
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

                prev_record, next_record = self._find_prev_next(existing, ts)

                calculated_valid_to = None
                if next_record is not None:
                    calculated_valid_to = next_record["valid_from"]

                # Shorten the preceding record if its interval extends past the new entry
                if (
                    not accumulate
                    and prev_record is not None
                    and (prev_record["valid_to"] is None or prev_record["valid_to"] > ts)
                ):
                    conn.execute(
                        "UPDATE entity_state SET valid_to=? WHERE state_id=?",
                        (ts, prev_record["state_id"]),
                    )
                    # Propagate valid_to invalidation to memories table
                    conn.execute(
                        "UPDATE memories SET valid_to=? WHERE scope=? AND key LIKE ? AND valid_to IS NULL",
                        (ts, namespace, f"{entity}.{attribute}.%"),
                    )

                record = EntityStateRecord(
                    namespace=namespace,
                    entity=entity,
                    attribute=attribute,
                    value=value,
                    certainty=certainty,
                    valid_from=ts,
                    valid_to=calculated_valid_to,
                    source_unit_id=source_unit_id,
                    qualifiers=qualifiers,
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
        conn.commit()
        return cur.rowcount > 0

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
