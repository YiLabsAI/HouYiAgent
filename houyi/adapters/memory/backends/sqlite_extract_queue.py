"""SQLite extract queue operations (L0 → L1 hand-off)."""

from __future__ import annotations

import threading
import time
import uuid

from houyi.adapters.memory.types import RawTurn


class SQLiteExtractQueue:
    """Handles extract queue operations."""

    def __init__(self, conn_manager, raw_turn_log, lock: threading.Lock):
        self._conn_manager = conn_manager
        self._raw_turn_log = raw_turn_log
        self._lock = lock

    def enqueue_extract(self, turn: RawTurn, *, now: float | None = None) -> str:
        """Schedule async extraction for a previously logged turn."""
        ts = time.time() if now is None else now
        existing = (
            self._conn_manager.get_connection()
            .execute(
                "SELECT queue_id FROM extract_queue WHERE turn_id = ?",
                (turn.turn_id,),
            )
            .fetchone()
        )
        if existing is not None:
            return str(existing["queue_id"])

        queue_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._conn_manager.get_connection().execute(
                "INSERT INTO extract_queue "
                "(queue_id, turn_id, namespace, session_id, state, "
                " attempts, enqueued_at) "
                "VALUES (?, ?, ?, ?, 'pending', 0, ?)",
                (queue_id, turn.turn_id, turn.namespace, turn.session_id, ts),
            )
            self._conn_manager.get_connection().commit()
        return queue_id

    def claim_extract_jobs(
        self,
        *,
        limit: int = 8,
        namespace: str | None = None,
        lease_seconds: float = 60.0,
        now: float | None = None,
    ) -> list[tuple[str, RawTurn]]:
        """Atomically claim up to limit pending jobs."""
        ts = time.time() if now is None else now
        stale_before = ts - lease_seconds

        conn = self._conn_manager.get_connection()
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                params: list = ["pending", "in_progress", stale_before]
                where = "(state = ? OR (state = ? AND claimed_at IS NOT NULL AND claimed_at < ?))"
                if namespace is not None:
                    where += " AND namespace = ?"
                    params.append(namespace)
                params.append(limit)

                rows = conn.execute(
                    f"SELECT * FROM extract_queue WHERE {where} ORDER BY enqueued_at ASC LIMIT ?",
                    params,
                ).fetchall()
                if not rows:
                    conn.commit()
                    return []

                claimed: list[tuple[str, RawTurn]] = []
                for row in rows:
                    conn.execute(
                        "UPDATE extract_queue SET state = 'in_progress', "
                        "claimed_at = ?, attempts = attempts + 1 "
                        "WHERE queue_id = ?",
                        (ts, row["queue_id"]),
                    )
                    turn = self._raw_turn_log.get_raw_turn(row["turn_id"])
                    if turn is None:
                        conn.execute(
                            "UPDATE extract_queue SET state = 'failed', "
                            "last_error = 'orphan_turn_missing' "
                            "WHERE queue_id = ?",
                            (row["queue_id"],),
                        )
                        continue
                    claimed.append((str(row["queue_id"]), turn))
                conn.commit()
                return claimed
            except Exception:
                conn.rollback()
                raise

    def mark_extract_done(self, queue_id: str) -> None:
        """Mark a claimed job as completed."""
        with self._lock:
            self._conn_manager.get_connection().execute(
                "UPDATE extract_queue SET state = 'done', last_error = NULL WHERE queue_id = ?",
                (queue_id,),
            )
            self._conn_manager.get_connection().commit()

    def mark_extract_failed(
        self,
        queue_id: str,
        error: str,
        *,
        retry: bool = True,
        max_attempts: int = 5,
    ) -> None:
        """Mark a claimed job as failed."""
        with self._lock:
            row = (
                self._conn_manager.get_connection()
                .execute(
                    "SELECT attempts FROM extract_queue WHERE queue_id = ?",
                    (queue_id,),
                )
                .fetchone()
            )
            if row is None:
                return
            attempts = int(row["attempts"])
            new_state = "pending" if retry and attempts < max_attempts else "failed"
            self._conn_manager.get_connection().execute(
                "UPDATE extract_queue SET state = ?, last_error = ?, "
                "claimed_at = NULL WHERE queue_id = ?",
                (new_state, error[:1000], queue_id),
            )
            self._conn_manager.get_connection().commit()

    def extract_queue_stats(self) -> dict[str, int]:
        """Aggregate counts per state."""
        rows = (
            self._conn_manager.get_connection()
            .execute("SELECT state, COUNT(*) AS n FROM extract_queue GROUP BY state")
            .fetchall()
        )
        return {str(r["state"]): int(r["n"]) for r in rows}
