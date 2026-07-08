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

        conn = self._conn_manager.get_connection()
        with self._lock:
            queue_id = self._enqueue_extract_locked(turn, conn, now=now)
            conn.commit()
        return queue_id

    def _enqueue_extract_locked(self, turn: RawTurn, conn, *, now: float | None = None) -> str:
        """Insert the queue row without acquiring the lock or committing.

        Callers MUST already hold _lock and are responsible for the
        eventual commit/rollback. Does not re-check for an existing
        queue_id for turn_id; callers wanting dedup must check first
        (see enqueue_extract).
        """
        ts = time.time() if now is None else now
        queue_id = uuid.uuid4().hex[:16]
        conn.execute(
            "INSERT INTO extract_queue "
            "(queue_id, turn_id, namespace, session_id, state, "
            " attempts, enqueued_at) "
            "VALUES (?, ?, ?, ?, 'pending', 0, ?)",
            (queue_id, turn.turn_id, turn.namespace, turn.session_id, ts),
        )
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

    def append_raw_turn_and_enqueue(
        self, turn: RawTurn, *, now: float | None = None
    ) -> tuple[RawTurn, str]:
        """Persist the L0 row and enqueue L1 extraction in one transaction.

        Replaces two independent commits (append_raw_turn + enqueue_extract)
        with a single one on the fast write path. Both underlying operations
        share the same _lock, so this is safe to call without any extra
        synchronization.
        """
        conn = self._conn_manager.get_connection()
        with self._lock:
            persisted = self._raw_turn_log._append_raw_turn_locked(turn, conn)
            queue_id = self._enqueue_extract_locked(persisted, conn, now=now)
            conn.commit()
        return persisted, queue_id

    def mark_extract_done(self, queue_id: str) -> None:
        """Mark a claimed job as completed."""
        with self._lock:
            self._conn_manager.get_connection().execute(
                "UPDATE extract_queue SET state = 'done', last_error = NULL WHERE queue_id = ?",
                (queue_id,),
            )
            self._conn_manager.get_connection().commit()

    def mark_extract_done_batch(self, queue_ids: list[str]) -> None:
        """Mark multiple claimed jobs as completed in a single commit.

        Collapses what would otherwise be N independent UPDATE+commit
        round trips (one per turn in a claimed batch) into one.
        """
        if not queue_ids:
            return
        conn = self._conn_manager.get_connection()
        with self._lock:
            conn.executemany(
                "UPDATE extract_queue SET state = 'done', last_error = NULL WHERE queue_id = ?",
                [(qid,) for qid in queue_ids],
            )
            conn.commit()

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

    def mark_extract_failed_batch(
        self,
        items: list[tuple[str, str]],
        *,
        retry: bool = True,
        max_attempts: int = 5,
    ) -> None:
        """Mark multiple claimed jobs as failed in a single commit.

        items is a list of (queue_id, error) pairs. Reads current attempts
        for all queue_ids in one query, computes new state per row in
        Python, then applies all updates in one executemany + commit.
        """
        if not items:
            return
        queue_ids = [qid for qid, _err in items]
        conn = self._conn_manager.get_connection()
        with self._lock:
            placeholders = ",".join("?" for _ in queue_ids)
            rows = conn.execute(
                f"SELECT queue_id, attempts FROM extract_queue WHERE queue_id IN ({placeholders})",
                queue_ids,
            ).fetchall()
            attempts_by_id = {str(r["queue_id"]): int(r["attempts"]) for r in rows}

            updates: list[tuple[str, str, str]] = []
            for queue_id, error in items:
                attempts = attempts_by_id.get(queue_id)
                if attempts is None:
                    continue
                new_state = "pending" if retry and attempts < max_attempts else "failed"
                updates.append((new_state, error[:1000], queue_id))

            if updates:
                conn.executemany(
                    "UPDATE extract_queue SET state = ?, last_error = ?, "
                    "claimed_at = NULL WHERE queue_id = ?",
                    updates,
                )
                conn.commit()

    def extract_queue_stats(self) -> dict[str, int]:
        """Aggregate counts per state."""
        rows = (
            self._conn_manager.get_connection()
            .execute("SELECT state, COUNT(*) AS n FROM extract_queue GROUP BY state")
            .fetchall()
        )
        return {str(r["state"]): int(r["n"]) for r in rows}
