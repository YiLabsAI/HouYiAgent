"""SQLite raw turn log operations (L0)."""

from __future__ import annotations

import json
import sqlite3
import threading

from houyi.adapters.memory.types import RawTurn


def _row_to_raw_turn(row: sqlite3.Row) -> RawTurn:
    """Inflate a raw_turn_log row into a RawTurn."""
    metadata_raw = row["metadata"]
    metadata: dict[str, str] = {}
    if metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
            if isinstance(parsed, dict):
                metadata = {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
    return RawTurn(
        turn_id=row["turn_id"],
        namespace=row["namespace"],
        session_id=row["session_id"],
        turn_index=int(row["turn_index"]),
        role=row["role"],
        content=row["content"],
        metadata=metadata,
        created_at=float(row["created_at"]),
    )


class SQLiteRawTurnLog:
    """Handles raw turn log operations."""

    def __init__(self, conn_manager, lock: threading.Lock):
        self._conn_manager = conn_manager
        self._lock = lock

    def append_raw_turn(self, turn: RawTurn) -> RawTurn:
        """Persist a conversation turn to the L0 log.

        Assigns the next monotonic turn_index within (namespace, session_id)
        automatically and returns a copy of turn with the assigned index.
        """
        conn = self._conn_manager.get_connection()
        with self._lock:
            if turn.turn_index > 0:
                resolved_index = turn.turn_index
            else:
                row = conn.execute(
                    "SELECT COALESCE(MAX(turn_index), -1) AS last "
                    "FROM raw_turn_log WHERE namespace = ? AND session_id = ?",
                    (turn.namespace, turn.session_id),
                ).fetchone()
                resolved_index = int(row["last"]) + 1
            metadata_json = json.dumps(turn.metadata, ensure_ascii=False) if turn.metadata else None
            conn.execute(
                "INSERT INTO raw_turn_log "
                "(turn_id, namespace, session_id, turn_index, role, content, "
                " metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    turn.turn_id,
                    turn.namespace,
                    turn.session_id,
                    resolved_index,
                    turn.role,
                    turn.content,
                    metadata_json,
                    turn.created_at,
                ),
            )
            conn.commit()
            return turn.model_copy(update={"turn_index": resolved_index})

    def list_raw_turns(
        self,
        namespace: str,
        session_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[RawTurn]:
        """Return turns in chronological order (oldest first)."""
        rows = (
            self._conn_manager.get_connection()
            .execute(
                "SELECT * FROM raw_turn_log "
                "WHERE namespace = ? AND session_id = ? "
                "ORDER BY turn_index ASC "
                "LIMIT ? OFFSET ?",
                (namespace, session_id, limit, offset),
            )
            .fetchall()
        )
        return [_row_to_raw_turn(r) for r in rows]

    def count_raw_turns(self, namespace: str, session_id: str) -> int:
        """Return the number of turns logged for a session."""
        row = (
            self._conn_manager.get_connection()
            .execute(
                "SELECT COUNT(*) AS n FROM raw_turn_log WHERE namespace = ? AND session_id = ?",
                (namespace, session_id),
            )
            .fetchone()
        )
        return int(row["n"])

    def get_raw_turn(self, turn_id: str) -> RawTurn | None:
        """Look up a single turn by id. Returns None when missing."""
        row = (
            self._conn_manager.get_connection()
            .execute(
                "SELECT * FROM raw_turn_log WHERE turn_id = ?",
                (turn_id,),
            )
            .fetchone()
        )
        return _row_to_raw_turn(row) if row is not None else None
