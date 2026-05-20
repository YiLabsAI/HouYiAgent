from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from houyi.application.evolution.artifacts import EvolutionArtifact, EvolutionArtifactType
from houyi.application.evolution.audit_log import AuditEntry
from houyi.application.evolution.events import EvolutionEvent, EvolutionEventType


class SQLiteEvolutionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
 CREATE TABLE IF NOT EXISTS evolution_events (
 offset INTEGER PRIMARY KEY AUTOINCREMENT,
 event_id TEXT UNIQUE NOT NULL,
 event_type TEXT NOT NULL,
 target TEXT NOT NULL,
 namespace TEXT NOT NULL,
 timestamp REAL NOT NULL,
 payload_json TEXT NOT NULL,
 metrics_json TEXT NOT NULL
 )
 """
            )
            conn.execute(
                """
 CREATE TABLE IF NOT EXISTS evolution_cursors (
 consumer TEXT PRIMARY KEY,
 cursor INTEGER NOT NULL
 )
 """
            )
            conn.execute(
                """
 CREATE TABLE IF NOT EXISTS evolution_artifacts (
 artifact_id TEXT PRIMARY KEY,
 artifact_type TEXT NOT NULL,
 content TEXT NOT NULL,
 version INTEGER NOT NULL,
 metadata_json TEXT NOT NULL,
 parent_id TEXT,
 state TEXT NOT NULL,
 created_order INTEGER NOT NULL
 )
 """
            )
            conn.execute(
                """
 CREATE TABLE IF NOT EXISTS evolution_active_artifacts (
 artifact_type TEXT PRIMARY KEY,
 artifact_id TEXT NOT NULL
 )
 """
            )
            conn.execute(
                """
 CREATE TABLE IF NOT EXISTS evolution_audit (
 audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
 timestamp REAL NOT NULL,
 consumer TEXT NOT NULL,
 action TEXT NOT NULL,
 cursor_before INTEGER NOT NULL,
 cursor_after INTEGER NOT NULL,
 events_consumed INTEGER NOT NULL,
 skipped INTEGER NOT NULL,
 reason TEXT NOT NULL,
 promotion_level TEXT,
 error TEXT
 )
 """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_consumer "
                "ON evolution_audit (consumer, audit_id)"
            )
            conn.execute(
                """
 CREATE TABLE IF NOT EXISTS evolution_shadow_artifacts (
 artifact_type TEXT PRIMARY KEY,
 artifact_id TEXT NOT NULL
 )
 """
            )

    def append(self, event: EvolutionEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
 INSERT OR IGNORE INTO evolution_events (
 event_id, event_type, target, namespace, timestamp, payload_json, metrics_json
 ) VALUES (?, ?, ?, ?, ?, ?, ?)
 """,
                (
                    event.event_id,
                    event.event_type.value,
                    event.target,
                    event.namespace,
                    event.timestamp,
                    json.dumps(event.payload, sort_keys=True),
                    json.dumps(event.metrics, sort_keys=True),
                ),
            )

    def read_since(
        self,
        cursor: int,
        *,
        limit: int | None = None,
    ) -> tuple[list[EvolutionEvent], int]:
        if cursor < 0:
            raise ValueError("cursor must be >= 0")
        sql = """
 SELECT offset, event_id, event_type, target, namespace, timestamp, payload_json, metrics_json
 FROM evolution_events
 WHERE offset > ?
 ORDER BY offset ASC
 """
        params: list[Any] = [cursor]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        events = [
            EvolutionEvent(
                event_type=EvolutionEventType(row[2]),
                target=row[3],
                payload=json.loads(row[6]),
                metrics={key: float(value) for key, value in json.loads(row[7]).items()},
                namespace=row[4],
                timestamp=float(row[5]),
                event_id=row[1],
            )
            for row in rows
        ]
        next_cursor = int(rows[-1][0]) if rows else cursor
        return events, next_cursor

    def get_cursor(self, consumer: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cursor FROM evolution_cursors WHERE consumer = ?",
                (consumer,),
            ).fetchone()
            return 0 if row is None else int(row[0])

    def set_cursor(self, consumer: str, cursor: int) -> None:
        if cursor < 0:
            raise ValueError("cursor must be >= 0")
        with self._connect() as conn:
            conn.execute(
                """
 INSERT INTO evolution_cursors (consumer, cursor) VALUES (?, ?)
 ON CONFLICT(consumer) DO UPDATE SET cursor = excluded.cursor
 """,
                (consumer, cursor),
            )

    def get_active(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact | None:
        with self._connect() as conn:
            row = conn.execute(
                """
 SELECT a.artifact_id, a.artifact_type, a.content, a.version, a.metadata_json, a.parent_id
 FROM evolution_active_artifacts active
 JOIN evolution_artifacts a ON active.artifact_id = a.artifact_id
 WHERE active.artifact_type = ?
 """,
                (artifact_type.value,),
            ).fetchone()
            return None if row is None else _artifact_from_row(row)

    def set_active(self, artifact: EvolutionArtifact) -> None:
        with self._connect() as conn:
            current = conn.execute(
                "SELECT artifact_id FROM evolution_active_artifacts WHERE artifact_type = ?",
                (artifact.artifact_type.value,),
            ).fetchone()
            if current is not None and current[0] != artifact.artifact_id:
                conn.execute(
                    "UPDATE evolution_artifacts SET state = 'history' WHERE artifact_id = ?",
                    (current[0],),
                )
            self._upsert_artifact(conn, artifact, state="active")
            conn.execute(
                """
 INSERT INTO evolution_active_artifacts (artifact_type, artifact_id) VALUES (?, ?)
 ON CONFLICT(artifact_type) DO UPDATE SET artifact_id = excluded.artifact_id
 """,
                (artifact.artifact_type.value, artifact.artifact_id),
            )
            shadow_row = conn.execute(
                "SELECT artifact_id FROM evolution_shadow_artifacts WHERE artifact_type = ?",
                (artifact.artifact_type.value,),
            ).fetchone()
            if shadow_row is not None and shadow_row[0] == artifact.artifact_id:
                conn.execute(
                    "DELETE FROM evolution_shadow_artifacts WHERE artifact_type = ?",
                    (artifact.artifact_type.value,),
                )

    def stage(self, artifact: EvolutionArtifact) -> None:
        with self._connect() as conn:
            self._upsert_artifact(conn, artifact, state="staged")

    def list_staged(self, artifact_type: EvolutionArtifactType) -> list[EvolutionArtifact]:
        with self._connect() as conn:
            rows = conn.execute(
                """
 SELECT artifact_id, artifact_type, content, version, metadata_json, parent_id
 FROM evolution_artifacts
 WHERE artifact_type = ? AND state = 'staged'
 ORDER BY created_order ASC
 """,
                (artifact_type.value,),
            ).fetchall()
            return [_artifact_from_row(row) for row in rows]

    def activate(self, artifact_id: str) -> EvolutionArtifact:
        with self._connect() as conn:
            row = conn.execute(
                """
 SELECT artifact_id, artifact_type, content, version, metadata_json, parent_id
 FROM evolution_artifacts
 WHERE artifact_id = ? AND state = 'staged'
 """,
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"staged artifact not found: {artifact_id}")
            artifact = _artifact_from_row(row)
            current = conn.execute(
                "SELECT artifact_id FROM evolution_active_artifacts WHERE artifact_type = ?",
                (artifact.artifact_type.value,),
            ).fetchone()
            if current is not None and current[0] != artifact.artifact_id:
                conn.execute(
                    "UPDATE evolution_artifacts SET state = 'history' WHERE artifact_id = ?",
                    (current[0],),
                )
            conn.execute(
                "UPDATE evolution_artifacts SET state = 'active' WHERE artifact_id = ?",
                (artifact.artifact_id,),
            )
            conn.execute(
                """
 INSERT INTO evolution_active_artifacts (artifact_type, artifact_id) VALUES (?, ?)
 ON CONFLICT(artifact_type) DO UPDATE SET artifact_id = excluded.artifact_id
 """,
                (artifact.artifact_type.value, artifact.artifact_id),
            )
            return artifact

    def rollback(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact:
        with self._connect() as conn:
            row = conn.execute(
                """
 SELECT artifact_id, artifact_type, content, version, metadata_json, parent_id
 FROM evolution_artifacts
 WHERE artifact_type = ? AND state = 'history'
 ORDER BY created_order DESC
 LIMIT 1
 """,
                (artifact_type.value,),
            ).fetchone()
            if row is None:
                raise LookupError(f"no rollback artifact for {artifact_type.value}")
            previous = _artifact_from_row(row)
            active = conn.execute(
                "SELECT artifact_id FROM evolution_active_artifacts WHERE artifact_type = ?",
                (artifact_type.value,),
            ).fetchone()
            if active is not None:
                conn.execute(
                    "UPDATE evolution_artifacts SET state = 'rolled_back' WHERE artifact_id = ?",
                    (active[0],),
                )
                conn.execute(
                    "UPDATE evolution_artifacts SET state = 'active' WHERE artifact_id = ?",
                    (previous.artifact_id,),
                )
                conn.execute(
                    """
 INSERT INTO evolution_active_artifacts (artifact_type, artifact_id) VALUES (?, ?)
 ON CONFLICT(artifact_type) DO UPDATE SET artifact_id = excluded.artifact_id
 """,
                    (artifact_type.value, previous.artifact_id),
                )
            return previous

    def list_history(self, artifact_type: EvolutionArtifactType) -> list[EvolutionArtifact]:
        with self._connect() as conn:
            rows = conn.execute(
                """
 SELECT artifact_id, artifact_type, content, version, metadata_json, parent_id
 FROM evolution_artifacts
 WHERE artifact_type = ? AND state IN ('history', 'rolled_back')
 ORDER BY created_order ASC
 """,
                (artifact_type.value,),
            ).fetchall()
            return [_artifact_from_row(row) for row in rows]

    def revert_to(self, artifact_id: str) -> EvolutionArtifact:
        with self._connect() as conn:
            row = conn.execute(
                """
 SELECT artifact_id, artifact_type, content, version, metadata_json, parent_id, state
 FROM evolution_artifacts
 WHERE artifact_id = ?
 """,
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"artifact not found: {artifact_id}")
            target_state = row[6]
            if target_state not in {"history", "rolled_back", "active"}:
                raise LookupError(
                    f"artifact {artifact_id} is in state '{target_state}'; cannot revert to it"
                )
            target = _artifact_from_row(row[:6])
            current = conn.execute(
                "SELECT artifact_id FROM evolution_active_artifacts WHERE artifact_type = ?",
                (target.artifact_type.value,),
            ).fetchone()
            if current is not None and current[0] != target.artifact_id:
                conn.execute(
                    "UPDATE evolution_artifacts SET state = 'rolled_back' WHERE artifact_id = ?",
                    (current[0],),
                )
            conn.execute(
                "UPDATE evolution_artifacts SET state = 'active' WHERE artifact_id = ?",
                (target.artifact_id,),
            )
            conn.execute(
                """
 INSERT INTO evolution_active_artifacts (artifact_type, artifact_id) VALUES (?, ?)
 ON CONFLICT(artifact_type) DO UPDATE SET artifact_id = excluded.artifact_id
 """,
                (target.artifact_type.value, target.artifact_id),
            )
            return target

    def set_shadow(self, artifact: EvolutionArtifact) -> None:
        with self._connect() as conn:
            current = conn.execute(
                "SELECT artifact_id FROM evolution_shadow_artifacts WHERE artifact_type = ?",
                (artifact.artifact_type.value,),
            ).fetchone()
            if current is not None and current[0] != artifact.artifact_id:
                conn.execute(
                    "UPDATE evolution_artifacts SET state = 'staged' WHERE artifact_id = ?",
                    (current[0],),
                )
            self._upsert_artifact(conn, artifact, state="shadow")
            conn.execute(
                """
 INSERT INTO evolution_shadow_artifacts (artifact_type, artifact_id) VALUES (?, ?)
 ON CONFLICT(artifact_type) DO UPDATE SET artifact_id = excluded.artifact_id
 """,
                (artifact.artifact_type.value, artifact.artifact_id),
            )

    def get_shadow(self, artifact_type: EvolutionArtifactType) -> EvolutionArtifact | None:
        with self._connect() as conn:
            row = conn.execute(
                """
 SELECT a.artifact_id, a.artifact_type, a.content, a.version, a.metadata_json, a.parent_id
 FROM evolution_shadow_artifacts shadow
 JOIN evolution_artifacts a ON shadow.artifact_id = a.artifact_id
 WHERE shadow.artifact_type = ?
 """,
                (artifact_type.value,),
            ).fetchone()
            return None if row is None else _artifact_from_row(row)

    def clear_shadow(self, artifact_type: EvolutionArtifactType) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT artifact_id FROM evolution_shadow_artifacts WHERE artifact_type = ?",
                (artifact_type.value,),
            ).fetchone()
            if row is None:
                return
            conn.execute(
                "DELETE FROM evolution_shadow_artifacts WHERE artifact_type = ?",
                (artifact_type.value,),
            )
            conn.execute(
                "UPDATE evolution_artifacts SET state = 'staged' WHERE artifact_id = ?",
                (row[0],),
            )

    def append_audit(self, entry: AuditEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                """
 INSERT INTO evolution_audit (
 timestamp, consumer, action, cursor_before, cursor_after,
 events_consumed, skipped, reason, promotion_level, error
 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
 """,
                (
                    entry.timestamp,
                    entry.consumer,
                    entry.action,
                    entry.cursor_before,
                    entry.cursor_after,
                    entry.events_consumed,
                    1 if entry.skipped else 0,
                    entry.reason,
                    entry.promotion_level,
                    entry.error,
                ),
            )

    def read_audit(
        self,
        *,
        consumer: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEntry]:
        cols = (
            "timestamp, consumer, action, cursor_before, cursor_after, "
            "events_consumed, skipped, reason, promotion_level, error"
        )
        params: list[Any] = []
        if limit is None:
            sql = f"SELECT {cols} FROM evolution_audit"
            if consumer is not None:
                sql += " WHERE consumer = ?"
                params.append(consumer)
            sql += " ORDER BY audit_id ASC"
        else:
            inner = f"SELECT {cols}, audit_id FROM evolution_audit"
            if consumer is not None:
                inner += " WHERE consumer = ?"
                params.append(consumer)
            inner += " ORDER BY audit_id DESC LIMIT ?"
            params.append(limit)
            sql = f"SELECT {cols} FROM ({inner}) ORDER BY audit_id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            AuditEntry(
                timestamp=float(row[0]),
                consumer=row[1],
                action=row[2],
                cursor_before=int(row[3]),
                cursor_after=int(row[4]),
                events_consumed=int(row[5]),
                skipped=bool(row[6]),
                reason=row[7],
                promotion_level=row[8],
                error=row[9],
            )
            for row in rows
        ]

    def _upsert_artifact(
        self,
        conn: sqlite3.Connection,
        artifact: EvolutionArtifact,
        *,
        state: str,
    ) -> None:
        conn.execute(
            """
 INSERT INTO evolution_artifacts (
 artifact_id, artifact_type, content, version, metadata_json, parent_id, state, created_order
 ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT MAX(created_order) + 1 FROM evolution_artifacts), 1))
 ON CONFLICT(artifact_id) DO UPDATE SET
 content = excluded.content,
 version = excluded.version,
 metadata_json = excluded.metadata_json,
 parent_id = excluded.parent_id,
 state = excluded.state
 """,
            (
                artifact.artifact_id,
                artifact.artifact_type.value,
                artifact.content,
                artifact.version,
                json.dumps(artifact.metadata, sort_keys=True),
                artifact.parent_id,
                state,
            ),
        )


def _artifact_from_row(row: sqlite3.Row | tuple[Any, ...]) -> EvolutionArtifact:
    return EvolutionArtifact(
        artifact_id=row[0],
        artifact_type=EvolutionArtifactType(row[1]),
        content=row[2],
        version=int(row[3]),
        metadata={key: str(value) for key, value in json.loads(row[4]).items()},
        parent_id=row[5],
    )
