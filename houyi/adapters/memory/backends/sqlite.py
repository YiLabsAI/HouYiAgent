"""SQLite-based memory backend with FTS5 full-text search and embedding cache.

Schema:
  memories      — main record table (scope+key UNIQUE)
  memories_fts  — FTS5 virtual table, auto-synced via triggers
  embedding_cache — per-provider/model embedding BLOB cache
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any

from houyi.adapters.memory.backends.base import MemoryBackend
from houyi.adapters.memory.types import (
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def _pack_floats(vec: list[float]) -> bytes:
    """Serialize float list to compact binary (float32)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_floats(data: bytes) -> list[float]:
    """Deserialize binary to float list."""
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class SQLiteMemoryBackend(MemoryBackend):
    """SQLite backend with FTS5 and embedding cache.

    Follows the same thread-safety pattern as ``SQLiteSpanStorage``:
    WAL mode, ``check_same_thread=False``, thread-local connections.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ):
        if db_path:
            self._db_path = Path(db_path)
        elif data_dir:
            self._db_path = Path(data_dir) / ".houyi" / "memory.db"
        else:
            self._db_path = Path(".houyi") / "memory.db"

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
            with self._lock:
                self._connections.add(conn)
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._conn()
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    record_id   TEXT PRIMARY KEY,
                    scope       TEXT NOT NULL,
                    key         TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'fact',
                    tags        TEXT DEFAULT '[]',
                    confidence  REAL DEFAULT 1.0,
                    decay       REAL DEFAULT 1.0,
                    provenance  TEXT,
                    metadata    TEXT DEFAULT '{}',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    ttl         REAL,
                    valid_from  REAL,
                    valid_to    REAL,
                    embedding   BLOB,
                    UNIQUE(scope, key)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_updated ON memories(updated_at DESC)")

            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    key, content, tags,
                    content='memories',
                    content_rowid='rowid',
                    tokenize='unicode61'
                )
            """)

            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_fts_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, key, content, tags)
                    VALUES (new.rowid, new.key, new.content, new.tags);
                END
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_fts_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, key, content, tags)
                    VALUES ('delete', old.rowid, old.key, old.content, old.tags);
                END
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_fts_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, key, content, tags)
                    VALUES ('delete', old.rowid, old.key, old.content, old.tags);
                    INSERT INTO memories_fts(rowid, key, content, tags)
                    VALUES (new.rowid, new.key, new.content, new.tags);
                END
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    record_id  TEXT NOT NULL,
                    provider   TEXT NOT NULL,
                    model      TEXT NOT NULL,
                    embedding  BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (record_id, provider, model),
                    FOREIGN KEY (record_id) REFERENCES memories(record_id) ON DELETE CASCADE
                )
            """)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def put(self, record: MemoryRecord) -> None:
        conn = self._conn()
        emb_blob = _pack_floats(record.embedding) if record.embedding else None
        conn.execute(
            """
            INSERT INTO memories
                (record_id, scope, key, content, memory_type, tags, confidence,
                 decay, provenance, metadata, created_at, updated_at, ttl,
                 valid_from, valid_to, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, key) DO UPDATE SET
                content=excluded.content,
                memory_type=excluded.memory_type,
                tags=excluded.tags,
                confidence=excluded.confidence,
                decay=excluded.decay,
                provenance=excluded.provenance,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at,
                ttl=excluded.ttl,
                valid_from=excluded.valid_from,
                valid_to=excluded.valid_to,
                embedding=excluded.embedding
            """,
            (
                record.record_id,
                record.scope.value,
                record.key,
                record.content,
                record.memory_type.value,
                json.dumps(record.tags, ensure_ascii=False),
                record.confidence,
                record.decay,
                record.provenance.model_dump_json() if record.provenance else None,
                json.dumps(record.metadata, ensure_ascii=False),
                record.created_at,
                record.updated_at,
                record.ttl,
                record.valid_from,
                record.valid_to,
                emb_blob,
            ),
        )
        conn.commit()

    def get(self, key: str, scope: MemoryScope) -> MemoryRecord | None:
        row = (
            self._conn()
            .execute(
                "SELECT * FROM memories WHERE scope=? AND key=?",
                (scope.value, key),
            )
            .fetchone()
        )
        if row is None:
            return None
        record = self._row_to_record(row)
        if record.is_expired:
            self.delete(key, scope)
            return None
        return record

    def list_by_scope(self, scope: MemoryScope) -> list[MemoryRecord]:
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM memories WHERE scope=? ORDER BY updated_at DESC",
                (scope.value,),
            )
            .fetchall()
        )
        return self._filter_expired(rows)

    def list_by_type(
        self,
        memory_type: MemoryType,
        scope: MemoryScope | None = None,
    ) -> list[MemoryRecord]:
        if scope is not None:
            rows = (
                self._conn()
                .execute(
                    "SELECT * FROM memories WHERE memory_type=? AND scope=? ORDER BY updated_at DESC",
                    (memory_type.value, scope.value),
                )
                .fetchall()
            )
        else:
            rows = (
                self._conn()
                .execute(
                    "SELECT * FROM memories WHERE memory_type=? ORDER BY updated_at DESC",
                    (memory_type.value,),
                )
                .fetchall()
            )
        return self._filter_expired(rows)

    def all_records(self, *, include_expired: bool = False) -> list[MemoryRecord]:
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM memories ORDER BY updated_at DESC",
            )
            .fetchall()
        )
        if include_expired:
            return [self._row_to_record(r) for r in rows]
        return self._filter_expired(rows)

    def delete(self, key: str, scope: MemoryScope) -> bool:
        conn = self._conn()
        cur = conn.execute(
            "DELETE FROM memories WHERE scope=? AND key=?",
            (scope.value, key),
        )
        conn.commit()
        return cur.rowcount > 0

    def clear(self, scope: MemoryScope | None = None) -> int:
        conn = self._conn()
        if scope is None:
            cur = conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM embedding_cache")
        else:
            cur = conn.execute("DELETE FROM memories WHERE scope=?", (scope.value,))
        conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # FTS5 full-text search
    # ------------------------------------------------------------------

    def search_fts(
        self,
        query: str,
        scope: MemoryScope | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        fts_query = self._sanitize_fts(query)
        if not fts_query:
            return []

        if scope is not None:
            rows = (
                self._conn()
                .execute(
                    """
                SELECT m.*, bm25(memories_fts) AS fts_rank
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.rowid
                WHERE memories_fts MATCH ? AND m.scope = ?
                ORDER BY bm25(memories_fts)
                LIMIT ?
                """,
                    (fts_query, scope.value, limit),
                )
                .fetchall()
            )
        else:
            rows = (
                self._conn()
                .execute(
                    """
                SELECT m.*, bm25(memories_fts) AS fts_rank
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.rowid
                WHERE memories_fts MATCH ?
                ORDER BY bm25(memories_fts)
                LIMIT ?
                """,
                    (fts_query, limit),
                )
                .fetchall()
            )

        results = []
        for row in rows:
            record = self._row_to_record(row)
            if not record.is_expired:
                score = -dict(row)["fts_rank"]
                results.append((record, score))
        return results

    @staticmethod
    def _sanitize_fts(query: str) -> str:
        """Escape special FTS5 characters and build OR query from terms."""
        import re

        terms = re.findall(r"\w+", query)
        if not terms:
            return ""
        escaped = [f'"{t}"' for t in terms]
        return " OR ".join(escaped)

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        record_id: str,
        provider: str,
        model: str,
    ) -> list[float] | None:
        row = (
            self._conn()
            .execute(
                "SELECT embedding FROM embedding_cache WHERE record_id=? AND provider=? AND model=?",
                (record_id, provider, model),
            )
            .fetchone()
        )
        if row is None:
            return None
        return _unpack_floats(row["embedding"])

    def put_embedding(
        self,
        record_id: str,
        provider: str,
        model: str,
        embedding: list[float],
    ) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO embedding_cache (record_id, provider, model, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(record_id, provider, model) DO UPDATE SET
                embedding=excluded.embedding,
                created_at=excluded.created_at
            """,
            (record_id, provider, model, _pack_floats(embedding), time.time()),
        )
        conn.commit()

    def search_embedding(
        self,
        query_embedding: list[float],
        scope: MemoryScope | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """Brute-force cosine search over in-row embeddings.

        For < 10K records this is sub-100ms. Larger stores should use
        a dedicated vector index (e.g. sqlite-vec extension).
        """
        if scope is not None:
            rows = (
                self._conn()
                .execute(
                    "SELECT * FROM memories WHERE embedding IS NOT NULL AND scope=?",
                    (scope.value,),
                )
                .fetchall()
            )
        else:
            rows = (
                self._conn()
                .execute(
                    "SELECT * FROM memories WHERE embedding IS NOT NULL",
                )
                .fetchall()
            )

        scored: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            record = self._row_to_record(row)
            if record.is_expired or record.embedding is None:
                continue
            sim = _cosine_sim(query_embedding, record.embedding)
            scored.append((record, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            for conn in self._connections:
                with contextlib.suppress(Exception):
                    conn.close()
            self._connections.clear()
        self._local.conn = None

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        d: dict[str, Any] = dict(row)
        prov = None
        if d.get("provenance"):
            with contextlib.suppress(Exception):
                prov = MemoryProvenance.model_validate_json(d["provenance"])

        emb = None
        emb_raw = d.get("embedding")
        if isinstance(emb_raw, bytes) and len(emb_raw) >= 4:
            emb = _unpack_floats(emb_raw)

        tags_raw = d.get("tags", "[]")
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else []
        except (json.JSONDecodeError, TypeError):
            tags = []

        meta_raw = d.get("metadata", "{}")
        try:
            metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return MemoryRecord(
            record_id=d["record_id"],
            scope=MemoryScope(d["scope"]),
            key=d["key"],
            content=d["content"],
            memory_type=MemoryType(d["memory_type"]),
            tags=tags,
            confidence=d.get("confidence", 1.0),
            decay=d.get("decay", 1.0),
            provenance=prov,
            metadata=metadata,
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            ttl=d.get("ttl"),
            valid_from=d.get("valid_from"),
            valid_to=d.get("valid_to"),
            embedding=emb,
        )

    def _filter_expired(self, rows: list[sqlite3.Row]) -> list[MemoryRecord]:
        result = []
        expired_keys: list[tuple[str, str]] = []
        for row in rows:
            record = self._row_to_record(row)
            if record.is_expired:
                expired_keys.append((record.key, record.scope.value))
            else:
                result.append(record)
        if expired_keys:
            conn = self._conn()
            for key, scope_val in expired_keys:
                conn.execute(
                    "DELETE FROM memories WHERE key=? AND scope=?",
                    (key, scope_val),
                )
            conn.commit()
        return result
