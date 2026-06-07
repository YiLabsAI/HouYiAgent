"""SQLite-based memory backend with FTS5 full-text search and embedding cache.

Schema:
 memories — main record table (scope+key UNIQUE)
 memories_fts — FTS5 virtual table, auto-synced via triggers
 memories_vec — sqlite-vec vec0 virtual table (created lazily,
 only when the extension is available; joined to
 memories via rowid)
 memories_vec_meta — single-row meta table storing the active vector
 dimension; required because vec0 tables are typed
 at CREATE time
 embedding_cache — per-provider/model embedding BLOB cache
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from houyi.adapters.memory.backends.base import MemoryBackend
from houyi.adapters.memory.backends.sqlite_connection import SQLiteConnectionManager
from houyi.adapters.memory.backends.sqlite_embedding_cache import SQLiteEmbeddingCache
from houyi.adapters.memory.backends.sqlite_extract_queue import SQLiteExtractQueue
from houyi.adapters.memory.backends.sqlite_fts_search import SQLiteFTSSearch
from houyi.adapters.memory.backends.sqlite_raw_turn_log import SQLiteRawTurnLog
from houyi.adapters.memory.backends.sqlite_schema import SQLiteSchemaManager
from houyi.adapters.memory.backends.sqlite_vector_search import SQLiteVectorSearch
from houyi.adapters.memory.types import (
    GraphTraversalResult,
    MemoryEdge,
    MemoryProvenance,
    MemoryRecord,
    MemoryRelation,
    MemoryScope,
    MemoryType,
    RawTurn,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 3


def _pack_floats(vec: list[float]) -> bytes:
    """Serialize float list to compact binary (float32)."""
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_floats(data: bytes) -> list[float]:
    """Deserialize binary to float list."""
    import struct

    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


class SQLiteMemoryBackend(MemoryBackend):
    """SQLite backend with FTS5 and embedding cache.

    Follows the same thread-safety pattern as SQLiteSpanStorage:
    WAL mode, check_same_thread=False, thread-local connections.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ):
        self._lock = threading.Lock()
        self._conn_manager = SQLiteConnectionManager(db_path, data_dir)
        self._schema_manager = SQLiteSchemaManager(self._conn_manager)
        self._in_transaction = threading.local()
        self._schema_manager.init_schema()

        self._vector_search = SQLiteVectorSearch(self._conn_manager, self._schema_manager)
        self._fts_search = SQLiteFTSSearch(self._conn_manager)
        self._embedding_cache = SQLiteEmbeddingCache(self._conn_manager)
        self._raw_turn_log = SQLiteRawTurnLog(self._conn_manager, self._lock)
        self._extract_queue = SQLiteExtractQueue(self._conn_manager, self._raw_turn_log, self._lock)

    def _conn(self) -> sqlite3.Connection:
        return self._conn_manager.get_connection()

    @property
    def _db_path(self) -> Path:
        """Expose db_path for test compatibility."""
        return self._conn_manager.db_path

    def _rowid_for(self, scope: MemoryScope, key: str) -> int | None:
        """Return the internal rowid for a (key, scope) pair, or None if missing."""
        return self._vector_search._rowid_for(scope, key)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def put(self, record: MemoryRecord) -> None:
        conn = self._conn()
        emb_blob = _pack_floats(record.embedding) if record.embedding else None
        embedding_pending = 0 if record.embedding else 1
        conn.execute(
            """
                INSERT INTO memories
                (record_id, scope, key, content, memory_type, tags, confidence,
                decay, provenance, metadata, created_at, updated_at, ttl,
                valid_from, valid_to, embedding,
                embedding_pending, embedding_provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                embedding=excluded.embedding,
                embedding_pending=excluded.embedding_pending,
                embedding_provider=excluded.embedding_provider
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
                embedding_pending,
                record.metadata.get("embedding_provider")
                if isinstance(record.metadata, dict)
                else None,
            ),
        )
        if record.embedding:
            self._vector_search.upsert_vec_row(record.scope, record.key, record.embedding)
        else:
            self._vector_search.delete_vec_row(record.scope, record.key)
        if not getattr(self._in_transaction, "active", False):
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

    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        row = (
            self._conn()
            .execute(
                "SELECT * FROM memories WHERE record_id=?",
                (record_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        record = self._row_to_record(row)
        if record.is_expired:
            self.delete(record.key, record.scope)
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
        rowid = self._vector_search._rowid_for(scope, key)
        cur = conn.execute(
            "DELETE FROM memories WHERE scope=? AND key=?",
            (scope.value, key),
        )
        if (
            rowid is not None
            and self._conn_manager.vec_available
            and self._conn_manager.vec_dim is not None
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("DELETE FROM memories_vec WHERE rowid=?", (rowid,))
        if not getattr(self._in_transaction, "active", False):
            conn.commit()
        return cur.rowcount > 0

    def clear(self, scope: MemoryScope | None = None) -> int:
        conn = self._conn()
        if scope is None:
            cur = conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM embedding_cache")
            conn.execute("DELETE FROM entity_state")
            conn.execute("DELETE FROM vague_candidates")
            conn.execute("DELETE FROM memory_edges")
            conn.execute("DELETE FROM memory_community_labels")
            if self._conn_manager.vec_available and self._conn_manager.vec_dim is not None:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute("DELETE FROM memories_vec")
        else:
            rowids: list[int] = [
                int(r["rowid"])
                for r in conn.execute("SELECT rowid FROM memories WHERE scope=?", (scope.value,))
            ]
            cur = conn.execute("DELETE FROM memories WHERE scope=?", (scope.value,))
            if (
                rowids
                and self._conn_manager.vec_available
                and self._conn_manager.vec_dim is not None
            ):
                with contextlib.suppress(sqlite3.OperationalError):
                    placeholders = ",".join("?" * len(rowids))
                    conn.execute(
                        f"DELETE FROM memories_vec WHERE rowid IN ({placeholders})",
                        rowids,
                    )
        if not getattr(self._in_transaction, "active", False):
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
        return self._fts_search.search_fts(query, scope, limit)

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        record_id: str,
        provider: str,
        model: str,
    ) -> list[float] | None:
        return self._embedding_cache.get_embedding(record_id, provider, model)

    def put_embedding(
        self,
        record_id: str,
        provider: str,
        model: str,
        embedding: list[float],
    ) -> None:
        self._embedding_cache.put_embedding(record_id, provider, model, embedding)

    def search_embedding(
        self,
        query_embedding: list[float],
        scope: MemoryScope | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        return self.search_vector(query_embedding, scope=scope, limit=limit)

    def search_vector(
        self,
        query_embedding: list[float],
        *,
        scope: MemoryScope | None = None,
        rowid_filter: list[int] | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        return self._vector_search.search_vector(
            query_embedding, scope=scope, rowid_filter=rowid_filter, limit=limit
        )

    def prefilter_rowids(
        self,
        *,
        scopes: list[MemoryScope] | None = None,
        updated_after: float | None = None,
        updated_before: float | None = None,
        limit: int = 3000,
    ) -> list[int]:
        """Return a bounded, recency-ordered rowid subset for ANN prefilter.

        This method performs predicate pushdown in SQL before vector search to
        avoid full-index distance scans on large corpora.
        """
        conn = self._conn()
        now = time.time()

        clauses = [
            "(ttl IS NULL OR created_at + ttl > ?)",
            "(valid_to IS NULL OR valid_to > ?)",
        ]
        params: list[Any] = [now, now]

        if scopes:
            placeholders = ",".join("?" for _ in scopes)
            clauses.append(f"scope IN ({placeholders})")
            params.extend(scope.value for scope in scopes)

        if updated_after is not None:
            clauses.append("updated_at >= ?")
            params.append(updated_after)

        if updated_before is not None:
            clauses.append("updated_at <= ?")
            params.append(updated_before)

        where_sql = " AND ".join(clauses)
        rows = conn.execute(
            f"SELECT rowid FROM memories WHERE {where_sql} ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [int(row["rowid"]) for row in rows]

    # ------------------------------------------------------------------
    # Embedding backfill bookkeeping
    # ------------------------------------------------------------------

    def list_pending_embeddings(self, limit: int = 64) -> list[tuple[int, MemoryRecord]]:
        rows = (
            self._conn()
            .execute(
                "SELECT rowid, * FROM memories "
                "WHERE embedding_pending = 1 "
                "ORDER BY created_at ASC "
                "LIMIT ?",
                (limit,),
            )
            .fetchall()
        )
        out: list[tuple[int, MemoryRecord]] = []
        for row in rows:
            record = self._row_to_record(row)
            if record.is_expired:
                continue
            out.append((int(row["rowid"]), record))
        return out

    def mark_embedding_filled(
        self,
        rowid: int,
        embedding: list[float],
        *,
        provider: str | None = None,
    ) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE memories SET embedding = ?, embedding_pending = 0, "
            "embedding_provider = COALESCE(?, embedding_provider) "
            "WHERE rowid = ?",
            (_pack_floats(embedding), provider, rowid),
        )
        if self._schema_manager.ensure_vec_table(len(embedding)):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, _pack_floats(embedding)),
                )
        if not getattr(self._in_transaction, "active", False):
            conn.commit()

    # ------------------------------------------------------------------
    # Raw turn log (L0)
    # ------------------------------------------------------------------

    def append_raw_turn(self, turn: RawTurn) -> RawTurn:
        return self._raw_turn_log.append_raw_turn(turn)

    def list_raw_turns(
        self,
        namespace: str,
        session_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[RawTurn]:
        return self._raw_turn_log.list_raw_turns(namespace, session_id, limit=limit, offset=offset)

    def count_raw_turns(self, namespace: str, session_id: str) -> int:
        return self._raw_turn_log.count_raw_turns(namespace, session_id)

    def get_raw_turn(self, turn_id: str) -> RawTurn | None:
        return self._raw_turn_log.get_raw_turn(turn_id)

    # ------------------------------------------------------------------
    # Extract queue (L0 → L1 hand-off)
    # ------------------------------------------------------------------

    def enqueue_extract(
        self,
        turn: RawTurn,
        *,
        now: float | None = None,
    ) -> str:
        return self._extract_queue.enqueue_extract(turn, now=now)

    def claim_extract_jobs(
        self,
        *,
        limit: int = 8,
        namespace: str | None = None,
        lease_seconds: float = 60.0,
        now: float | None = None,
    ) -> list[tuple[str, RawTurn]]:
        return self._extract_queue.claim_extract_jobs(
            limit=limit, namespace=namespace, lease_seconds=lease_seconds, now=now
        )

    def mark_extract_done(self, queue_id: str) -> None:
        self._extract_queue.mark_extract_done(queue_id)

    def mark_extract_failed(
        self,
        queue_id: str,
        error: str,
        *,
        retry: bool = True,
        max_attempts: int = 5,
    ) -> None:
        self._extract_queue.mark_extract_failed(
            queue_id, error, retry=retry, max_attempts=max_attempts
        )

    def extract_queue_stats(self) -> dict[str, int]:
        return self._extract_queue.extract_queue_stats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Any:
        """Return an immediate transaction context spanning the entire backend."""
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        was_active = getattr(self._in_transaction, "active", False)
        self._in_transaction.active = True
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._in_transaction.active = was_active

    def close(self) -> None:
        self._conn_manager.close_all()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # GraphIndex (HouYi-Mesh)
    # ------------------------------------------------------------------

    def add_edge(self, edge: MemoryEdge) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO memory_edges
            (edge_id, namespace, source_unit_id, target_unit_id,
             source_type, target_type, relation, weight,
             valid_from, valid_to, created_at, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
            weight=excluded.weight,
            valid_from=excluded.valid_from,
            valid_to=excluded.valid_to,
            provenance=excluded.provenance
            """,
            (
                edge.edge_id,
                edge.namespace,
                edge.source_unit_id,
                edge.target_unit_id,
                edge.source_type,
                edge.target_type,
                edge.relation.value,
                edge.weight,
                edge.valid_from,
                edge.valid_to,
                edge.created_at,
                edge.provenance,
            ),
        )
        if not getattr(self._in_transaction, "active", False):
            conn.commit()

    def delete_edge(self, edge_id: str) -> bool:
        conn = self._conn()
        cur = conn.execute("DELETE FROM memory_edges WHERE edge_id=?", (edge_id,))
        if not getattr(self._in_transaction, "active", False):
            conn.commit()
        return cur.rowcount > 0

    def invalidate_edge(self, edge_id: str, valid_to: float) -> bool:
        conn = self._conn()
        cur = conn.execute(
            "UPDATE memory_edges SET valid_to=? WHERE edge_id=? AND valid_to IS NULL",
            (valid_to, edge_id),
        )
        if not getattr(self._in_transaction, "active", False):
            conn.commit()
        return cur.rowcount > 0

    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        row = (
            self._conn()
            .execute("SELECT * FROM memory_edges WHERE edge_id=?", (edge_id,))
            .fetchone()
        )
        if row is None:
            return None
        d = dict(row)
        return MemoryEdge(
            edge_id=d["edge_id"],
            namespace=d["namespace"],
            source_unit_id=d["source_unit_id"],
            target_unit_id=d["target_unit_id"],
            source_type=d["source_type"],
            target_type=d["target_type"],
            relation=MemoryRelation(d["relation"]),
            weight=d["weight"],
            valid_from=d["valid_from"],
            valid_to=d.get("valid_to"),
            created_at=d["created_at"],
            provenance=d.get("provenance"),
        )

    def get_community_id(
        self,
        namespace: str,
        node_type: str,
        node_id: str,
    ) -> str | None:
        row = (
            self._conn()
            .execute(
                """
                SELECT community_id FROM memory_community_labels
                WHERE namespace=? AND node_type=? AND node_id=?
                """,
                (namespace, node_type, node_id),
            )
            .fetchone()
        )
        return row[0] if row is not None else None

    def put_community_label(
        self,
        namespace: str,
        node_type: str,
        node_id: str,
        community_id: str,
        weight: float = 1.0,
        updated_at: float | None = None,
    ) -> None:
        ts = updated_at if updated_at is not None else time.time()
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO memory_community_labels
            (namespace, node_type, node_id, community_id, weight, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, node_type, node_id) DO UPDATE SET
            community_id=excluded.community_id,
            weight=excluded.weight,
            updated_at=excluded.updated_at
            """,
            (namespace, node_type, node_id, community_id, weight, ts),
        )
        if not getattr(self._in_transaction, "active", False):
            conn.commit()

    def traverse_graph(
        self,
        *,
        namespace: str,
        start_nodes: list[tuple[str, str]],
        max_depth: int = 3,
        direction: str = "bidirectional",
        as_of: float | None = None,
        relation_types: list[str] | None = None,
    ) -> list[GraphTraversalResult]:
        if not start_nodes:
            return []

        ts = as_of if as_of is not None else time.time()
        conn = self._conn()

        relation_filter_clause = ""
        if relation_types:
            placeholders = ",".join("?" for _ in relation_types)
            relation_filter_clause = f"AND e.relation IN ({placeholders})"

        anchor_parts = []
        anchor_params = []
        for node_id, node_type in start_nodes:
            anchor_parts.append("SELECT ?, ?, 0, ',' || ? || ',', NULL, 1.0, NULL")
            anchor_params.extend([node_id, node_type, node_id])
        anchor_sql = " UNION ALL ".join(anchor_parts)

        # Decide query based on direction
        if direction == "forward":
            query = f"""
                WITH RECURSIVE bfs(node_id, node_type, depth, path, relation, weight, parent_node_id) AS (
                    {anchor_sql}
                    UNION ALL
                    SELECT
                        e.target_unit_id,
                        e.target_type,
                        b.depth + 1,
                        b.path || e.target_unit_id || ',',
                        e.relation,
                        e.weight,
                        b.node_id
                    FROM memory_edges e
                    JOIN bfs b ON (e.source_unit_id = b.node_id AND e.source_type = b.node_type)
                    WHERE b.depth < ?
                      AND e.namespace = ?
                      AND e.valid_from <= ?
                      AND (e.valid_to IS NULL OR e.valid_to > ?)
                      AND instr(b.path, ',' || e.target_unit_id || ',') = 0
                      {relation_filter_clause}
                )
                SELECT node_id, node_type, depth, relation, weight, parent_node_id FROM bfs WHERE depth > 0;
            """
        elif direction == "backward":
            query = f"""
                WITH RECURSIVE bfs(node_id, node_type, depth, path, relation, weight, parent_node_id) AS (
                    {anchor_sql}
                    UNION ALL
                    SELECT
                        e.source_unit_id,
                        e.source_type,
                        b.depth + 1,
                        b.path || e.source_unit_id || ',',
                        e.relation,
                        e.weight,
                        b.node_id
                    FROM memory_edges e
                    JOIN bfs b ON (e.target_unit_id = b.node_id AND e.target_type = b.node_type)
                    WHERE b.depth < ?
                      AND e.namespace = ?
                      AND e.valid_from <= ?
                      AND (e.valid_to IS NULL OR e.valid_to > ?)
                      AND instr(b.path, ',' || e.source_unit_id || ',') = 0
                      {relation_filter_clause}
                )
                SELECT node_id, node_type, depth, relation, weight, parent_node_id FROM bfs WHERE depth > 0;
            """
        else:  # bidirectional
            query = f"""
                WITH RECURSIVE bfs(node_id, node_type, depth, path, relation, weight, parent_node_id) AS (
                    {anchor_sql}
                    UNION ALL
                    SELECT
                        CASE WHEN e.source_unit_id = b.node_id THEN e.target_unit_id ELSE e.source_unit_id END,
                        CASE WHEN e.source_unit_id = b.node_id THEN e.target_type ELSE e.source_type END,
                        b.depth + 1,
                        b.path || (CASE WHEN e.source_unit_id = b.node_id THEN e.target_unit_id ELSE e.source_unit_id END) || ',',
                        e.relation,
                        e.weight,
                        b.node_id
                    FROM memory_edges e
                    JOIN bfs b ON (
                        (e.source_unit_id = b.node_id AND e.source_type = b.node_type)
                        OR (e.target_unit_id = b.node_id AND e.target_type = b.node_type)
                    )
                    WHERE b.depth < ?
                      AND e.namespace = ?
                      AND e.valid_from <= ?
                      AND (e.valid_to IS NULL OR e.valid_to > ?)
                      AND instr(b.path, ',' || (CASE WHEN e.source_unit_id = b.node_id THEN e.target_unit_id ELSE e.source_unit_id END) || ',') = 0
                      {relation_filter_clause}
                )
                SELECT node_id, node_type, depth, relation, weight, parent_node_id FROM bfs WHERE depth > 0;
            """

        params: tuple[Any, ...] = (*anchor_params, max_depth, namespace, ts, ts)
        if relation_types:
            params = (*params, *relation_types)

        rows = conn.execute(query, params).fetchall()

        visited: dict[tuple[str, str], tuple[int, str | None, float | None, str | None]] = {}
        for row in rows:
            key = (row["node_id"], row["node_type"])
            depth = int(row["depth"])
            rel = row["relation"]
            wt = row["weight"]
            parent = row["parent_node_id"]
            if key not in visited or depth < visited[key][0]:
                visited[key] = (depth, rel, wt, parent)

        return [
            GraphTraversalResult(
                node_id=k[0],
                node_type=k[1],
                depth=d,
                last_edge_relation=r,
                last_edge_weight=w,
                parent_node_id=p,
            )
            for k, (d, r, w, p) in visited.items()
        ]

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
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
            if not getattr(self._in_transaction, "active", False):
                conn.commit()
        return result
