"""SQLite vector search operations.

Handles vector similarity search using sqlite-vec extension with Python fallback.
"""

from __future__ import annotations

import contextlib
import logging
import math
import sqlite3
import struct
import time
from typing import Any, Literal

from houyi.adapters.memory.types import MemoryRecord, MemoryScope

VectorSearchPath = Literal["vec0", "scan"]

logger = logging.getLogger(__name__)


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


class SQLiteVectorSearch:
    """Handles vector similarity search operations."""

    def __init__(self, conn_manager, schema_manager):
        self._conn_manager = conn_manager
        self._schema_manager = schema_manager
        # Records the path used by the most recent search_vector call.
        # None until the first call. Useful for bench / health checks
        # that need to assert which physical path served a query.
        self._last_search_path: VectorSearchPath | None = None

    @property
    def last_search_path(self) -> VectorSearchPath | None:
        """Return the path used by the most recent search_vector call.

        vec0 means the sqlite-vec extension served the query; scan
        means the Python full-table cosine fallback ran. None before
        any call.
        """
        return self._last_search_path

    def search_vector(
        self,
        query_embedding: list[float],
        *,
        scope: MemoryScope | None = None,
        rowid_filter: list[int] | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """Two-stage friendly vector search.

        When sqlite-vec is available and the active memories_vec dimension
        matches len(query_embedding), the ranking runs as a single SQL
        statement and joins back to the main table by rowid. Otherwise we
        fall back to a brute-force Python cosine pass over the BLOB column.
        """
        if (
            self._conn_manager.vec_available
            and self._conn_manager.vec_dim is not None
            and self._conn_manager.vec_dim == len(query_embedding)
        ):
            try:
                result = self._search_vector_vec0(
                    query_embedding,
                    scope=scope,
                    rowid_filter=rowid_filter,
                    limit=limit,
                )
            except sqlite3.OperationalError:
                # vec0 lookup raised at runtime (e.g. dim mismatch the
                # cached vec_dim did not catch). Fall through to scan
                # so the call still returns results.
                pass
            else:
                self._last_search_path = "vec0"
                return result
        result = self._search_vector_scan(
            query_embedding,
            scope=scope,
            rowid_filter=rowid_filter,
            limit=limit,
        )
        self._last_search_path = "scan"
        return result

    def _search_vector_vec0(
        self,
        query_embedding: list[float],
        *,
        scope: MemoryScope | None,
        rowid_filter: list[int] | None,
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        conn = self._conn_manager.get_connection()
        now = time.time()
        params: list[Any] = [_pack_floats(query_embedding), limit]
        sql = (
            "SELECT m.*, v.distance AS vec_distance "
            "FROM (SELECT rowid, distance FROM memories_vec "
            " WHERE embedding MATCH ? AND k = ?) AS v "
            "JOIN memories m ON m.rowid = v.rowid "
        )
        where: list[str] = ["(m.valid_to IS NULL OR m.valid_to > ?)"]
        params_suffix: list[Any] = [now]
        if scope is not None:
            where.append("m.scope = ?")
            params_suffix.append(scope.value)
        if rowid_filter is not None:
            if not rowid_filter:
                return []
            placeholders = ",".join("?" * len(rowid_filter))
            where.append(f"m.rowid IN ({placeholders})")
            params_suffix.extend(rowid_filter)
        if where:
            sql += "WHERE " + " AND ".join(where) + " "
        sql += "ORDER BY v.distance ASC"
        params.extend(params_suffix)
        rows = conn.execute(sql, params).fetchall()

        results: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            record = _row_to_record(row)
            if not record.is_active:
                continue
            distance = float(dict(row).get("vec_distance") or 0.0)
            # sqlite-vec MATCH returns L2 distance for vec0 by default.
            # Assuming normalized vectors: L2^2 = 2 - 2*cos => cos = 1 - L2^2 / 2
            cosine_sim = 1.0 - (distance * distance) / 2.0
            similarity = max(0.0, min(1.0, cosine_sim))
            results.append((record, similarity))
        return results

    def _search_vector_scan(
        self,
        query_embedding: list[float],
        *,
        scope: MemoryScope | None,
        rowid_filter: list[int] | None,
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        """O(N) full-table scan fallback for vector recall.

        Used when the sqlite-vec extension is unavailable or when the
        query embedding's dimension does not match the stored vec0
        table. Iterates every row with a non-null embedding blob,
        decodes the vector, and computes cosine similarity in Python.
        Cost grows linearly with the number of memory rows; acceptable
        for dev / CI / small datasets but not production-scale recall.
        """
        now = time.time()
        sql = "SELECT * FROM memories WHERE embedding IS NOT NULL AND (valid_to IS NULL OR valid_to > ?)"
        params: list[Any] = [now]
        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope.value)
        if rowid_filter is not None:
            if not rowid_filter:
                return []
            placeholders = ",".join("?" * len(rowid_filter))
            sql += f" AND rowid IN ({placeholders})"
            params.extend(rowid_filter)
        rows = self._conn_manager.get_connection().execute(sql, params).fetchall()

        scored: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            record = _row_to_record(row)
            if not record.is_active or record.embedding is None:
                continue
            sim = _cosine_sim(query_embedding, record.embedding)
            scored.append((record, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def upsert_vec_row(self, scope: MemoryScope, key: str, embedding: list[float]) -> None:
        if not self._schema_manager.ensure_vec_table(len(embedding)):
            return
        rowid = self._rowid_for(scope, key)
        if rowid is None:
            return
        conn = self._conn_manager.get_connection()
        blob = _pack_floats(embedding)
        try:
            conn.execute("DELETE FROM memories_vec WHERE rowid=?", (rowid,))
            conn.execute(
                "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )
        except sqlite3.OperationalError as exc:
            logger.warning("memories_vec upsert failed for rowid=%s: %s", rowid, exc)

    def delete_vec_row(self, scope: MemoryScope, key: str) -> None:
        if not self._conn_manager.vec_available or self._conn_manager.vec_dim is None:
            return
        rowid = self._rowid_for(scope, key)
        if rowid is None:
            return
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn_manager.get_connection().execute(
                "DELETE FROM memories_vec WHERE rowid=?", (rowid,)
            )

    def _rowid_for(self, scope: MemoryScope, key: str) -> int | None:
        row = (
            self._conn_manager.get_connection()
            .execute(
                "SELECT rowid FROM memories WHERE scope=? AND key=?",
                (scope.value, key),
            )
            .fetchone()
        )
        return None if row is None else int(row["rowid"])


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    """Convert SQLite row to MemoryRecord."""
    import json

    from houyi.adapters.memory.types import MemoryProvenance, MemoryType

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
