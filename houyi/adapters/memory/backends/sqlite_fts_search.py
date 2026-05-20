"""SQLite FTS5 full-text search operations."""

from __future__ import annotations

import re
from typing import Any

from houyi.adapters.memory.types import MemoryRecord, MemoryScope


class SQLiteFTSSearch:
    """Handles FTS5 full-text search operations."""

    _FTS_MAX_TERMS = 32
    _FTS_MAX_TERM_LEN = 64

    def __init__(self, conn_manager):
        self._conn_manager = conn_manager

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
                self._conn_manager.get_connection()
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
                self._conn_manager.get_connection()
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
            record = _row_to_record(row)
            if not record.is_expired:
                score = -dict(row)["fts_rank"]
                results.append((record, score))
        return results

    @staticmethod
    def _sanitize_fts(query: str) -> str:
        """Build an FTS5 OR query from tokenized input.

        - Unicode-aware (\\w+ matches CJK and accented letters).
        - Deduplicates tokens while preserving order.
        - Caps both the number of terms and each term's length so the
          generated MATCH expression stays bounded.
        - Wraps each term in double quotes so FTS5 treats it as a literal,
          guarding against operator injection (AND / OR / NEAR).
        """
        seen: set[str] = set()
        terms: list[str] = []
        for raw in re.findall(r"\w+", query, flags=re.UNICODE):
            token = raw[: SQLiteFTSSearch._FTS_MAX_TERM_LEN].lower()
            if not token or token in seen:
                continue
            seen.add(token)
            terms.append(token)
            if len(terms) >= SQLiteFTSSearch._FTS_MAX_TERMS:
                break
        if not terms:
            return ""
        escaped = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in terms]
        return " OR ".join(escaped)


def _row_to_record(row: Any) -> MemoryRecord:
    """Convert SQLite row to MemoryRecord."""
    import contextlib
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
        from houyi.adapters.memory.backends.sqlite_vector_search import _unpack_floats

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
