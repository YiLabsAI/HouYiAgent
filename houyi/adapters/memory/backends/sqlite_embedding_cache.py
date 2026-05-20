"""SQLite embedding cache operations."""

from __future__ import annotations

import time

from houyi.adapters.memory.backends.sqlite_vector_search import _pack_floats, _unpack_floats


class SQLiteEmbeddingCache:
    """Handles embedding cache operations."""

    def __init__(self, conn_manager):
        self._conn_manager = conn_manager

    def get_embedding(self, record_id: str, provider: str, model: str) -> list[float] | None:
        row = (
            self._conn_manager.get_connection()
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
        self, record_id: str, provider: str, model: str, embedding: list[float]
    ) -> None:
        conn = self._conn_manager.get_connection()
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
