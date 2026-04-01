"""MemoryBackend protocol — pluggable storage backend for the Memory Engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

from houyi.adapters.memory.types import MemoryRecord, MemoryScope, MemoryType


class MemoryBackend(ABC):
    """Pluggable storage backend for the Memory Engine.

    Default implementation: SQLiteMemoryBackend (stdlib sqlite3, zero deps).
    Extension points: QMDBackend, RedisBackend, etc.

    All methods are synchronous — SQLite and most local backends are inherently
    synchronous.  Async callers (MemoryEngine, MemoryRetriever) invoke these
    from their own event-loop–safe wrappers.
    """

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def put(self, record: MemoryRecord) -> None:
        """Insert or update a memory record."""

    @abstractmethod
    def get(self, key: str, scope: MemoryScope) -> MemoryRecord | None:
        """Retrieve a single record by (key, scope). Returns None if missing or expired."""

    @abstractmethod
    def list_by_scope(self, scope: MemoryScope) -> list[MemoryRecord]:
        """List non-expired records in *scope*, ordered by updated_at DESC."""

    @abstractmethod
    def list_by_type(
        self,
        memory_type: MemoryType,
        scope: MemoryScope | None = None,
    ) -> list[MemoryRecord]:
        """List non-expired records by memory type, optionally filtered by scope."""

    @abstractmethod
    def all_records(self, *, include_expired: bool = False) -> list[MemoryRecord]:
        """Return all records across all scopes."""

    @abstractmethod
    def delete(self, key: str, scope: MemoryScope) -> bool:
        """Delete a record. Returns True if found and deleted."""

    @abstractmethod
    def clear(self, scope: MemoryScope | None = None) -> int:
        """Clear records. If scope is None, clear all. Returns count deleted."""

    # ------------------------------------------------------------------
    # Full-text search (FTS5)
    # ------------------------------------------------------------------

    @abstractmethod
    def search_fts(
        self,
        query: str,
        scope: MemoryScope | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """BM25 full-text search. Returns (record, bm25_score) pairs ranked by relevance."""

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    @abstractmethod
    def get_embedding(
        self,
        record_id: str,
        provider: str,
        model: str,
    ) -> list[float] | None:
        """Retrieve a cached embedding vector."""

    @abstractmethod
    def put_embedding(
        self,
        record_id: str,
        provider: str,
        model: str,
        embedding: list[float],
    ) -> None:
        """Cache an embedding vector for a record."""

    @abstractmethod
    def search_embedding(
        self,
        query_embedding: list[float],
        scope: MemoryScope | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """Cosine similarity search over cached embeddings.

        Returns (record, similarity) pairs ranked by descending similarity.
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:  # noqa: B027
        """Release resources. Default is a no-op; subclasses override as needed."""
