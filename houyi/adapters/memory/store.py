"""Memory Store: application-layer facade over MemoryBackend.

Manages expiry checks, context rendering, and backward-compatible API.
Storage is delegated to the injected MemoryBackend (default: SQLite).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from houyi.adapters.memory.backends.base import MemoryBackend
from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.types import (
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)

logger = logging.getLogger(__name__)


class MemoryStore:
    """Application-layer facade over a pluggable MemoryBackend.

    External API is identical to the previous JSON-based implementation —
    all existing callers continue to work unchanged.
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        *,
        data_dir: str | Path | None = None,
    ):
        if backend is not None:
            self._backend = backend
        elif data_dir is not None:
            self._backend = SQLiteMemoryBackend(data_dir=data_dir)
        else:
            self._backend = SQLiteMemoryBackend(db_path=":memory:")

    @property
    def backend(self) -> MemoryBackend:
        """Access the underlying backend directly."""
        return self._backend

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def put(
        self,
        key: str,
        content: str,
        scope: MemoryScope = MemoryScope.SESSION,
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
        *,
        memory_type: MemoryType = MemoryType.FACT,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        decay: float = 1.0,
        provenance: MemoryProvenance | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryRecord:
        """Store or update a memory record.

        Returns:
            The created or updated MemoryRecord.
        """
        existing = self._backend.get(key, scope)
        now = time.time()

        if existing is not None:
            existing.content = content
            existing.updated_at = now
            if metadata:
                existing.metadata.update(metadata)
            if ttl is not None:
                existing.ttl = ttl
            existing.memory_type = memory_type
            if tags is not None:
                existing.tags = tags
            existing.confidence = confidence
            existing.decay = decay
            if provenance is not None:
                existing.provenance = provenance
            if embedding is not None:
                existing.embedding = embedding
            self._backend.put(existing)
            return existing

        record = MemoryRecord(
            scope=scope,
            key=key,
            content=content,
            metadata=metadata or {},
            ttl=ttl,
            memory_type=memory_type,
            tags=tags or [],
            confidence=confidence,
            decay=decay,
            provenance=provenance,
            embedding=embedding,
        )
        self._backend.put(record)
        return record

    def put_record(self, record: MemoryRecord) -> MemoryRecord:
        """Store a pre-built MemoryRecord directly.

        Useful for the write pipeline (extractor → classifier → store)
        where a fully populated record is assembled before storage.
        """
        record.updated_at = time.time()
        self._backend.put(record)
        return record

    def get(self, key: str, scope: MemoryScope = MemoryScope.SESSION) -> MemoryRecord | None:
        """Retrieve a memory record by key and scope.

        Returns:
            MemoryRecord if found and not expired, else None.
        """
        return self._backend.get(key, scope)

    def list_by_scope(self, scope: MemoryScope) -> list[MemoryRecord]:
        """List all non-expired records in a scope, ordered by updated_at DESC."""
        return self._backend.list_by_scope(scope)

    def list_by_type(
        self,
        memory_type: MemoryType,
        scope: MemoryScope | None = None,
    ) -> list[MemoryRecord]:
        """List non-expired records by memory type, optionally filtered by scope."""
        return self._backend.list_by_type(memory_type, scope)

    def all_records(self, *, include_expired: bool = False) -> list[MemoryRecord]:
        """Return all records across all scopes."""
        return self._backend.all_records(include_expired=include_expired)

    def delete(self, key: str, scope: MemoryScope = MemoryScope.SESSION) -> bool:
        """Delete a memory record. Returns True if found and deleted."""
        return self._backend.delete(key, scope)

    def clear(self, scope: MemoryScope | None = None) -> int:
        """Clear records. If scope is None, clear all. Returns count deleted."""
        return self._backend.clear(scope)

    # ------------------------------------------------------------------
    # Context rendering
    # ------------------------------------------------------------------

    def as_context_text(self, scope: MemoryScope = MemoryScope.SESSION) -> str:
        """Render all records in a scope as context text for LLM injection."""
        records = self.list_by_scope(scope)
        if not records:
            return ""
        lines = []
        for r in records:
            tag_suffix = f" [{', '.join(r.tags)}]" if r.tags else ""
            type_prefix = f"[{r.memory_type.value}] " if r.memory_type != MemoryType.FACT else ""
            lines.append(f"- {type_prefix}{r.key}: {r.content}{tag_suffix}")
        return "\n".join(lines)
