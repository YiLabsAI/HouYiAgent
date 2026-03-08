"""Memory Store: simple KV storage for Phase 1.

Provides in-memory storage with optional JSON file persistence.
Phase 1: Manual read/write, no automatic extraction or retrieval scoring.
Phase 2: Will be extended with embedding-based retrieval and scoring.

"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from houyi.adapters.memory.types import MemoryRecord, MemoryScope

logger = logging.getLogger(__name__)


class MemoryStore:
    """Simple key-value memory store with optional file persistence.

    Thread-safety: Not thread-safe. For Phase 1 single-user scenarios only.
    Phase 2 will add proper locking for concurrent access.
    """

    def __init__(self, data_dir: str | Path | None = None):
        """Initialize memory store.

        Args:
            data_dir: Directory for JSON persistence. If None, memory is
                      in-memory only (lost on restart).
        """
        self._records: dict[str, MemoryRecord] = {}
        self._data_dir = Path(data_dir) if data_dir else None
        if self._data_dir:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def put(
        self,
        key: str,
        content: str,
        scope: MemoryScope = MemoryScope.SESSION,
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> MemoryRecord:
        """Store or update a memory record.

        Args:
            key: Unique key within scope.
            content: Memory content text.
            scope: Memory scope (session/user/workspace).
            metadata: Optional metadata dict.
            ttl: Time-to-live in seconds (None = no expiry).

        Returns:
            The created or updated MemoryRecord.
        """
        store_key = f"{scope.value}:{key}"
        now = time.time()

        if store_key in self._records:
            record = self._records[store_key]
            record.content = content
            record.updated_at = now
            if metadata:
                record.metadata.update(metadata)
            if ttl is not None:
                record.ttl = ttl
        else:
            record = MemoryRecord(
                scope=scope,
                key=key,
                content=content,
                metadata=metadata or {},
                ttl=ttl,
            )
            self._records[store_key] = record

        self._persist_record(record)
        return record

    def get(self, key: str, scope: MemoryScope = MemoryScope.SESSION) -> MemoryRecord | None:
        """Retrieve a memory record by key and scope.

        Args:
            key: Record key.
            scope: Memory scope.

        Returns:
            MemoryRecord if found and not expired, else None.
        """
        store_key = f"{scope.value}:{key}"
        record = self._records.get(store_key)
        if record is None:
            return None
        if record.is_expired:
            self.delete(key, scope)
            return None
        return record

    def list_by_scope(self, scope: MemoryScope) -> list[MemoryRecord]:
        """List all non-expired records in a scope.

        Args:
            scope: Memory scope to filter by.

        Returns:
            List of MemoryRecords, ordered by updated_at descending.
        """
        records = []
        expired_keys = []
        for store_key, record in self._records.items():
            if record.scope != scope:
                continue
            if record.is_expired:
                expired_keys.append(store_key)
                continue
            records.append(record)

        # Clean up expired
        for k in expired_keys:
            del self._records[k]

        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records

    def delete(self, key: str, scope: MemoryScope = MemoryScope.SESSION) -> bool:
        """Delete a memory record.

        Args:
            key: Record key.
            scope: Memory scope.

        Returns:
            True if record was found and deleted.
        """
        store_key = f"{scope.value}:{key}"
        if store_key in self._records:
            record = self._records.pop(store_key)
            self._delete_from_disk(record)
            return True
        return False

    def clear(self, scope: MemoryScope | None = None) -> int:
        """Clear memory records.

        Args:
            scope: If provided, only clear records in this scope.
                   If None, clear all records.

        Returns:
            Number of records deleted.
        """
        if scope is None:
            count = len(self._records)
            self._records.clear()
            if self._data_dir:
                for f in self._data_dir.glob("*.json"):
                    f.unlink(missing_ok=True)
            return count

        to_delete = [k for k, r in self._records.items() if r.scope == scope]
        for k in to_delete:
            record = self._records.pop(k)
            self._delete_from_disk(record)
        return len(to_delete)

    def as_context_text(self, scope: MemoryScope = MemoryScope.SESSION) -> str:
        """Render all records in a scope as context text for injection.

        Args:
            scope: Memory scope.

        Returns:
            Formatted text suitable for ContextPlanner memory injection.
        """
        records = self.list_by_scope(scope)
        if not records:
            return ""
        lines = [f"- {r.key}: {r.content}" for r in records]
        return "\n".join(lines)

    # --- Persistence helpers ---

    def _persist_record(self, record: MemoryRecord) -> None:
        """Write a single record to disk."""
        if not self._data_dir:
            return
        scope_dir = self._data_dir / record.scope.value
        scope_dir.mkdir(parents=True, exist_ok=True)
        file_path = scope_dir / f"{record.record_id}.json"
        try:
            data = record.model_dump(mode="json")
            tmp_path = file_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            tmp_path.rename(file_path)
        except Exception as e:
            logger.error("Failed to persist memory record %s: %s", record.record_id, e)

    def _delete_from_disk(self, record: MemoryRecord) -> None:
        """Delete a record file from disk."""
        if not self._data_dir:
            return
        file_path = self._data_dir / record.scope.value / f"{record.record_id}.json"
        file_path.unlink(missing_ok=True)

    def _load_from_disk(self) -> None:
        """Load all records from disk on startup."""
        if not self._data_dir:
            return
        count = 0
        for scope_dir in self._data_dir.iterdir():
            if not scope_dir.is_dir():
                continue
            for file_path in scope_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text())
                    record = MemoryRecord(**data)
                    if record.is_expired:
                        file_path.unlink(missing_ok=True)
                        continue
                    store_key = f"{record.scope.value}:{record.key}"
                    self._records[store_key] = record
                    count += 1
                except Exception as e:
                    logger.warning("Failed to load memory file %s: %s", file_path, e)
        if count > 0:
            logger.info("Loaded %d memory records from %s", count, self._data_dir)
