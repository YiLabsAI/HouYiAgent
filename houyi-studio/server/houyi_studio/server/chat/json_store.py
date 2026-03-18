"""JSON Store: file-based persistence for chat conversations.

Each conversation is stored as a single JSON file with atomic writes.
An index file maintains the conversation list for fast listing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from .types import Conversation

logger = logging.getLogger(__name__)

# Default data directory (relative to project root)
_DEFAULT_DATA_DIR = "data/conversations"


def _project_root() -> Path:
    current = Path(__file__).resolve()
    git_root: Path | None = None
    for parent in current.parents:
        if (parent / ".git").exists():
            git_root = parent
    if git_root is not None:
        return git_root
    return Path.cwd()


def resolve_chat_data_dir(data_dir: str | Path | None = None) -> Path:
    raw_path = Path(data_dir) if data_dir else Path(_DEFAULT_DATA_DIR)
    if raw_path.is_absolute():
        return raw_path
    return _project_root() / raw_path


class JsonStore:
    """File-based JSON store for chat conversations.

    Guarantees:
    - Atomic writes (write to .tmp then rename)
    - Schema versioning (forward compatible)
    - Per-conversation locking for concurrent write safety

    Concurrency: Uses asyncio.Lock per conversation_id to serialize
    read-modify-write cycles. Safe for concurrent async callers.
    """

    def __init__(self, data_dir: str | Path | None = None):
        """Initialize JSON store.

        Args:
            data_dir: Directory for conversation JSON files.
                      Defaults to 'data/conversations' relative to project root.
        """
        self._data_dir = resolve_chat_data_dir(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._backup_dir = self._data_dir / "_backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._load_index()

    def create(self, conversation: Conversation) -> Conversation:
        """Create a new conversation.

        Args:
            conversation: Conversation to persist.

        Returns:
            The persisted Conversation.

        Raises:
            ValueError: If conversation_id already exists.
        """
        if self._file_path(conversation.conversation_id).exists():
            raise ValueError(f"Conversation {conversation.conversation_id} already exists")

        self._write_conversation(conversation)
        self._update_index(conversation)
        logger.info("Created conversation %s: %s", conversation.conversation_id, conversation.title)
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Conversation if found, else None.
        """
        file_path = self._file_path(conversation_id)
        if not file_path.exists():
            return None
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return Conversation(**data)
        except Exception as e:
            logger.error("Failed to read conversation %s: %s", conversation_id, e)
            return None

    def update(self, conversation: Conversation) -> Conversation:
        """Update an existing conversation (full replace).

        Args:
            conversation: Updated conversation.

        Returns:
            The updated Conversation.

        Raises:
            FileNotFoundError: If conversation does not exist.
        """
        if not self._file_path(conversation.conversation_id).exists():
            raise FileNotFoundError(f"Conversation {conversation.conversation_id} not found")

        conversation.updated_at = time.time()
        self._write_conversation(conversation)
        self._update_index(conversation)
        return conversation

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: Conversation to delete.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self._file_path(conversation_id)
        if not file_path.exists():
            return False

        file_path.unlink()
        self._index.pop(conversation_id, None)
        self._write_index()
        logger.info("Deleted conversation %s", conversation_id)
        return True

    def list_conversations(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List conversation summaries.

        Args:
            status: Filter by status (active/archived). None = all.
            limit: Max results.
            offset: Pagination offset.

        Returns:
            List of conversation summary dicts, sorted by updated_at desc.
        """
        self._refresh_index_from_conversation_files()
        summaries = list(self._index.values())

        if status:
            summaries = [s for s in summaries if s.get("status") == status]

        # Sort by updated_at descending
        summaries.sort(key=lambda s: s.get("updated_at", 0), reverse=True)

        return summaries[offset : offset + limit]

    def count(self, status: str | None = None) -> int:
        """Count conversations.

        Args:
            status: Filter by status. None = all.

        Returns:
            Number of conversations.
        """
        if status is None:
            return len(self._index)
        return sum(1 for s in self._index.values() if s.get("status") == status)

    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Full-text search across conversation titles and message content.

        Phase 1 implementation: simple case-insensitive substring match
        across all conversations. Sufficient for small-to-medium scale.

        Args:
            query: Search keyword (case-insensitive).
            limit: Max results to return.

        Returns:
            List of search result dicts with conversation_id, title,
            message_id, role, content snippet, and created_at.
        """
        if not query.strip():
            return []

        q = query.lower()
        results: list[dict[str, Any]] = []

        for conv_id in list(self._index.keys()):
            conv = self.get(conv_id)
            if conv is None:
                continue

            # Search title
            if q in conv.title.lower():
                results.append(
                    {
                        "conversation_id": conv.conversation_id,
                        "title": conv.title,
                        "match_type": "title",
                        "message_id": None,
                        "role": None,
                        "snippet": conv.title,
                        "created_at": conv.created_at,
                        "updated_at": conv.updated_at,
                    }
                )

            # Search messages
            for msg in conv.messages:
                if q in msg.content.lower():
                    # Extract snippet around the match
                    idx = msg.content.lower().index(q)
                    start = max(0, idx - 40)
                    end = min(len(msg.content), idx + len(query) + 40)
                    snippet = (
                        ("..." if start > 0 else "")
                        + msg.content[start:end]
                        + ("..." if end < len(msg.content) else "")
                    )

                    results.append(
                        {
                            "conversation_id": conv.conversation_id,
                            "title": conv.title,
                            "match_type": "message",
                            "message_id": msg.message_id,
                            "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                            "snippet": snippet,
                            "bookmarked": msg.bookmarked,
                            "created_at": msg.created_at,
                            "updated_at": conv.updated_at,
                        }
                    )

            if len(results) >= limit:
                break

        return results[:limit]

    def create_backup(
        self,
        conversation_id: str,
        *,
        trigger: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        file_path = self._file_path(conversation_id)
        if not file_path.exists():
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        backup_id = uuid.uuid4().hex[:12]
        created_at = time.time()
        backup_filename = f"{conversation_id}--{backup_id}.json"
        backup_path = self._backup_dir / backup_filename
        tmp_path = backup_path.with_suffix(".tmp")
        payload = file_path.read_text(encoding="utf-8")
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.rename(backup_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        entry = {
            "backup_id": backup_id,
            "conversation_id": conversation_id,
            "trigger": trigger,
            "created_at": created_at,
            "path": backup_filename,
            "record_id": None,
            "metadata": dict(metadata or {}),
        }
        index = self._load_backup_index()
        index[backup_id] = entry
        self._write_backup_index(index)
        return dict(entry)

    def get_backup(self, backup_id: str) -> dict[str, Any] | None:
        entry = self._load_backup_index().get(backup_id)
        if not isinstance(entry, dict):
            return None
        return dict(entry)

    def list_backups(self, conversation_id: str | None = None) -> list[dict[str, Any]]:
        backups = list(self._load_backup_index().values())
        if conversation_id is not None:
            backups = [
                item
                for item in backups
                if isinstance(item, dict) and item.get("conversation_id") == conversation_id
            ]
        return [dict(item) for item in backups]

    def read_backup(self, backup_id: str) -> Conversation:
        entry = self._load_backup_index().get(backup_id)
        if not isinstance(entry, dict):
            raise FileNotFoundError(f"Backup {backup_id} not found")
        backup_path = self._backup_dir / str(entry.get("path") or "")
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file missing for {backup_id}")
        return Conversation(**json.loads(backup_path.read_text(encoding="utf-8")))

    def attach_backup_record(self, backup_id: str, *, record_id: str) -> dict[str, Any] | None:
        index = self._load_backup_index()
        entry = index.get(backup_id)
        if not isinstance(entry, dict):
            return None
        updated = dict(entry)
        updated["record_id"] = record_id
        index[backup_id] = updated
        self._write_backup_index(index)
        return dict(updated)

    def restore_backup(self, backup_id: str) -> Conversation:
        conversation = self.read_backup(backup_id)
        conversation.updated_at = time.time()
        self._write_conversation(conversation)
        self._update_index(conversation)
        return conversation

    def get_bookmarks(self) -> list[dict[str, Any]]:
        """Get all bookmarked conversations and messages.

        Returns a flat list of bookmark entries, each containing:
        - type: "conversation" or "message"
        - conversation_id, title, updated_at
        - For messages: message_id, role, snippet, created_at

        Sorted by created_at/updated_at descending.
        """
        results: list[dict[str, Any]] = []

        for conv_id in list(self._index.keys()):
            conv = self.get(conv_id)
            if conv is None:
                continue

            # Bookmarked conversation
            if conv.bookmarked:
                results.append(
                    {
                        "type": "conversation",
                        "conversation_id": conv.conversation_id,
                        "title": conv.title,
                        "message_count": len(conv.messages),
                        "model": conv.model,
                        "created_at": conv.created_at,
                        "updated_at": conv.updated_at,
                    }
                )

            # Bookmarked messages
            for msg in conv.messages:
                if msg.bookmarked:
                    snippet = msg.content[:120] + ("..." if len(msg.content) > 120 else "")
                    results.append(
                        {
                            "type": "message",
                            "conversation_id": conv.conversation_id,
                            "title": conv.title,
                            "message_id": msg.message_id,
                            "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                            "snippet": snippet,
                            "created_at": msg.created_at,
                            "updated_at": conv.updated_at,
                        }
                    )

        # Sort by timestamp descending
        results.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return results

    async def lock(self, conversation_id: str) -> asyncio.Lock:
        """Get or create a per-conversation lock.

        Callers should use this to wrap read-modify-write cycles::

            async with store.lock(conv_id):
                conv = store.get(conv_id)
                conv.messages.append(msg)
                store.update(conv)
        """
        async with self._global_lock:
            if conversation_id not in self._locks:
                self._locks[conversation_id] = asyncio.Lock()
            return self._locks[conversation_id]

    # --- Internal helpers ---

    def _file_path(self, conversation_id: str) -> Path:
        """Get file path for a conversation."""
        return self._data_dir / f"{conversation_id}.json"

    def _backup_index_path(self) -> Path:
        return self._backup_dir / "index.json"

    def _write_conversation(self, conversation: Conversation) -> None:
        """Atomic write: write to .tmp then rename."""
        file_path = self._file_path(conversation.conversation_id)
        tmp_path = file_path.with_suffix(".tmp")
        try:
            data = conversation.model_dump(mode="json")
            tmp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.rename(file_path)
        except Exception as e:
            logger.error("Failed to write conversation %s: %s", conversation.conversation_id, e)
            tmp_path.unlink(missing_ok=True)
            raise

    def _update_index(self, conversation: Conversation) -> None:
        """Update in-memory index and persist."""
        self._index[conversation.conversation_id] = conversation.to_summary()
        self._write_index()

    def _refresh_index_from_conversation_files(self) -> None:
        """Recompute summaries from conversation files to avoid stale index counts."""
        refreshed: dict[str, dict[str, Any]] = {}
        changed = False
        for file_path in self._data_dir.glob("*.json"):
            if file_path.name == "index.json":
                continue
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                conv = Conversation(**data)
                summary = conv.to_summary()
                refreshed[conv.conversation_id] = summary
                if self._index.get(conv.conversation_id) != summary:
                    changed = True
            except Exception as e:
                logger.warning("Failed to refresh summary from %s: %s", file_path.name, e)

        stale_ids = set(self._index.keys()) - set(refreshed.keys())
        if stale_ids:
            changed = True

        if changed or len(refreshed) != len(self._index):
            self._index = refreshed
            self._write_index()

    def _write_index(self) -> None:
        """Write index file atomically."""
        index_path = self._data_dir / "index.json"
        tmp_path = index_path.with_suffix(".tmp")
        try:
            data = {
                "conversations": list(self._index.values()),
                "updated_at": time.time(),
            }
            tmp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.rename(index_path)
        except Exception as e:
            logger.error("Failed to write index: %s", e)
            tmp_path.unlink(missing_ok=True)

    def _load_backup_index(self) -> dict[str, dict[str, Any]]:
        index_path = self._backup_index_path()
        if not index_path.exists():
            return {}
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read backup index: %s", exc)
            return {}
        backups = data.get("backups")
        if not isinstance(backups, list):
            return {}
        result: dict[str, dict[str, Any]] = {}
        for item in backups:
            if not isinstance(item, dict):
                continue
            backup_id = item.get("backup_id")
            if isinstance(backup_id, str) and backup_id:
                result[backup_id] = item
        return result

    def _write_backup_index(self, index: dict[str, dict[str, Any]]) -> None:
        index_path = self._backup_index_path()
        tmp_path = index_path.with_suffix(".tmp")
        payload = {
            "backups": sorted(
                index.values(),
                key=lambda item: float(item.get("created_at", 0) or 0),
                reverse=True,
            ),
            "updated_at": time.time(),
        }
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.rename(index_path)
        except Exception as exc:
            logger.error("Failed to write backup index: %s", exc)
            tmp_path.unlink(missing_ok=True)

    def _load_index(self) -> None:
        """Load index from disk, or rebuild from conversation files."""
        index_path = self._data_dir / "index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                self._index.clear()
                for summary in data.get("conversations", []):
                    cid = summary.get("conversation_id")
                    if cid:
                        self._index[cid] = summary
                # If index is empty but conversation files exist, rebuild
                json_files = [f for f in self._data_dir.glob("*.json") if f.name != "index.json"]
                if not self._index and json_files:
                    logger.warning(
                        "Index is empty but %d conversation files exist, rebuilding",
                        len(json_files),
                    )
                else:
                    logger.info("Loaded %d conversations from index", len(self._index))
                    self._refresh_index_from_conversation_files()
                    return
            except Exception as e:
                logger.warning("Failed to load index, rebuilding: %s", e)

        # Rebuild index from conversation files
        count = 0
        for file_path in self._data_dir.glob("*.json"):
            if file_path.name == "index.json":
                continue
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                conv = Conversation(**data)
                self._index[conv.conversation_id] = conv.to_summary()
                count += 1
            except Exception as e:
                logger.warning("Failed to load %s: %s", file_path.name, e)

        self._write_index()

        if count > 0:
            self._write_index()
            logger.info("Rebuilt index from %d conversation files", count)
