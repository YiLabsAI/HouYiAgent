"""Content store for sensitive observability data.

Stores large content (prompts, responses, tool inputs/outputs) separately
from span metadata to:
- Keep span storage lean and fast
- Enable content-level access control
- Support content redaction/anonymization
- Allow different retention policies for metadata vs content
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ContentType(str, Enum):
    """Types of content that can be stored."""

    LLM_PROMPT = "llm_prompt"
    LLM_RESPONSE = "llm_response"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    RETRIEVER_QUERY = "retriever_query"
    RETRIEVER_DOCS = "retriever_docs"
    ERROR_TRACE = "error_trace"
    CUSTOM = "custom"


@dataclass
class ContentRef:
    """Reference to stored content."""

    content_id: str
    content_type: ContentType
    span_id: str
    trace_id: str
    size_bytes: int
    created_at: str
    checksum: str


@dataclass
class ContentStoreConfig:
    """Configuration for content store."""

    base_path: str | Path = ".houyi/content"
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    retention_days: int = 30
    compress: bool = True
    hash_algorithm: str = "sha256"


class ContentStore(ABC):
    """Abstract base class for content storage."""

    @abstractmethod
    def store(
        self,
        content: str | bytes | dict[str, Any],
        content_type: ContentType,
        span_id: str,
        trace_id: str,
    ) -> ContentRef:
        """Store content and return a reference."""
        pass

    @abstractmethod
    def retrieve(self, content_id: str) -> str | bytes | None:
        """Retrieve content by ID."""
        pass

    @abstractmethod
    def delete(self, content_id: str) -> bool:
        """Delete content by ID."""
        pass

    @abstractmethod
    def delete_by_trace(self, trace_id: str) -> int:
        """Delete all content for a trace. Returns count deleted."""
        pass

    @abstractmethod
    def cleanup_old_content(self, before_timestamp: float) -> int:
        """Delete content older than timestamp. Returns count deleted."""
        pass

    @abstractmethod
    def get_ref(self, content_id: str) -> ContentRef | None:
        """Get content reference without retrieving content."""
        pass

    @abstractmethod
    def list_refs(self, trace_id: str) -> list[ContentRef]:
        """List all content references for a trace."""
        pass


class FileContentStore(ContentStore):
    """File-based content store.

    Stores content as individual files organized by trace_id.
    Uses a manifest file per trace to track content references.
    """

    def __init__(self, config: ContentStoreConfig | None = None):
        self.config = config or ContentStoreConfig()
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()

    def _compute_hash(self, data: bytes) -> str:
        """Compute hash of content."""
        if self.config.hash_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.config.hash_algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()

    def _get_trace_dir(self, trace_id: str) -> Path:
        """Get directory for a trace."""
        # Use first 2 chars of trace_id for sharding
        shard = trace_id[:2] if len(trace_id) >= 2 else "00"
        return self.base_path / shard / trace_id

    def _get_content_path(self, trace_id: str, content_id: str) -> Path:
        """Get path for content file."""
        return self._get_trace_dir(trace_id) / f"{content_id}.json"

    def _get_manifest_path(self, trace_id: str) -> Path:
        """Get path for trace manifest."""
        return self._get_trace_dir(trace_id) / "manifest.json"

    def _load_manifest(self, trace_id: str) -> dict[str, Any]:
        """Load trace manifest."""
        manifest_path = self._get_manifest_path(trace_id)
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return {
            "trace_id": trace_id,
            "contents": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_manifest(self, trace_id: str, manifest: dict[str, Any]) -> None:
        """Save trace manifest."""
        manifest_path = self._get_manifest_path(trace_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def store(
        self,
        content: str | bytes | dict[str, Any],
        content_type: ContentType,
        span_id: str,
        trace_id: str,
    ) -> ContentRef:
        """Store content and return a reference."""
        # Serialize content
        if isinstance(content, dict):
            content_bytes = json.dumps(content, ensure_ascii=False).encode("utf-8")
        elif isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        # Check size limit
        if len(content_bytes) > self.config.max_content_size:
            raise ValueError(
                f"Content size {len(content_bytes)} exceeds max {self.config.max_content_size}"
            )

        # Generate content ID
        checksum = self._compute_hash(content_bytes)
        content_id = f"{span_id}_{content_type.value}_{checksum[:8]}"

        # Create reference
        now = datetime.now(timezone.utc).isoformat()
        ref = ContentRef(
            content_id=content_id,
            content_type=content_type,
            span_id=span_id,
            trace_id=trace_id,
            size_bytes=len(content_bytes),
            created_at=now,
            checksum=checksum,
        )

        with self._lock:
            # Save content file
            content_path = self._get_content_path(trace_id, content_id)
            content_path.parent.mkdir(parents=True, exist_ok=True)

            content_data = {
                "content_id": content_id,
                "content_type": content_type.value,
                "span_id": span_id,
                "trace_id": trace_id,
                "checksum": checksum,
                "created_at": now,
                "data": content.decode("utf-8") if isinstance(content, bytes) else content,
            }

            with open(content_path, "w", encoding="utf-8") as f:
                json.dump(content_data, f, ensure_ascii=False, indent=2)

            # Update manifest
            manifest = self._load_manifest(trace_id)
            manifest["contents"][content_id] = {
                "content_type": content_type.value,
                "span_id": span_id,
                "size_bytes": len(content_bytes),
                "created_at": now,
                "checksum": checksum,
            }
            self._save_manifest(trace_id, manifest)

        return ref

    def retrieve(self, content_id: str) -> str | bytes | None:
        """Retrieve content by ID."""
        # Need to find the trace_id first
        # Search through shards
        for shard_dir in self.base_path.iterdir():
            if not shard_dir.is_dir():
                continue
            for trace_dir in shard_dir.iterdir():
                if not trace_dir.is_dir():
                    continue
                content_path = trace_dir / f"{content_id}.json"
                if content_path.exists():
                    with open(content_path, encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("data")
        return None

    def retrieve_by_trace(self, trace_id: str, content_id: str) -> str | bytes | None:
        """Retrieve content by trace_id and content_id (faster)."""
        content_path = self._get_content_path(trace_id, content_id)
        if content_path.exists():
            with open(content_path, encoding="utf-8") as f:
                data = json.load(f)
                return data.get("data")
        return None

    def delete(self, content_id: str) -> bool:
        """Delete content by ID."""
        for shard_dir in self.base_path.iterdir():
            if not shard_dir.is_dir():
                continue
            for trace_dir in shard_dir.iterdir():
                if not trace_dir.is_dir():
                    continue
                content_path = trace_dir / f"{content_id}.json"
                if content_path.exists():
                    trace_id = trace_dir.name
                    with self._lock:
                        content_path.unlink()
                        # Update manifest
                        manifest = self._load_manifest(trace_id)
                        if content_id in manifest["contents"]:
                            del manifest["contents"][content_id]
                            self._save_manifest(trace_id, manifest)
                    return True
        return False

    def delete_by_trace(self, trace_id: str) -> int:
        """Delete all content for a trace."""
        trace_dir = self._get_trace_dir(trace_id)
        if not trace_dir.exists():
            return 0

        count = 0
        with self._lock:
            for content_file in trace_dir.glob("*.json"):
                if content_file.name != "manifest.json":
                    content_file.unlink()
                    count += 1

            # Remove manifest and directory
            manifest_path = self._get_manifest_path(trace_id)
            if manifest_path.exists():
                manifest_path.unlink()

            # Try to remove empty directories
            try:
                trace_dir.rmdir()
                trace_dir.parent.rmdir()
            except OSError:
                pass  # Directory not empty

        return count

    def cleanup_old_content(self, before_timestamp: float) -> int:
        """Delete content older than timestamp."""
        count = 0
        before_dt = datetime.fromtimestamp(before_timestamp)

        for shard_dir in self.base_path.iterdir():
            if not shard_dir.is_dir():
                continue
            for trace_dir in shard_dir.iterdir():
                if not trace_dir.is_dir():
                    continue

                manifest = self._load_manifest(trace_dir.name)
                to_delete = []

                for content_id, info in manifest.get("contents", {}).items():
                    created_at = datetime.fromisoformat(info["created_at"])
                    if created_at < before_dt:
                        to_delete.append(content_id)

                for content_id in to_delete:
                    content_path = trace_dir / f"{content_id}.json"
                    if content_path.exists():
                        content_path.unlink()
                        count += 1
                    del manifest["contents"][content_id]

                if to_delete:
                    self._save_manifest(trace_dir.name, manifest)

                # Clean up empty trace directories
                if not manifest.get("contents"):
                    try:
                        manifest_path = self._get_manifest_path(trace_dir.name)
                        if manifest_path.exists():
                            manifest_path.unlink()
                        trace_dir.rmdir()
                    except OSError:
                        pass

        return count

    def get_ref(self, content_id: str) -> ContentRef | None:
        """Get content reference without retrieving content."""
        for shard_dir in self.base_path.iterdir():
            if not shard_dir.is_dir():
                continue
            for trace_dir in shard_dir.iterdir():
                if not trace_dir.is_dir():
                    continue

                manifest = self._load_manifest(trace_dir.name)
                if content_id in manifest.get("contents", {}):
                    info = manifest["contents"][content_id]
                    return ContentRef(
                        content_id=content_id,
                        content_type=ContentType(info["content_type"]),
                        span_id=info["span_id"],
                        trace_id=trace_dir.name,
                        size_bytes=info["size_bytes"],
                        created_at=info["created_at"],
                        checksum=info["checksum"],
                    )
        return None

    def list_refs(self, trace_id: str) -> list[ContentRef]:
        """List all content references for a trace."""
        manifest = self._load_manifest(trace_id)
        refs = []

        for content_id, info in manifest.get("contents", {}).items():
            refs.append(
                ContentRef(
                    content_id=content_id,
                    content_type=ContentType(info["content_type"]),
                    span_id=info["span_id"],
                    trace_id=trace_id,
                    size_bytes=info["size_bytes"],
                    created_at=info["created_at"],
                    checksum=info["checksum"],
                )
            )

        return refs

    def get_statistics(self) -> dict[str, Any]:
        """Get content store statistics."""
        total_files = 0
        total_size = 0
        by_type: dict[str, int] = {}

        for shard_dir in self.base_path.iterdir():
            if not shard_dir.is_dir():
                continue
            for trace_dir in shard_dir.iterdir():
                if not trace_dir.is_dir():
                    continue

                manifest = self._load_manifest(trace_dir.name)
                for _content_id, info in manifest.get("contents", {}).items():
                    total_files += 1
                    total_size += info.get("size_bytes", 0)
                    content_type = info.get("content_type", "unknown")
                    by_type[content_type] = by_type.get(content_type, 0) + 1

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "by_type": by_type,
            "base_path": str(self.base_path),
        }


# Global content store instance
_content_store: ContentStore | None = None


def get_content_store() -> ContentStore:
    """Get global content store instance."""
    global _content_store
    if _content_store is None:
        _content_store = FileContentStore()
    return _content_store


def set_content_store(store: ContentStore) -> None:
    """Set global content store instance."""
    global _content_store
    _content_store = store


def reset_content_store() -> None:
    """Reset global content store instance."""
    global _content_store
    _content_store = None
