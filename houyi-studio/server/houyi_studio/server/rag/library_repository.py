"""Persistent storage and CRUD operations for knowledge libraries.

Responsibilities:
    - Own the _libraries dict (the in-memory representation of
      libraries.json) and the _cancel_flags dict.
    - Provide path helpers scoped to a specific service instance so that
      tests can supply an isolated storage_dir.
    - Load / save / migrate the JSON metadata file.
    - Expose CRUD methods for library management.

Dependencies:
    - .embedding_config for UPLOADS_SUBDIR, INDEX_SUBDIR,
      _LIB_ID_PREFIX, and _default_storage_dir.

Thread Safety:
    Not thread-safe.  Callers must serialise writes externally when
    accessed from multiple async tasks (the current server is
    single-writer).
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .embedding_config import (
    _LIB_ID_PREFIX,
    INDEX_SUBDIR,
    UPLOADS_SUBDIR,
    _default_storage_dir,
)

logger = logging.getLogger(__name__)


class LibraryRepository:
    """Manages knowledge-library persistence and CRUD.

    Args:
        storage_dir: Override the storage root.  When None, falls back to
            the HOUYI_KNOWLEDGE_STORAGE env var or .houyi/knowledge.
            **Always** supply an explicit path in tests.
    """

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        if storage_dir is not None:
            self._storage_dir = Path(storage_dir).resolve()
        else:
            self._storage_dir = _default_storage_dir()
        self._libraries: dict[str, dict[str, Any]] = {}
        self._cancel_flags: dict[str, bool] = {}
        self._ensure_storage_dir()
        self._load_libraries()

    # ── Storage path helpers ──────────────────────────────────

    @property
    def storage_dir(self) -> Path:
        """The root storage directory for this repository instance."""
        return self._storage_dir

    def library_storage_dir(self, library_id: str) -> Path:
        """Return the root directory for *library_id*."""
        return self._storage_dir / library_id

    def library_upload_dir(self, library_id: str) -> Path:
        """Return the uploads sub-directory for *library_id*."""
        return self.library_storage_dir(library_id) / UPLOADS_SUBDIR

    def library_index_dir(self, library_id: str) -> Path:
        """Return the index sub-directory for *library_id*."""
        return self.library_storage_dir(library_id) / INDEX_SUBDIR

    # ── Internal persistence ──────────────────────────────────

    def _ensure_storage_dir(self) -> None:
        """Create the storage directory tree if it does not exist."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _load_libraries(self) -> None:
        """Deserialise libraries.json into _libraries."""
        metadata_file = self._storage_dir / "libraries.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    self._libraries = json.load(f)
                logger.debug("Loaded %d knowledge libraries", len(self._libraries))
                self._migrate_library_data()
            except Exception as e:
                logger.warning("Failed to load knowledge libraries: %s", e)
                self._libraries = {}

    def _migrate_library_data(self) -> None:
        """Ensure consistency of legacy library data on disk."""
        needs_save = False
        for lib_id, lib in self._libraries.items():
            if "documents" not in lib:
                lib["documents"] = {}
                needs_save = True

            documents = lib.get("documents", {})
            doc_count = len(documents)
            indexed_count = sum(1 for d in documents.values() if d.get("status") == "indexed")
            error_count = sum(1 for d in documents.values() if d.get("status") == "error")

            total_chunks = sum(d.get("chunk_count", 0) for d in documents.values())
            old_status = lib.get("status")
            if doc_count == 0:
                new_status = "empty"
            elif error_count == doc_count:
                new_status = "error"
            elif indexed_count < doc_count and error_count > 0:
                new_status = "partial"
            elif total_chunks == 0 and indexed_count > 0:
                new_status = "degraded"
            else:
                new_status = "ready"

            if old_status != new_status:
                lib["status"] = new_status
                needs_save = True
                logger.debug(
                    "Migrated library %s status: %s -> %s",
                    lib_id,
                    old_status,
                    new_status,
                )

            if lib.get("doc_count") != doc_count:
                lib["doc_count"] = doc_count
                needs_save = True
            if lib.get("chunk_count") != total_chunks:
                lib["chunk_count"] = total_chunks
                needs_save = True

            seen_paths: set[str] = set()
            docs_to_remove: list[str] = []
            for doc_id, doc in documents.items():
                file_path = doc.get("file_path")
                if file_path in seen_paths:
                    docs_to_remove.append(doc_id)
                else:
                    seen_paths.add(file_path)

            for doc_id in docs_to_remove:
                del documents[doc_id]
                needs_save = True
                logger.debug("Removed duplicate document %s", doc_id)

            if docs_to_remove:
                lib["doc_count"] = len(documents)

        if needs_save:
            self._save_libraries()
            logger.debug("Library data migration complete")

    def _save_libraries(self) -> None:
        """Persist _libraries to libraries.json."""
        metadata_file = self._storage_dir / "libraries.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(self._libraries, f, indent=2)
        except Exception as e:
            logger.error("Failed to save knowledge libraries: %s", e)

    # ── Public helpers for save (used by other services) ──────

    def save(self) -> None:
        """Flush current state to disk (convenience wrapper)."""
        self._save_libraries()

    # ── CRUD ──────────────────────────────────────────────────

    def list_libraries(self) -> list[dict[str, Any]]:
        """Return metadata for every known library.

        Returns:
            A list of library dicts (order is arbitrary).
        """
        return list(self._libraries.values())

    def create_library(
        self,
        name: str,
        description: str = "",
        mode: str = "auto",
        knowledge_dir: str = "./knowledge",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create a new knowledge library.

        Args:
            name: Human-readable library name.
            description: Optional description.
            mode: RAG mode (agentic, indexed, auto).
            knowledge_dir: Path to the knowledge source directory.
            metadata: Extra metadata (chunking settings, etc.).

        Returns:
            The created library dict, or None if *name* is a duplicate.
        """
        for lib in self._libraries.values():
            if lib.get("name") == name:
                logger.warning("Library with name '%s' already exists", name)
                return None

        library_id = f"lib_{uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        library: dict[str, Any] = {
            "library_id": library_id,
            "name": name,
            "description": description,
            "mode": mode,
            "knowledge_dir": knowledge_dir,
            "created_at": now,
            "updated_at": now,
            "doc_count": 0,
            "chunk_count": 0,
            "status": "empty",
            "documents": {},
            "metadata": metadata or {},
        }

        dir_path = Path(knowledge_dir)
        if dir_path.exists():
            doc_count = sum(
                1
                for f in dir_path.rglob("*")
                if f.is_file() and f.suffix in [".md", ".txt", ".pdf", ".json", ".csv"]
            )
            library["doc_count"] = doc_count

        storage_dir = self.library_storage_dir(library_id)
        self.library_upload_dir(library_id).mkdir(parents=True, exist_ok=True)
        self.library_index_dir(library_id).mkdir(parents=True, exist_ok=True)
        logger.debug("Created library storage at: %s", storage_dir)

        self._libraries[library_id] = library
        self._save_libraries()

        logger.info("Created knowledge library: %s (%s)", name, library_id)
        return library

    def delete_library(self, library_id: str) -> bool:
        """Delete a library and all its on-disk data.

        Safety:
            Validates the *library_id* format and ensures the resolved path
            is a strict child of storage_dir to prevent path-traversal.

        Args:
            library_id: The library to remove.

        Returns:
            True if the library was deleted, False otherwise.
        """
        if library_id not in self._libraries:
            return False

        if not library_id.startswith(_LIB_ID_PREFIX):
            logger.error(
                "Refusing to delete library with unexpected ID format: %s",
                library_id,
            )
            return False

        storage_dir = self.library_storage_dir(library_id)
        resolved = storage_dir.resolve()

        try:
            resolved.relative_to(self._storage_dir.resolve())
        except ValueError:
            logger.error(
                "SECURITY: Refusing to delete path outside storage root. "
                "target=%s, storage_root=%s",
                resolved,
                self._storage_dir.resolve(),
            )
            return False

        if resolved.parent != self._storage_dir.resolve():
            logger.error(
                "SECURITY: Refusing to delete nested path. target=%s, expected_parent=%s",
                resolved,
                self._storage_dir.resolve(),
            )
            return False

        if storage_dir.exists():
            try:
                shutil.rmtree(storage_dir)
                logger.debug("Deleted library storage: %s", storage_dir)
            except Exception as e:
                logger.warning("Failed to delete library storage %s: %s", storage_dir, e)

        del self._libraries[library_id]
        self._save_libraries()

        logger.info("Deleted knowledge library: %s", library_id)
        return True

    def get_library(self, library_id: str) -> dict[str, Any] | None:
        """Look up a library by its ID.

        Args:
            library_id: The identifier to search for.

        Returns:
            The library dict, or None if not found.
        """
        return self._libraries.get(library_id)

    def update_library(
        self,
        library_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Apply a partial update to a library.

        Args:
            library_id: The library to update.
            updates: Key/value pairs to merge.  The metadata key is
                *merged* rather than replaced.

        Returns:
            The updated library dict, or None if not found.
        """
        library = self._libraries.get(library_id)
        if not library:
            return None

        for key in [
            "name",
            "description",
            "mode",
            "doc_count",
            "chunk_count",
            "status",
            "metadata",
        ]:
            if key in updates:
                if key == "metadata":
                    old_meta = library.get("metadata", {})
                    library["metadata"] = {**old_meta, **updates[key]}
                else:
                    library[key] = updates[key]

        library["updated_at"] = datetime.now().isoformat()
        self._save_libraries()

        logger.info(
            "Updated knowledge library: %s with keys: %s",
            library_id,
            list(updates.keys()),
        )
        return library

    # ── Cancel flag helpers ───────────────────────────────────

    def cancel_ingest(self, library_id: str) -> None:
        """Signal that an in-progress ingest should be cancelled.

        Args:
            library_id: The library whose ingest should stop.
        """
        self._cancel_flags[library_id] = True
        logger.info("Cancel requested for library: %s", library_id)

    @property
    def cancel_flags(self) -> dict[str, bool]:
        """Direct access to the cancel-flags dict (shared with services)."""
        return self._cancel_flags
