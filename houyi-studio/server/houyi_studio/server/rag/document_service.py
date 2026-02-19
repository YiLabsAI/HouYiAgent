"""Document and chunk management for knowledge libraries.

Responsibilities:
    - CRUD operations on documents within a library.
    - Chunk listing and preview (simple splitting for the UI).
    - Status transitions (enable / disable / error).

Dependencies:
    - :class:`~.library_repository.LibraryRepository` (constructor-injected).

Thread Safety:
    Delegates all persistence to the repository; same caveats apply.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .library_repository import LibraryRepository

logger = logging.getLogger(__name__)


class DocumentService:
    """Manages documents and chunks inside knowledge libraries.

    Args:
        repo: The shared :class:`LibraryRepository` instance.
    """

    def __init__(self, repo: LibraryRepository) -> None:
        self._repo = repo

    # ── Document CRUD ─────────────────────────────────────────

    def list_documents(self, library_id: str) -> list[dict[str, Any]]:
        """List all documents in a library.

        Args:
            library_id: Target library.

        Returns:
            A list of document metadata dicts (empty if the library is
            unknown).
        """
        library = self._repo.get_library(library_id)
        if not library:
            return []
        return list(library.get("documents", {}).values())

    def get_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a single document by ID.

        Args:
            library_id: Owning library.
            doc_id: Document identifier.

        Returns:
            The document dict, or ``None``.
        """
        library = self._repo.get_library(library_id)
        if not library:
            return None
        return library.get("documents", {}).get(doc_id)

    def add_document(
        self,
        library_id: str,
        file_path: str,
        status: str = "pending",
    ) -> dict[str, Any] | None:
        """Register a new document in a library.

        Args:
            library_id: Target library.
            file_path: Absolute or relative path to the file on disk.
            status: Initial status (``pending`` / ``indexing`` / ``indexed``
                / ``disabled`` / ``error``).

        Returns:
            The created document dict, or ``None`` if the library does
            not exist or the file is missing.
        """
        library = self._repo.get_library(library_id)
        if not library:
            return None

        path = Path(file_path)
        if not path.exists():
            return None

        doc_id = f"doc_{uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        stat = path.stat()
        document: dict[str, Any] = {
            "doc_id": doc_id,
            "library_id": library_id,
            "filename": path.name,
            "file_path": str(path.resolve()),
            "file_size": stat.st_size,
            "file_type": path.suffix.lower(),
            "status": status,
            "chunk_count": 0,
            "retrieval_count": 0,
            "created_at": now,
            "updated_at": now,
        }

        if "documents" not in library:
            library["documents"] = {}

        library["documents"][doc_id] = document
        self._repo.save()

        logger.debug("Added document %s to library %s", path.name, library_id)
        return document

    def update_document_status(
        self,
        library_id: str,
        doc_id: str,
        status: str,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any] | None:
        """Transition a document to a new status.

        Args:
            library_id: Owning library.
            doc_id: Document identifier.
            status: New status string.
            chunk_count: Optionally update the chunk count.
            error_message: Attach an error message (cleared automatically
                when the new status is not ``error``).

        Returns:
            The updated document dict, or ``None`` if not found.
        """
        library = self._repo.get_library(library_id)
        if not library:
            return None

        documents = library.get("documents", {})
        document = documents.get(doc_id)
        if not document:
            return None

        document["status"] = status
        document["updated_at"] = datetime.now().isoformat()

        if chunk_count is not None:
            document["chunk_count"] = chunk_count

        if error_message is not None:
            document["error_message"] = error_message
        elif "error_message" in document and status != "error":
            del document["error_message"]

        self._repo.save()
        logger.debug("Updated document %s status to %s", doc_id, status)
        return document

    def delete_document(self, library_id: str, doc_id: str) -> bool:
        """Remove a document from its library and recalculate stats.

        Args:
            library_id: Owning library.
            doc_id: Document to delete.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        library = self._repo.get_library(library_id)
        if not library:
            return False

        documents = library.get("documents", {})
        if doc_id not in documents:
            return False

        del documents[doc_id]

        doc_count = len(documents)
        library["doc_count"] = doc_count
        library["chunk_count"] = sum(d.get("chunk_count", 0) for d in documents.values())

        if doc_count == 0:
            library["status"] = "empty"
        else:
            indexed_count = sum(1 for d in documents.values() if d.get("status") == "indexed")
            error_count = sum(1 for d in documents.values() if d.get("status") == "error")
            if error_count == doc_count:
                library["status"] = "error"
            elif indexed_count < doc_count and error_count > 0:
                library["status"] = "partial"
            else:
                library["status"] = "ready"

        library["updated_at"] = datetime.now().isoformat()
        self._repo.save()

        logger.debug("Deleted document %s from library %s", doc_id, library_id)
        return True

    def disable_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Disable a document (exclude from retrieval).

        Args:
            library_id: Owning library.
            doc_id: Document to disable.

        Returns:
            The updated document dict, or ``None``.
        """
        return self.update_document_status(library_id, doc_id, "disabled")

    def enable_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Re-enable a previously disabled document.

        Args:
            library_id: Owning library.
            doc_id: Document to enable.

        Returns:
            The updated document dict, or ``None``.
        """
        return self.update_document_status(library_id, doc_id, "indexed")

    def increment_retrieval_count(self, library_id: str, doc_id: str) -> None:
        """Bump the retrieval counter for a document.

        The change is kept in memory only; a periodic ``repo.save()``
        will persist it (avoids hot-path disk I/O).

        Args:
            library_id: Owning library.
            doc_id: Document whose counter should increment.
        """
        library = self._repo.get_library(library_id)
        if not library:
            return

        documents = library.get("documents", {})
        document = documents.get(doc_id)
        if document:
            document["retrieval_count"] = document.get("retrieval_count", 0) + 1

    # ── Chunk helpers ─────────────────────────────────────────

    def list_chunks(self, library_id: str, doc_id: str) -> list[dict[str, Any]]:
        """Return stored chunks for a document.

        Args:
            library_id: Owning library.
            doc_id: Document identifier.

        Returns:
            A (possibly empty) list of chunk dicts.
        """
        library = self._repo.get_library(library_id)
        if not library:
            return []

        documents = library.get("documents", {})
        document = documents.get(doc_id)
        if not document:
            return []

        return document.get("chunks", [])

    def preview_chunks(
        self,
        content: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "recursive",
    ) -> list[dict[str, Any]]:
        """Preview how *content* would be split into chunks.

        This is a lightweight, in-memory operation for the UI — no
        persistence is involved.

        Args:
            content: Raw text to chunk.
            chunk_size: Target size per chunk (characters).
            chunk_overlap: Overlap between consecutive chunks.
            strategy: ``"recursive"`` (default) or ``"sentence"``.

        Returns:
            A list of dicts with ``index``, ``content`` (truncated),
            and ``char_count``.
        """
        chunks: list[str] = []
        if strategy == "sentence":
            sentences = re.split(r"(?<=[.!?])\s+", content)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) <= chunk_size:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sentence
            if current:
                chunks.append(current)
        else:
            pos = 0
            while pos < len(content):
                end = min(pos + chunk_size, len(content))
                chunk = content[pos:end]
                chunks.append(chunk)
                pos = end - chunk_overlap if end < len(content) else end

        return [
            {
                "index": i,
                "content": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "char_count": len(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]
