"""Facade that assembles the RAG sub-services into a single public API.

Responsibilities:
    - Instantiate :class:`LibraryRepository`, :class:`DocumentService`,
      :class:`IngestService`, and :class:`SearchService`.
    - Delegate every public method to the appropriate sub-service so that
      existing callers can continue using a single ``KnowledgeService``
      object.
    - Host the ``get_knowledge_service()`` singleton.

Dependencies:
    - All sibling modules in the ``rag`` package.

Thread Safety:
    Same constraints as the underlying services — single-writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .document_service import DocumentService
from .ingest_service import IngestService
from .library_repository import LibraryRepository
from .search_service import SearchService


class KnowledgeService:
    """Unified facade over the RAG sub-services.

    This class preserves the original ``KnowledgeService`` API so that
    existing call-sites do not need to change.  Internally every method
    is a thin delegation to the focused sub-service that owns the logic.

    Args:
        storage_dir: Override the storage root (pass explicitly in tests).
    """

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self._repo = LibraryRepository(storage_dir)
        self._documents = DocumentService(self._repo)
        self._ingest = IngestService(self._repo)
        self._search = SearchService(self._repo)

    # ── Storage path helpers (kept for backward compat) ───────

    @property
    def storage_dir(self) -> Path:
        """The root storage directory for this service instance."""
        return self._repo.storage_dir

    def library_storage_dir(self, library_id: str) -> Path:
        """Return the root directory for *library_id*."""
        return self._repo.library_storage_dir(library_id)

    def library_upload_dir(self, library_id: str) -> Path:
        """Return the uploads sub-directory for *library_id*."""
        return self._repo.library_upload_dir(library_id)

    def library_index_dir(self, library_id: str) -> Path:
        """Return the index sub-directory for *library_id*."""
        return self._repo.library_index_dir(library_id)

    # ── Library CRUD (→ LibraryRepository) ────────────────────

    def list_libraries(self) -> list[dict[str, Any]]:
        """List all knowledge libraries."""
        return self._repo.list_libraries()

    def create_library(
        self,
        name: str,
        description: str = "",
        mode: str = "auto",
        knowledge_dir: str = "./knowledge",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create a new knowledge library."""
        return self._repo.create_library(
            name,
            description,
            mode,
            knowledge_dir,
            metadata,
        )

    def delete_library(self, library_id: str) -> bool:
        """Delete a library and all its data."""
        return self._repo.delete_library(library_id)

    def get_library(self, library_id: str) -> dict[str, Any] | None:
        """Look up a library by ID."""
        return self._repo.get_library(library_id)

    def update_library(
        self,
        library_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Apply a partial update to a library."""
        return self._repo.update_library(library_id, updates)

    def cancel_ingest(self, library_id: str) -> None:
        """Signal cancellation of an in-progress ingest."""
        self._repo.cancel_ingest(library_id)

    # ── Ingest (→ IngestService) ──────────────────────────────

    async def ingest_files(
        self,
        library_id: str,
        paths: list[str],
        progress_callback: Any = None,
        incremental: bool = False,
    ) -> dict[str, Any]:
        """Ingest files into a knowledge library."""
        return await self._ingest.ingest_files(
            library_id,
            paths,
            progress_callback,
            incremental,
        )

    # ── Search (→ SearchService) ──────────────────────────────

    async def search_knowledge(
        self,
        query: str,
        library_id: str | None = None,
        mode: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search the knowledge base."""
        return await self._search.search_knowledge(
            query,
            library_id,
            mode,
            top_k,
        )

    # ── Document management (→ DocumentService) ───────────────

    def list_documents(self, library_id: str) -> list[dict[str, Any]]:
        """List all documents in a library."""
        return self._documents.list_documents(library_id)

    def get_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID."""
        return self._documents.get_document(library_id, doc_id)

    def add_document(
        self,
        library_id: str,
        file_path: str,
        status: str = "pending",
    ) -> dict[str, Any] | None:
        """Add a document to a library."""
        return self._documents.add_document(library_id, file_path, status)

    def update_document_status(
        self,
        library_id: str,
        doc_id: str,
        status: str,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any] | None:
        """Update a document's status."""
        return self._documents.update_document_status(
            library_id,
            doc_id,
            status,
            chunk_count,
            error_message,
        )

    def delete_document(self, library_id: str, doc_id: str) -> bool:
        """Delete a document from a library."""
        return self._documents.delete_document(library_id, doc_id)

    def disable_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Disable a document (exclude from retrieval)."""
        return self._documents.disable_document(library_id, doc_id)

    def enable_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Re-enable a disabled document."""
        return self._documents.enable_document(library_id, doc_id)

    def increment_retrieval_count(self, library_id: str, doc_id: str) -> None:
        """Bump the retrieval counter for a document."""
        self._documents.increment_retrieval_count(library_id, doc_id)

    # ── Chunk helpers (→ DocumentService) ─────────────────────

    def list_chunks(self, library_id: str, doc_id: str) -> list[dict[str, Any]]:
        """List stored chunks for a document."""
        return self._documents.list_chunks(library_id, doc_id)

    def preview_chunks(
        self,
        content: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "recursive",
    ) -> list[dict[str, Any]]:
        """Preview how content would be chunked."""
        return self._documents.preview_chunks(
            content,
            chunk_size,
            chunk_overlap,
            strategy,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_knowledge_service: KnowledgeService | None = None


def get_knowledge_service() -> KnowledgeService:
    """Return (or create) the global :class:`KnowledgeService` singleton."""
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service
