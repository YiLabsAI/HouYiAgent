"""RAG (Retrieval-Augmented Generation) knowledge-base package.

Public API
----------
Facade / singleton:
    :class:`KnowledgeService`
        Unified facade that delegates to the focused sub-services.
    :func:`get_knowledge_service`
        Module-level singleton accessor.

Sub-services (for direct use when finer control is needed):
    :class:`LibraryRepository`
        Library CRUD and JSON persistence.
    :class:`DocumentService`
        Document/chunk management within a library.
    :class:`IngestService`
        File ingestion pipeline.
    :class:`SearchService`
        RAG query execution.

Embedding helpers:
    :func:`resolve_embedding_config`
    :func:`_detect_embedding_config`
    :func:`_make_embedding_config`
    :func:`_auto_detect_embedding`

Path / constant helpers:
    :data:`UPLOADS_SUBDIR`, :data:`INDEX_SUBDIR`, :data:`_LIB_ID_PREFIX`,
    :data:`KNOWLEDGE_STORAGE_DIR`
    :func:`_default_storage_dir`, :func:`get_library_storage_dir`,
    :func:`get_library_upload_dir`, :func:`get_library_index_dir`,
    :func:`is_index_path`
"""

from .document_service import DocumentService
from .embedding_config import (
    _LIB_ID_PREFIX,
    INDEX_SUBDIR,
    KNOWLEDGE_STORAGE_DIR,
    UPLOADS_SUBDIR,
    _auto_detect_embedding,
    _default_storage_dir,
    _detect_embedding_config,
    _make_embedding_config,
    get_library_index_dir,
    get_library_storage_dir,
    get_library_upload_dir,
    is_index_path,
    resolve_embedding_config,
)
from .ingest_service import IngestService
from .knowledge_service import KnowledgeService, get_knowledge_service
from .library_repository import LibraryRepository
from .search_service import SearchService

__all__ = [
    # Facade / singleton
    "KnowledgeService",
    "get_knowledge_service",
    # Sub-services
    "LibraryRepository",
    "DocumentService",
    "IngestService",
    "SearchService",
    # Embedding helpers
    "resolve_embedding_config",
    "_detect_embedding_config",
    "_make_embedding_config",
    "_auto_detect_embedding",
    # Path / constant helpers
    "UPLOADS_SUBDIR",
    "INDEX_SUBDIR",
    "_LIB_ID_PREFIX",
    "KNOWLEDGE_STORAGE_DIR",
    "_default_storage_dir",
    "get_library_storage_dir",
    "get_library_upload_dir",
    "get_library_index_dir",
    "is_index_path",
]
