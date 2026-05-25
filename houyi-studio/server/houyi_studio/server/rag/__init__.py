"""RAG (Retrieval-Augmented Generation) knowledge-base package.

Public API
----------
Facade / singleton:
    KnowledgeService
        Unified facade that delegates to the focused sub-services.
    get_knowledge_service
        Module-level singleton accessor.

Sub-services (for direct use when finer control is needed):
    LibraryRepository
        Library CRUD and JSON persistence.
    DocumentService
        Document/chunk management within a library.
    IngestService
        File ingestion pipeline.
    SearchService
        RAG query execution.

Embedding helpers:
    resolve_embedding_config
    _detect_embedding_config
    _make_embedding_config
    _auto_detect_embedding

Path / constant helpers:
    UPLOADS_SUBDIR, INDEX_SUBDIR, _LIB_ID_PREFIX,
    KNOWLEDGE_STORAGE_DIR
    _default_storage_dir, get_library_storage_dir,
    get_library_upload_dir, get_library_index_dir,
    is_index_path
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
    "INDEX_SUBDIR",
    "KNOWLEDGE_STORAGE_DIR",
    # Path / constant helpers
    "UPLOADS_SUBDIR",
    "_LIB_ID_PREFIX",
    "DocumentService",
    "IngestService",
    # Facade / singleton
    "KnowledgeService",
    # Sub-services
    "LibraryRepository",
    "SearchService",
    "_auto_detect_embedding",
    "_default_storage_dir",
    "_detect_embedding_config",
    "_make_embedding_config",
    "get_knowledge_service",
    "get_library_index_dir",
    "get_library_storage_dir",
    "get_library_upload_dir",
    "is_index_path",
    # Embedding helpers
    "resolve_embedding_config",
]
