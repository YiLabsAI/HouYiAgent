"""RAG Knowledge Base service for the console server."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Storage directory for knowledge library metadata and data
# Use absolute path to avoid issues with relative paths
KNOWLEDGE_STORAGE_DIR = Path(os.getenv("HOUYI_KNOWLEDGE_STORAGE", ".houyi/knowledge")).resolve()

# Subdirectory names (constants to avoid hardcoding)
UPLOADS_SUBDIR = "uploads"
INDEX_SUBDIR = "index"


def get_library_storage_dir(library_id: str) -> Path:
    """Get the storage directory for a specific library.

    Each library has its own isolated storage:
    - {STORAGE}/{library_id}/{UPLOADS_SUBDIR}/  - Uploaded files
    - {STORAGE}/{library_id}/{INDEX_SUBDIR}/    - Vector/sparse/graph indexes

    Args:
        library_id: Library identifier

    Returns:
        Absolute path to library storage directory
    """
    return KNOWLEDGE_STORAGE_DIR / library_id


def get_library_upload_dir(library_id: str) -> Path:
    """Get the upload directory for a specific library."""
    return get_library_storage_dir(library_id) / UPLOADS_SUBDIR


def get_library_index_dir(library_id: str) -> Path:
    """Get the index directory for a specific library."""
    return get_library_storage_dir(library_id) / INDEX_SUBDIR


def is_index_path(path: Path) -> bool:
    """Check if a path is an internal index file that should be skipped.

    Index files are in .houyi/*/index/ directories.
    Upload files in .houyi/*/uploads/ should NOT be skipped.

    Args:
        path: Path to check

    Returns:
        True if this is an index file that should be skipped
    """
    path_str = str(path)
    # Skip files in index directories, but allow uploads
    return f"/{INDEX_SUBDIR}/" in path_str and ".houyi" in path_str


@dataclass(slots=True)
class RAGService:
    """Legacy RAG service stub for execution engine compatibility."""

    async def search(self, _context: Any, _query: str | None = None) -> list[Any]:
        return []


class KnowledgeService:
    """Service for managing knowledge libraries and RAG operations."""

    def __init__(self) -> None:
        """Initialize the knowledge service."""
        self._libraries: dict[str, dict[str, Any]] = {}
        self._cancel_flags: dict[str, bool] = {}
        self._ensure_storage_dir()
        self._load_libraries()

    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        KNOWLEDGE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_libraries(self) -> None:
        """Load libraries from storage."""
        metadata_file = KNOWLEDGE_STORAGE_DIR / "libraries.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    self._libraries = json.load(f)
                logger.debug("Loaded %d knowledge libraries", len(self._libraries))
                # Migrate old data - ensure all libraries have correct status
                self._migrate_library_data()
            except Exception as e:
                logger.warning("Failed to load knowledge libraries: %s", e)
                self._libraries = {}

    def _migrate_library_data(self) -> None:
        """Migrate old library data to ensure consistency."""
        needs_save = False
        for lib_id, lib in self._libraries.items():
            # Ensure documents dict exists
            if "documents" not in lib:
                lib["documents"] = {}
                needs_save = True

            # Calculate correct status based on documents
            documents = lib.get("documents", {})
            doc_count = len(documents)
            indexed_count = sum(1 for d in documents.values() if d.get("status") == "indexed")
            error_count = sum(1 for d in documents.values() if d.get("status") == "error")

            # Fix status if missing or incorrect
            old_status = lib.get("status")
            if doc_count == 0:
                new_status = "empty"
            elif error_count == doc_count:
                new_status = "error"
            elif indexed_count < doc_count and error_count > 0:
                new_status = "partial"
            else:
                new_status = "ready"

            if old_status != new_status:
                lib["status"] = new_status
                needs_save = True
                logger.debug("Migrated library %s status: %s -> %s", lib_id, old_status, new_status)

            # Fix doc_count if incorrect
            if lib.get("doc_count") != doc_count:
                lib["doc_count"] = doc_count
                needs_save = True

            # Remove duplicate documents by file_path
            seen_paths = set()
            docs_to_remove = []
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

            # Update doc_count after dedup
            if docs_to_remove:
                lib["doc_count"] = len(documents)

        if needs_save:
            self._save_libraries()
            logger.debug("Library data migration complete")

    def _save_libraries(self) -> None:
        """Save libraries to storage."""
        metadata_file = KNOWLEDGE_STORAGE_DIR / "libraries.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(self._libraries, f, indent=2)
        except Exception as e:
            logger.error("Failed to save knowledge libraries: %s", e)

    def list_libraries(self) -> list[dict[str, Any]]:
        """List all knowledge libraries.

        Returns:
            List of library metadata dictionaries
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
            name: Library name
            description: Library description
            mode: RAG mode (agentic, indexed, auto)
            knowledge_dir: Path to knowledge directory
            metadata: Additional metadata including:
                - contextual_retrieval: bool - Enable contextual retrieval
                - chunk_size: int - Chunk size for splitting
                - chunk_overlap: int - Overlap between chunks

        Returns:
            Created library metadata or None if duplicate name
        """
        # Check for duplicate name
        for lib in self._libraries.values():
            if lib.get("name") == name:
                logger.warning("Library with name '%s' already exists", name)
                return None

        library_id = f"lib_{uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        library = {
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

        # Count documents if directory exists
        dir_path = Path(knowledge_dir)
        if dir_path.exists():
            doc_count = sum(
                1
                for f in dir_path.rglob("*")
                if f.is_file() and f.suffix in [".md", ".txt", ".pdf", ".json", ".csv"]
            )
            library["doc_count"] = doc_count

        # Create library storage directories
        storage_dir = get_library_storage_dir(library_id)
        get_library_upload_dir(library_id).mkdir(parents=True, exist_ok=True)
        get_library_index_dir(library_id).mkdir(parents=True, exist_ok=True)
        logger.debug("Created library storage at: %s", storage_dir)

        self._libraries[library_id] = library
        self._save_libraries()

        logger.info("Created knowledge library: %s (%s)", name, library_id)
        return library

    def delete_library(self, library_id: str) -> bool:
        """Delete a knowledge library and all its data.

        This removes:
        - Library metadata from libraries.json
        - Uploaded files in {STORAGE}/{library_id}/uploads/
        - Index files in {STORAGE}/{library_id}/index/

        Args:
            library_id: Library ID to delete

        Returns:
            True if deleted, False if not found
        """
        if library_id not in self._libraries:
            return False

        # Delete library storage directory (uploads, index, etc.)
        storage_dir = get_library_storage_dir(library_id)
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
        """Get a knowledge library by ID.

        Args:
            library_id: Library ID

        Returns:
            Library metadata or None if not found
        """
        return self._libraries.get(library_id)

    def update_library(
        self,
        library_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Update a knowledge library.

        Args:
            library_id: Library ID
            updates: Fields to update

        Returns:
            Updated library or None if not found
        """
        library = self._libraries.get(library_id)
        if not library:
            return None

        # Update allowed fields (including metadata for settings)
        for key in ["name", "description", "mode", "doc_count", "chunk_count", "status", "metadata"]:
            if key in updates:
                if key == "metadata":
                    # Merge metadata instead of replacing
                    old_meta = library.get("metadata", {})
                    library["metadata"] = {**old_meta, **updates[key]}
                else:
                    library[key] = updates[key]

        library["updated_at"] = datetime.now().isoformat()
        self._save_libraries()

        logger.info("Updated knowledge library: %s with keys: %s", library_id, list(updates.keys()))
        return library

    def cancel_ingest(self, library_id: str) -> None:
        """Request cancellation of an ongoing ingest operation."""
        self._cancel_flags[library_id] = True
        logger.info("Cancel requested for library: %s", library_id)

    async def ingest_files(
        self,
        library_id: str,
        paths: list[str],
        progress_callback: Any = None,
        incremental: bool = False,
    ) -> dict[str, Any]:
        """Ingest files into a knowledge library.

        Args:
            library_id: Target library ID
            paths: List of file/directory paths to ingest
            progress_callback: Async callback for progress updates
            incremental: If True, skip files that haven't changed since last ingest

        Returns:
            Ingest statistics
        """
        library = self._libraries.get(library_id)
        if not library:
            return {
                "success": False,
                "error": f"Library not found: {library_id}",
                "stats": {},
            }

        # Collect all files to process
        all_files: list[Path] = []
        for path_str in paths:
            path = Path(path_str)
            if path.is_file():
                # Skip internal index files, but allow uploaded files
                if not is_index_path(path):
                    all_files.append(path)
            elif path.is_dir():
                for ext in [".md", ".txt", ".pdf", ".json", ".csv", ".html"]:
                    for f in path.rglob(f"*{ext}"):
                        # Skip internal index files, but allow uploaded files
                        if not is_index_path(f):
                            all_files.append(f)

        if not all_files:
            return {
                "success": False,
                "error": "No valid files found in the specified paths",
                "stats": {"files_found": 0},
            }

        # Incremental: filter to only new/modified files OR files that previously failed
        files_skipped = 0
        if incremental:
            file_index = library.get("file_index", {})
            documents = library.get("documents", {})

            # Check if config has changed - if so, need full rebuild
            metadata = library.get("metadata", {})
            current_config = {
                "chunk_size": metadata.get("chunk_size", 512),
                "chunk_overlap": metadata.get("chunk_overlap", 50),
                "chunking_strategy": metadata.get("chunking_strategy", "recursive"),
            }
            import hashlib
            import json
            current_config_hash = hashlib.md5(json.dumps(current_config, sort_keys=True).encode()).hexdigest()[:8]
            saved_config_hash = file_index.get("_config_hash", "")

            # If no saved hash (old data) or hash changed, force full rebuild
            if not saved_config_hash:
                logger.debug("No saved config hash (old data), forcing full rebuild to ensure consistency")
                file_index = {}  # Clear file index to force full rebuild
            elif saved_config_hash != current_config_hash:
                logger.debug("Config changed (hash %s -> %s), forcing full rebuild", saved_config_hash, current_config_hash)
                file_index = {}  # Clear file index to force full rebuild

            # Build a map of file_path -> document status for quick lookup
            doc_status_by_path = {}
            for doc in documents.values():
                doc_path = doc.get("file_path")
                if doc_path:
                    doc_status_by_path[doc_path] = doc.get("status", "pending")

            filtered = []
            for fp in all_files:
                key = str(fp.resolve())
                try:
                    stat = fp.stat()
                    current_mtime = stat.st_mtime
                    current_size = stat.st_size
                except OSError:
                    filtered.append(fp)  # can't stat, try anyway
                    continue

                # Check if document previously failed - always retry failed docs
                doc_status = doc_status_by_path.get(key)
                if doc_status in ("error", "pending"):
                    logger.debug("Retrying previously failed file: %s (status=%s)", fp.name, doc_status)
                    filtered.append(fp)
                    continue

                # Check file modification time/size for successfully indexed files
                prev = file_index.get(key)
                if prev and prev.get("mtime") == current_mtime and prev.get("size") == current_size:
                    files_skipped += 1
                else:
                    filtered.append(fp)

            all_files = filtered
            if not all_files:
                logger.debug("Incremental ingest: all %d files up to date", files_skipped)
                return {
                    "success": True,
                    "stats": {
                        "files_processed": 0,
                        "files_skipped": files_skipped,
                        "chunks_created": 0,
                        "errors": [],
                    },
                }
            logger.debug("Incremental ingest: %d new/modified/failed, %d skipped", len(all_files), files_skipped)

        total_files = len(all_files)
        logger.debug("Starting ingest for library %s: %d files", library_id, total_files)

        # Update library status
        library["status"] = "indexing"
        self._save_libraries()

        stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "files_skipped": files_skipped,
            "errors": [],
        }

        # Clear cancel flag before starting
        self._cancel_flags.pop(library_id, None)

        try:
            # Try to use RAG service for proper indexing
            from houyi.rag import RAG as HouyiRAG
            from houyi.rag.config import EmbeddingConfig, RAGConfig

            knowledge_dir = library.get("knowledge_dir", "./knowledge")

            # Try embedding providers in order of preference:
            # 1. Gemini/Vertex (if configured) - recommended
            # 2. OpenAI (if API key available)
            # 3. Local fastembed (requires model download)
            embedding_config = None
            import os

            # Check for Vertex/Gemini configuration
            vertex_project = os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if vertex_project:
                try:
                    from google import genai  # noqa: F401
                    embedding_config = EmbeddingConfig(
                        provider="gemini",
                        model="text-embedding-004",
                        dimension=768,
                    )
                    logger.debug("Using Gemini embedding (Vertex AI)")
                except ImportError:
                    logger.debug("google-genai not installed, trying other providers")

            # Try OpenAI if Gemini not available
            if embedding_config is None and os.environ.get("OPENAI_API_KEY"):
                embedding_config = EmbeddingConfig(
                    provider="openai",
                    model="text-embedding-3-small",
                    dimension=1536,
                )
                logger.debug("Using OpenAI embedding")

            # Try local embedding as last resort
            if embedding_config is None:
                try:
                    import fastembed  # noqa: F401
                    embedding_config = EmbeddingConfig(
                        provider="local",
                        model="BAAI/bge-small-en-v1.5",
                        dimension=384,
                    )
                    logger.debug("Using local embedding (fastembed)")
                except ImportError:
                    # No embedding available - fall back to simple file counting
                    logger.warning("No embedding provider available, falling back to simple file counting")
                    raise ImportError("No embedding provider available") from None

            # Get library-level options from metadata
            lib_metadata = library.get("metadata", {})
            contextual_retrieval = lib_metadata.get("contextual_retrieval", False)

            config = RAGConfig(
                mode="indexed",
                knowledge_dir=knowledge_dir,
                # Use library-specific index directory to avoid conflicts
                index_dir=str(get_library_index_dir(library_id)),
                embedding=embedding_config,
                contextual_retrieval=contextual_retrieval,
            )
            rag_service = HouyiRAG(config)

            # Process files with progress updates
            for i, file_path in enumerate(all_files):
                # Check for cancellation
                if self._cancel_flags.get(library_id):
                    logger.debug("Ingest cancelled for library %s at file %d/%d", library_id, i, total_files)
                    self._cancel_flags.pop(library_id, None)
                    library["status"] = "ready" if stats["files_processed"] > 0 else "empty"
                    self._save_libraries()
                    return {
                        "success": False,
                        "error": "Cancelled by user",
                        "stats": stats,
                    }

                # Check if document already exists (by file path) - update instead of creating new
                file_path_str = str(file_path.resolve())
                if "documents" not in library:
                    library["documents"] = {}

                existing_doc_id = None
                for doc_id, doc in library["documents"].items():
                    if doc.get("file_path") == file_path_str:
                        existing_doc_id = doc_id
                        break

                file_size = 0
                try:
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                except OSError:
                    pass

                if existing_doc_id:
                    # Update existing document
                    doc_id = existing_doc_id
                    doc_metadata = library["documents"][doc_id]
                    doc_metadata["file_size"] = file_size
                    doc_metadata["status"] = "pending"
                    doc_metadata["updated_at"] = datetime.now().isoformat()
                else:
                    # Create new document
                    doc_id = f"doc_{uuid4().hex[:8]}"
                    doc_metadata = {
                        "doc_id": doc_id,
                        "library_id": library_id,
                        "file_path": file_path_str,
                        "file_name": file_path.name,
                        "file_size": file_size,
                        "status": "pending",
                        "chunk_count": 0,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "metadata": {},
                    }
                    library["documents"][doc_id] = doc_metadata

                try:
                    # Send progress update
                    if progress_callback:
                        progress = (i / total_files) * 100
                        await progress_callback(
                            progress=progress,
                            current_file=str(file_path.name),
                            files_processed=i,
                            total_files=total_files,
                        )

                    # Ingest the file
                    result = await rag_service.index(paths=[str(file_path)])
                    stats["files_processed"] += 1
                    chunks_created = result.get("chunks", 0)
                    stats["chunks_created"] += chunks_created

                    # Update document metadata with success status
                    doc_metadata["status"] = "indexed"
                    doc_metadata["chunk_count"] = chunks_created
                    doc_metadata["updated_at"] = datetime.now().isoformat()

                except Exception as e:
                    stats["files_failed"] += 1
                    stats["errors"].append(f"{file_path.name}: {e}")
                    logger.warning("Failed to ingest %s: %s", file_path, e)
                    # Mark document as failed but keep it in the list
                    doc_metadata["status"] = "error"
                    doc_metadata["metadata"]["error"] = str(e)

            # Final progress update
            if progress_callback:
                await progress_callback(
                    progress=100,
                    current_file="",
                    files_processed=total_files,
                    total_files=total_files,
                )

        except ImportError:
            logger.warning("RAG service not available, using simple file counting")
            # Fallback: just count files without actual indexing
            for i, file_path in enumerate(all_files):
                # Check for cancellation
                if self._cancel_flags.get(library_id):
                    self._cancel_flags.pop(library_id, None)
                    library["status"] = "ready" if stats["files_processed"] > 0 else "empty"
                    self._save_libraries()
                    return {
                        "success": False,
                        "error": "Cancelled by user",
                        "stats": stats,
                    }

                if progress_callback:
                    progress = ((i + 1) / total_files) * 100
                    await progress_callback(
                        progress=progress,
                        current_file=str(file_path.name),
                        files_processed=i + 1,
                        total_files=total_files,
                    )
                stats["files_processed"] += 1

                # Check if document already exists (by file path) - update instead of creating new
                file_path_str = str(file_path.resolve())
                if "documents" not in library:
                    library["documents"] = {}

                existing_doc_id = None
                for doc_id, doc in library["documents"].items():
                    if doc.get("file_path") == file_path_str:
                        existing_doc_id = doc_id
                        break

                file_size = 0
                try:
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                except OSError:
                    pass

                if existing_doc_id:
                    # Update existing document
                    doc_metadata = library["documents"][existing_doc_id]
                    doc_metadata["file_size"] = file_size
                    doc_metadata["status"] = "indexed"
                    doc_metadata["updated_at"] = datetime.now().isoformat()
                else:
                    # Create new document
                    doc_id = f"doc_{uuid4().hex[:8]}"
                    doc_metadata = {
                        "doc_id": doc_id,
                        "library_id": library_id,
                        "file_path": file_path_str,
                        "file_name": file_path.name,
                        "file_size": file_size,
                        "status": "indexed",
                        "chunk_count": 0,  # Unknown in fallback mode
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "metadata": {},
                    }
                    library["documents"][doc_id] = doc_metadata

        except Exception as e:
            logger.error("Ingest failed: %s", e)
            library["status"] = "error"
            self._save_libraries()
            return {
                "success": False,
                "error": str(e),
                "stats": stats,
            }

        # Update library metadata based on actual documents
        documents = library.get("documents", {})
        total_docs = len(documents)
        indexed_docs = sum(1 for d in documents.values() if d.get("status") == "indexed")
        total_chunks = sum(d.get("chunk_count", 0) for d in documents.values())

        # Set status based on document states
        if total_docs == 0:
            library["status"] = "empty"
        elif indexed_docs == 0 and stats["files_failed"] > 0:
            library["status"] = "error"
        elif indexed_docs < total_docs:
            library["status"] = "partial"  # Some files failed
        else:
            library["status"] = "ready"

        library["doc_count"] = total_docs
        library["chunk_count"] = total_chunks
        library["updated_at"] = datetime.now().isoformat()

        # Update file index for incremental ingest support
        # ONLY add successfully indexed files to file_index
        file_index = library.get("file_index", {}) if incremental else {}
        documents = library.get("documents", {})
        for fp in all_files:
            key = str(fp.resolve())
            # Check if this file was successfully indexed
            doc_status = None
            for doc in documents.values():
                if doc.get("file_path") == key:
                    doc_status = doc.get("status")
                    break

            # Only add to file_index if successfully indexed
            if doc_status == "indexed":
                try:
                    stat = fp.stat()
                    file_index[key] = {"mtime": stat.st_mtime, "size": stat.st_size}
                except OSError:
                    pass
            elif key in file_index and doc_status in ("error", "pending"):
                # Remove from file_index if it previously succeeded but now failed
                # (This shouldn't happen normally, but just in case)
                pass

        # Save config hash for detecting config changes on next incremental rebuild
        metadata = library.get("metadata", {})
        current_config = {
            "chunk_size": metadata.get("chunk_size", 512),
            "chunk_overlap": metadata.get("chunk_overlap", 50),
            "chunking_strategy": metadata.get("chunking_strategy", "recursive"),
        }
        import hashlib
        import json
        config_hash = hashlib.md5(json.dumps(current_config, sort_keys=True).encode()).hexdigest()[:8]
        file_index["_config_hash"] = config_hash

        library["file_index"] = file_index

        if files_skipped > 0:
            stats["files_skipped"] = files_skipped

        self._save_libraries()

        logger.info(
            "Ingest complete for library %s: %d files processed, %d failed, %d chunks",
            library_id,
            stats["files_processed"],
            stats["files_failed"],
            stats["chunks_created"],
        )

        # Determine success based on whether any files were processed
        if stats["files_processed"] == 0 and stats["files_failed"] > 0:
            return {
                "success": False,
                "error": f"All {stats['files_failed']} files failed to process",
                "stats": stats,
            }

        return {
            "success": True,
            "stats": stats,
        }

    async def search_knowledge(
        self,
        query: str,
        library_id: str | None = None,
        mode: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search knowledge base.

        Args:
            query: Search query
            library_id: Optional library ID to search in
            mode: Optional mode override
            top_k: Number of results to return

        Returns:
            Search results dictionary
        """
        # Get library
        library = None
        if library_id:
            library = self._libraries.get(library_id)
            if not library:
                return {
                    "query": query,
                    "library_id": library_id,
                    "results": [],
                    "mode_used": "none",
                    "total_results": 0,
                    "error": f"Library not found: {library_id}",
                }

        # Determine mode
        library_mode = library.get("mode") if library else None
        logger.info("search_knowledge: mode=%s, library_id=%s", library_mode, library_id)
        effective_mode = mode or library_mode or "agentic"

        # Determine knowledge_dir and index_dir for the library
        # If library exists, use the library's upload directory for knowledge and index directory for indexes
        if library_id:
            # For libraries with uploaded files, use the library-specific directories
            knowledge_dir = str(get_library_upload_dir(library_id))
            index_dir = str(get_library_index_dir(library_id))
        else:
            # Fallback for search without library
            knowledge_dir = "./knowledge"
            index_dir = None

        try:
            # Try to use the RAG service
            import os

            from houyi.rag import RAG as HouyiRAG
            from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig, RAGConfig
            from houyi.rag.types import RetrievalStrategy

            # Determine embedding config for indexed mode
            embedding_config = None
            if effective_mode in ("indexed", "auto"):
                logger.debug("Search mode=%s, checking embedding providers...", effective_mode)

                # Check for Vertex/Gemini configuration
                vertex_project = os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
                logger.debug("VERTEX_PROJECT=%s, GOOGLE_CLOUD_PROJECT=%s",
                           os.environ.get("VERTEX_PROJECT"), os.environ.get("GOOGLE_CLOUD_PROJECT"))
                if vertex_project:
                    try:
                        from google import genai  # noqa: F401
                        embedding_config = EmbeddingConfig(
                            provider="gemini",
                            model="text-embedding-004",
                            dimension=768,
                        )
                        logger.debug("Using Gemini embedding for search")
                    except ImportError:
                        logger.warning("google-genai not installed for search")

                # Try OpenAI if Gemini not available
                if embedding_config is None and os.environ.get("OPENAI_API_KEY"):
                    embedding_config = EmbeddingConfig(
                        provider="openai",
                        model="text-embedding-3-small",
                        dimension=1536,
                    )
                    logger.debug("Using OpenAI embedding for search")

                # Try local embedding as last resort
                if embedding_config is None:
                    try:
                        import fastembed  # noqa: F401
                        embedding_config = EmbeddingConfig(
                            provider="local",
                            model="BAAI/bge-small-en-v1.5",
                            dimension=384,
                        )
                        logger.debug("Using local embedding for search")
                    except ImportError:
                        # Fall back to agentic mode if no embedding available
                        logger.warning("No embedding provider for indexed mode, falling back to agentic")
                        effective_mode = "agentic"

                logger.debug("Final effective_mode=%s, embedding_config=%s", effective_mode, embedding_config)

            # Build indexed config from library metadata
            indexed_config = None
            graph_config = None
            if library and effective_mode == "indexed":
                lib_metadata = library.get("metadata", {})
                strategies_raw = lib_metadata.get("strategies", ["bm25", "vector"])
                strategies = []
                has_graph = False
                for s in strategies_raw:
                    s_lower = s.lower() if isinstance(s, str) else s
                    if s_lower == "bm25":
                        strategies.append(RetrievalStrategy.BM25)
                    elif s_lower == "vector":
                        strategies.append(RetrievalStrategy.VECTOR)
                    elif s_lower == "graph":
                        strategies.append(RetrievalStrategy.GRAPH)
                        has_graph = True
                if strategies:
                    indexed_config = IndexedConfig(strategies=strategies)
                    logger.debug("Using strategies from library metadata: %s", [s.value for s in strategies])
                if has_graph:
                    graph_config = GraphConfig(enabled=True)
                    logger.debug("Graph retrieval enabled")

            # Build config - only pass embedding if we have one
            config_kwargs: dict[str, Any] = {
                "mode": effective_mode,
                "knowledge_dir": knowledge_dir,
                "index_dir": index_dir,
            }
            if embedding_config is not None:
                config_kwargs["embedding"] = embedding_config
            if indexed_config is not None:
                config_kwargs["indexed"] = indexed_config
            if graph_config is not None:
                config_kwargs["graph"] = graph_config

            config = RAGConfig(**config_kwargs)
            rag_service = HouyiRAG(config)

            # Perform search
            result = await rag_service.query(query, top_k=top_k)

            # Format results
            search_results = []
            for sr in result.search_results:
                search_results.append({
                    "chunk_id": sr.chunk_id,
                    "content": sr.content,
                    "score": sr.score,
                    "source": {
                        "file_path": sr.source.file_path if sr.source else "",
                        "location": sr.source.location if sr.source else "",
                        "snippet": sr.source.snippet if sr.source else "",
                    } if sr.source else None,
                    "metadata": sr.metadata,
                })

            # v1.1: Extract quality summary
            quality_data = None
            if result.quality:
                quality_data = {
                    "min_score": result.quality.min_score,
                    "max_score": result.quality.max_score,
                    "avg_score": result.quality.avg_score,
                    "above_threshold_count": result.quality.above_threshold_count,
                    "total_count": result.quality.total_count,
                    "relevance": result.quality.relevance,
                    "coverage": result.quality.coverage,
                    "confidence_level": result.quality.confidence_level,
                    "suggestion": result.quality.suggestion,
                    "score_distribution": result.quality.score_distribution,
                }

            return {
                "query": query,
                "library_id": library_id or "",
                "answer": result.answer,
                "results": search_results,
                "mode_used": result.mode_used.value if hasattr(result.mode_used, "value") else str(result.mode_used),
                "strategies_used": [s.value if hasattr(s, "value") else str(s) for s in result.strategies_used],
                "confidence": result.confidence,
                "total_results": len(search_results),
                "metadata": result.metadata,
                "quality": quality_data,  # v1.1
            }

        except ImportError:
            logger.warning("RAG service not available, using fallback search")
            return await self._fallback_search(query, knowledge_dir, top_k)

        except Exception as e:
            logger.error("RAG search failed: %s", e)
            return {
                "query": query,
                "library_id": library_id or "",
                "results": [],
                "mode_used": "error",
                "total_results": 0,
                "error": str(e),
            }

    async def _fallback_search(
        self,
        query: str,
        knowledge_dir: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Fallback search using simple file grep.

        Args:
            query: Search query
            knowledge_dir: Directory to search
            top_k: Number of results

        Returns:
            Search results
        """
        results = []
        dir_path = Path(knowledge_dir)

        if not dir_path.exists():
            return {
                "query": query,
                "library_id": "",
                "results": [],
                "mode_used": "fallback",
                "total_results": 0,
                "error": f"Directory not found: {knowledge_dir}",
            }

        try:
            # Use simple text search for query terms
            keywords = query.lower().split()

            for file_path in dir_path.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix not in [".md", ".txt", ".json"]:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    content_lower = content.lower()

                    # Calculate simple relevance score
                    score = sum(1 for kw in keywords if kw in content_lower) / len(keywords)

                    if score > 0:
                        # Find matching snippet
                        snippet = ""
                        for kw in keywords:
                            idx = content_lower.find(kw)
                            if idx >= 0:
                                start = max(0, idx - 100)
                                end = min(len(content), idx + 200)
                                snippet = content[start:end].strip()
                                break

                        results.append({
                            "chunk_id": f"chunk_{file_path.stem}_{len(results)}",
                            "content": content[:500] if len(content) > 500 else content,
                            "score": score,
                            "source": {
                                "file_path": str(file_path),
                                "location": "",
                                "snippet": snippet,
                            },
                            "metadata": {"file_name": file_path.name},
                        })

                except Exception as e:
                    logger.debug("Failed to read file %s: %s", file_path, e)

            # Sort by score and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

            return {
                "query": query,
                "library_id": "",
                "results": results,
                "mode_used": "fallback",
                "total_results": len(results),
            }

        except Exception as e:
            logger.error("Fallback search failed: %s", e)
            return {
                "query": query,
                "library_id": "",
                "results": [],
                "mode_used": "error",
                "total_results": 0,
                "error": str(e),
            }

    # ========== Document Management ==========

    def list_documents(self, library_id: str) -> list[dict[str, Any]]:
        """List all documents in a library.

        Args:
            library_id: Library ID

        Returns:
            List of document metadata dictionaries
        """
        library = self._libraries.get(library_id)
        if not library:
            return []

        documents = library.get("documents", {})
        return list(documents.values())

    def get_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID.

        Args:
            library_id: Library ID
            doc_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        library = self._libraries.get(library_id)
        if not library:
            return None

        documents = library.get("documents", {})
        return documents.get(doc_id)

    def add_document(
        self,
        library_id: str,
        file_path: str,
        status: str = "pending",
    ) -> dict[str, Any] | None:
        """Add a document to a library.

        Args:
            library_id: Library ID
            file_path: Path to the document file
            status: Initial status (pending/indexing/indexed/disabled/error)

        Returns:
            Created document metadata or None if library not found
        """
        library = self._libraries.get(library_id)
        if not library:
            return None

        path = Path(file_path)
        if not path.exists():
            return None

        doc_id = f"doc_{uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        # Get file info
        stat = path.stat()
        document = {
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
        self._save_libraries()

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
        """Update document status.

        Args:
            library_id: Library ID
            doc_id: Document ID
            status: New status
            chunk_count: Optional chunk count update
            error_message: Optional error message

        Returns:
            Updated document or None if not found
        """
        library = self._libraries.get(library_id)
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

        self._save_libraries()
        logger.debug("Updated document %s status to %s", doc_id, status)
        return document

    def delete_document(self, library_id: str, doc_id: str) -> bool:
        """Delete a document from a library.

        Args:
            library_id: Library ID
            doc_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        library = self._libraries.get(library_id)
        if not library:
            return False

        documents = library.get("documents", {})
        if doc_id not in documents:
            return False

        del documents[doc_id]

        # Recalculate library stats
        doc_count = len(documents)
        library["doc_count"] = doc_count
        library["chunk_count"] = sum(d.get("chunk_count", 0) for d in documents.values())

        # Recalculate status based on remaining documents
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
        self._save_libraries()

        logger.debug("Deleted document %s from library %s", doc_id, library_id)
        return True

    def disable_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Disable a document (exclude from retrieval).

        Args:
            library_id: Library ID
            doc_id: Document ID

        Returns:
            Updated document or None if not found
        """
        return self.update_document_status(library_id, doc_id, "disabled")

    def enable_document(self, library_id: str, doc_id: str) -> dict[str, Any] | None:
        """Enable a disabled document.

        Args:
            library_id: Library ID
            doc_id: Document ID

        Returns:
            Updated document or None if not found
        """
        return self.update_document_status(library_id, doc_id, "indexed")

    def increment_retrieval_count(self, library_id: str, doc_id: str) -> None:
        """Increment the retrieval count for a document.

        Args:
            library_id: Library ID
            doc_id: Document ID
        """
        library = self._libraries.get(library_id)
        if not library:
            return

        documents = library.get("documents", {})
        document = documents.get(doc_id)
        if document:
            document["retrieval_count"] = document.get("retrieval_count", 0) + 1
            # Don't save immediately for performance - batch save later

    # ========== Chunk Management ==========

    def list_chunks(self, library_id: str, doc_id: str) -> list[dict[str, Any]]:
        """List all chunks for a document.

        Args:
            library_id: Library ID
            doc_id: Document ID

        Returns:
            List of chunk metadata
        """
        library = self._libraries.get(library_id)
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
        """Preview how content would be chunked.

        Args:
            content: Content to chunk
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy

        Returns:
            List of preview chunks
        """
        # Simple chunking for preview
        chunks = []
        if strategy == "sentence":
            import re
            sentences = re.split(r'(?<=[.!?])\s+', content)
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
            # Recursive split
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


# Global service instance
_knowledge_service: KnowledgeService | None = None


def get_knowledge_service() -> KnowledgeService:
    """Get the global knowledge service instance."""
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service
