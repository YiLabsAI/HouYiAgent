"""File ingestion pipeline for knowledge libraries.

Responsibilities:
    - Discover files from user-supplied paths.
    - Perform incremental-ingest filtering (config-hash, mtime/size).
    - Drive the ``houyi.rag.RAG.index()`` pipeline per-file with
      progress callbacks and cancellation support.
    - Fall back to simple file counting when the RAG engine or an
      embedding provider is unavailable.

Dependencies:
    - :class:`~.library_repository.LibraryRepository` for library access
      and persistence.
    - :func:`~.embedding_config.resolve_embedding_config` and
      :func:`~.embedding_config.is_index_path` for embedding resolution
      and path filtering.

Thread Safety:
    One ingest operation per library at a time (enforced externally via
    cancel flags).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from houyi.infrastructure.config import ENV_EMBEDDING_PROVIDER
from houyi.rag.indexed.document.loaders import SUPPORTED_DOCUMENT_SUFFIXES

from .embedding_config import is_index_path, resolve_embedding_config
from .library_repository import LibraryRepository

logger = logging.getLogger(__name__)


class IngestService:
    """Orchestrates file ingestion into a knowledge library.

    Args:
        repo: The shared :class:`LibraryRepository` instance.
    """

    def __init__(self, repo: LibraryRepository) -> None:
        self._repo = repo

    # ── Public entry point ────────────────────────────────────

    async def ingest_files(
        self,
        library_id: str,
        paths: list[str],
        progress_callback: Any = None,
        incremental: bool = False,
    ) -> dict[str, Any]:
        """Ingest files into a knowledge library.

        Args:
            library_id: Target library ID.
            paths: File and/or directory paths to ingest.
            progress_callback: Optional async callable receiving
                ``progress``, ``current_file``, ``files_processed``,
                ``total_files`` keyword arguments.
            incremental: When ``True``, skip unchanged files.

        Returns:
            A result dict with ``success``, ``stats``, and optionally
            ``error`` / ``warning`` keys.
        """
        library = self._repo.get_library(library_id)
        if not library:
            return {
                "success": False,
                "error": f"Library not found: {library_id}",
                "stats": {},
            }

        all_files = self._collect_files(paths)

        if not all_files:
            return {
                "success": False,
                "error": "No valid files found in the specified paths",
                "stats": {"files_found": 0},
            }

        files_skipped = 0
        if incremental:
            all_files, files_skipped = self._filter_incremental(
                library,
                all_files,
            )
            if not all_files:
                logger.debug(
                    "Incremental ingest: all %d files up to date",
                    files_skipped,
                )
                return {
                    "success": True,
                    "stats": {
                        "files_processed": 0,
                        "files_skipped": files_skipped,
                        "chunks_created": 0,
                        "errors": [],
                    },
                }
            logger.debug(
                "Incremental ingest: %d new/modified/failed, %d skipped",
                len(all_files),
                files_skipped,
            )

        total_files = len(all_files)
        logger.debug("Starting ingest for library %s: %d files", library_id, total_files)

        library["status"] = "indexing"
        self._repo.save()

        stats: dict[str, Any] = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "files_skipped": files_skipped,
            "errors": [],
        }

        self._repo.cancel_flags.pop(library_id, None)

        try:
            await self._ingest_with_rag(
                library,
                library_id,
                all_files,
                total_files,
                stats,
                progress_callback,
            )
        except ImportError:
            logger.warning("RAG service not available, using simple file counting")
            await self._ingest_fallback(
                library,
                library_id,
                all_files,
                total_files,
                stats,
                progress_callback,
            )
        except Exception as e:
            logger.error("Ingest failed: %s", e)
            library["status"] = "error"
            self._repo.save()
            return {
                "success": False,
                "error": str(e),
                "stats": stats,
            }

        self._finalise(library, library_id, all_files, stats, incremental, files_skipped)

        if stats["files_processed"] == 0 and stats["files_failed"] > 0:
            return {
                "success": False,
                "error": f"All {stats['files_failed']} files failed to process",
                "stats": stats,
            }

        result: dict[str, Any] = {"success": True, "stats": stats}

        if library["status"] == "degraded":
            result["warning"] = (
                "No embedding provider configured — files were imported but "
                "semantic search is unavailable. Install an embedding provider "
                "(set OPENAI_API_KEY, or install fastembed for local embeddings) "
                "and rebuild the index."
            )

        return result

    # ── File discovery ────────────────────────────────────────

    @staticmethod
    def _collect_files(paths: list[str]) -> list[Path]:
        """Expand *paths* into a flat list of ingestible files."""
        all_files: list[Path] = []
        for path_str in paths:
            path = Path(path_str)
            if path.is_file():
                if not is_index_path(path):
                    all_files.append(path)
            elif path.is_dir():
                for ext in SUPPORTED_DOCUMENT_SUFFIXES:
                    for f in path.rglob(f"*{ext}"):
                        if not is_index_path(f):
                            all_files.append(f)
        return all_files

    # ── Incremental filtering ─────────────────────────────────

    def _filter_incremental(
        self,
        library: dict[str, Any],
        all_files: list[Path],
    ) -> tuple[list[Path], int]:
        """Return ``(files_to_ingest, skipped_count)``."""
        file_index = library.get("file_index", {})
        documents = library.get("documents", {})

        metadata = library.get("metadata", {})
        emb_cfg, _ = resolve_embedding_config(
            preferred_provider=metadata.get("embedding_provider"),
            preferred_model=metadata.get("embedding_model"),
            preferred_dimension=metadata.get("embedding_dimension"),
        )
        current_config = {
            "chunk_size": metadata.get("chunk_size", 512),
            "chunk_overlap": metadata.get("chunk_overlap", 50),
            "chunking_strategy": metadata.get("chunking_strategy", "recursive"),
            "embedding_provider": emb_cfg.provider if emb_cfg else "none",
            "embedding_model": emb_cfg.model if emb_cfg else "none",
            "embedding_dimension": emb_cfg.dimension if emb_cfg else 0,
        }
        current_config_hash = hashlib.md5(
            json.dumps(current_config, sort_keys=True).encode()
        ).hexdigest()[:8]
        saved_config_hash = file_index.get("_config_hash", "")

        if not saved_config_hash:
            logger.debug(
                "No saved config hash (old data), forcing full rebuild to ensure consistency"
            )
            file_index = {}
        elif saved_config_hash != current_config_hash:
            logger.debug(
                "Config changed (hash %s -> %s), forcing full rebuild",
                saved_config_hash,
                current_config_hash,
            )
            file_index = {}

        doc_status_by_path: dict[str, str] = {}
        for doc in documents.values():
            doc_path = doc.get("file_path")
            if doc_path:
                doc_status_by_path[doc_path] = doc.get("status", "pending")

        filtered: list[Path] = []
        files_skipped = 0
        for fp in all_files:
            key = str(fp.resolve())
            try:
                stat = fp.stat()
                current_mtime = stat.st_mtime
                current_size = stat.st_size
            except OSError:
                filtered.append(fp)
                continue

            doc_status = doc_status_by_path.get(key)
            if doc_status in ("error", "pending"):
                logger.debug(
                    "Retrying previously failed file: %s (status=%s)",
                    fp.name,
                    doc_status,
                )
                filtered.append(fp)
                continue

            prev = file_index.get(key)
            if prev and prev.get("mtime") == current_mtime and prev.get("size") == current_size:
                files_skipped += 1
            else:
                filtered.append(fp)

        return filtered, files_skipped

    # ── RAG-backed ingest ─────────────────────────────────────

    async def _ingest_with_rag(
        self,
        library: dict[str, Any],
        library_id: str,
        all_files: list[Path],
        total_files: int,
        stats: dict[str, Any],
        progress_callback: Any,
    ) -> None:
        from houyi.rag import RAG as HouyiRAG
        from houyi.rag.config import RAGConfig

        knowledge_dir = library.get("knowledge_dir", "./knowledge")

        lib_metadata = library.get("metadata", {})
        strict_explicit_local = (
            lib_metadata.get("embedding_provider") == "local"
            or os.environ.get(ENV_EMBEDDING_PROVIDER) == "local"
        )
        embedding_config, provider_name = resolve_embedding_config(
            preferred_provider=lib_metadata.get("embedding_provider"),
            preferred_model=lib_metadata.get("embedding_model"),
            preferred_dimension=lib_metadata.get("embedding_dimension"),
            strict_explicit=strict_explicit_local,
        )
        if embedding_config is None:
            logger.warning("No embedding provider available, falling back to simple file counting")
            raise ImportError("No embedding provider available") from None
        logger.info(
            "Using %s embedding for ingest (model=%s, dim=%d)",
            provider_name,
            embedding_config.model,
            embedding_config.dimension,
        )
        if embedding_config.provider == "local":
            try:
                import fastembed  # noqa: F401
            except ImportError:
                logger.warning(
                    "Local embedding provider selected but fastembed is unavailable; falling back to metadata-only ingest"
                )
                raise ImportError("fastembed package required for local embedding") from None

        library.setdefault("metadata", {})["embedding_provider"] = embedding_config.provider
        library["metadata"]["embedding_model"] = embedding_config.model
        library["metadata"]["embedding_dimension"] = embedding_config.dimension

        lib_metadata = library.get("metadata", {})
        contextual_retrieval = lib_metadata.get("contextual_retrieval", False)

        config = RAGConfig(
            mode="indexed",
            knowledge_dir=knowledge_dir,
            index_dir=str(self._repo.library_index_dir(library_id)),
            embedding=embedding_config,
            contextual_retrieval=contextual_retrieval,
        )
        rag_service = HouyiRAG(config)

        for i, file_path in enumerate(all_files):
            if self._repo.cancel_flags.get(library_id):
                logger.debug(
                    "Ingest cancelled for library %s at file %d/%d",
                    library_id,
                    i,
                    total_files,
                )
                self._repo.cancel_flags.pop(library_id, None)
                library["status"] = "ready" if stats["files_processed"] > 0 else "empty"
                self._repo.save()
                stats.setdefault("errors", [])
                return

            doc_id, doc_metadata, created_new = self._upsert_document_record(
                library,
                library_id,
                file_path,
            )

            try:
                if progress_callback:
                    progress = (i / total_files) * 100
                    await progress_callback(
                        progress=progress,
                        current_file=str(file_path.name),
                        files_processed=i,
                        total_files=total_files,
                    )

                result = await rag_service.index(paths=[str(file_path)])
                stats["files_processed"] += 1
                chunks_created = result.get("chunks", 0)
                stats["chunks_created"] += chunks_created

                doc_metadata["status"] = "indexed"
                doc_metadata["chunk_count"] = chunks_created
                doc_metadata["updated_at"] = datetime.now().isoformat()
                doc_metadata.pop("error_message", None)
                if isinstance(doc_metadata.get("metadata"), dict):
                    doc_metadata["metadata"].pop("error", None)

            except Exception as e:
                if embedding_config.provider == "local":
                    logger.warning(
                        "Local embedding failed for %s, fallback to metadata-only ingest: %s",
                        file_path,
                        e,
                    )
                    stats["files_processed"] += 1
                    doc_metadata["status"] = "indexed"
                    doc_metadata["chunk_count"] = 0
                    doc_metadata["updated_at"] = datetime.now().isoformat()
                    metadata = doc_metadata.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                        doc_metadata["metadata"] = metadata
                    metadata["degraded_reason"] = str(e)
                    metadata.pop("error", None)
                    continue
                stats["files_failed"] += 1
                stats["errors"].append(f"{file_path.name}: {e}")
                logger.warning("Failed to ingest %s: %s", file_path, e)
                if created_new:
                    with contextlib.suppress(Exception):
                        del library["documents"][doc_id]
                else:
                    doc_metadata["status"] = "error"
                    doc_metadata["metadata"]["error"] = str(e)

        if progress_callback:
            await progress_callback(
                progress=100,
                current_file="",
                files_processed=total_files,
                total_files=total_files,
            )

    # ── Fallback ingest (no RAG engine) ───────────────────────

    async def _ingest_fallback(
        self,
        library: dict[str, Any],
        library_id: str,
        all_files: list[Path],
        total_files: int,
        stats: dict[str, Any],
        progress_callback: Any,
    ) -> None:
        for i, file_path in enumerate(all_files):
            if self._repo.cancel_flags.get(library_id):
                self._repo.cancel_flags.pop(library_id, None)
                library["status"] = "ready" if stats["files_processed"] > 0 else "empty"
                self._repo.save()
                return

            if progress_callback:
                progress = ((i + 1) / total_files) * 100
                await progress_callback(
                    progress=progress,
                    current_file=str(file_path.name),
                    files_processed=i + 1,
                    total_files=total_files,
                )
            stats["files_processed"] += 1

            file_path_str = str(file_path.resolve())
            if "documents" not in library:
                library["documents"] = {}

            existing_doc_id = None
            for doc_id, doc in library["documents"].items():
                if doc.get("file_path") == file_path_str:
                    existing_doc_id = doc_id
                    break

            file_size = 0
            with contextlib.suppress(OSError):
                file_size = file_path.stat().st_size if file_path.exists() else 0

            if existing_doc_id:
                doc_metadata = library["documents"][existing_doc_id]
                doc_metadata["file_size"] = file_size
                doc_metadata["status"] = "indexed"
                doc_metadata["updated_at"] = datetime.now().isoformat()
                doc_metadata.pop("error_message", None)
                if isinstance(doc_metadata.get("metadata"), dict):
                    doc_metadata["metadata"].pop("error", None)
            else:
                doc_id = f"doc_{uuid4().hex[:8]}"
                doc_metadata = {
                    "doc_id": doc_id,
                    "library_id": library_id,
                    "file_path": file_path_str,
                    "file_name": file_path.name,
                    "file_size": file_size,
                    "status": "indexed",
                    "chunk_count": 0,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "metadata": {},
                }
                library["documents"][doc_id] = doc_metadata

    # ── Shared helpers ────────────────────────────────────────

    @staticmethod
    def _upsert_document_record(
        library: dict[str, Any],
        library_id: str,
        file_path: Path,
    ) -> tuple[str, dict[str, Any], bool]:
        """Create or update the document metadata record for *file_path*.

        Returns:
            ``(doc_id, doc_metadata, created_new)`` tuple.
        """
        file_path_str = str(file_path.resolve())
        if "documents" not in library:
            library["documents"] = {}

        existing_doc_id = None
        for did, doc in library["documents"].items():
            if doc.get("file_path") == file_path_str:
                existing_doc_id = did
                break

        file_size = 0
        with contextlib.suppress(OSError):
            file_size = file_path.stat().st_size if file_path.exists() else 0

        if existing_doc_id:
            doc_metadata = library["documents"][existing_doc_id]
            doc_metadata["file_size"] = file_size
            doc_metadata["status"] = "pending"
            doc_metadata["updated_at"] = datetime.now().isoformat()
            doc_metadata.pop("error_message", None)
            if isinstance(doc_metadata.get("metadata"), dict):
                doc_metadata["metadata"].pop("error", None)
            return existing_doc_id, doc_metadata, False

        doc_id = f"doc_{uuid4().hex[:8]}"
        doc_metadata: dict[str, Any] = {
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
        return doc_id, doc_metadata, True

    def _finalise(
        self,
        library: dict[str, Any],
        library_id: str,
        all_files: list[Path],
        stats: dict[str, Any],
        incremental: bool,
        files_skipped: int,
    ) -> None:
        """Update library metadata after ingest completes."""
        documents = library.get("documents", {})
        total_docs = len(documents)
        indexed_docs = sum(1 for d in documents.values() if d.get("status") == "indexed")
        total_chunks = sum(d.get("chunk_count", 0) for d in documents.values())

        if total_docs == 0:
            library["status"] = "empty"
        elif indexed_docs == 0 and stats["files_failed"] > 0:
            library["status"] = "error"
        elif indexed_docs < total_docs:
            library["status"] = "partial"
        elif total_chunks == 0 and indexed_docs > 0:
            library["status"] = "degraded"
        else:
            library["status"] = "ready"

        library["doc_count"] = total_docs
        library["chunk_count"] = total_chunks
        library["updated_at"] = datetime.now().isoformat()

        file_index = library.get("file_index", {}) if incremental else {}
        for fp in all_files:
            key = str(fp.resolve())
            doc_status = None
            for doc in documents.values():
                if doc.get("file_path") == key:
                    doc_status = doc.get("status")
                    break

            if doc_status == "indexed":
                try:
                    stat = fp.stat()
                    file_index[key] = {"mtime": stat.st_mtime, "size": stat.st_size}
                except OSError:
                    pass
            elif key in file_index and doc_status in ("error", "pending"):
                pass  # keep stale entry; will retry next time

        metadata = library.get("metadata", {})
        current_config = {
            "chunk_size": metadata.get("chunk_size", 512),
            "chunk_overlap": metadata.get("chunk_overlap", 50),
            "chunking_strategy": metadata.get("chunking_strategy", "recursive"),
            "embedding_provider": metadata.get("embedding_provider", "none"),
            "embedding_model": metadata.get("embedding_model", "none"),
            "embedding_dimension": metadata.get("embedding_dimension", 0),
        }
        config_hash = hashlib.md5(json.dumps(current_config, sort_keys=True).encode()).hexdigest()[
            :8
        ]
        file_index["_config_hash"] = config_hash

        library["file_index"] = file_index

        if files_skipped > 0:
            stats["files_skipped"] = files_skipped

        self._repo.save()

        logger.debug(
            "Ingest complete for library %s: %d files processed, %d failed, %d chunks",
            library_id,
            stats["files_processed"],
            stats["files_failed"],
            stats["chunks_created"],
        )
