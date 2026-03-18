"""Tests for IngestService — file discovery, cancellation, and incremental filtering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from houyi_studio.server.rag import ingest_service as ingest_service_module
from houyi_studio.server.rag.ingest_service import IngestService
from houyi_studio.server.rag.library_repository import LibraryRepository

import houyi.rag as rag_module
from houyi.rag.indexed.document.loaders import SUPPORTED_DOCUMENT_SUFFIXES


@pytest.fixture
def repo(tmp_path: Path) -> LibraryRepository:
    """Isolated repository for ingest tests."""
    return LibraryRepository(storage_dir=tmp_path)


@pytest.fixture
def svc(repo: LibraryRepository) -> IngestService:
    """IngestService wired to the test repository."""
    return IngestService(repo)


@pytest.fixture
def library_id(repo: LibraryRepository) -> str:
    """Pre-created library for ingest tests."""
    lib = repo.create_library(name="IngestLib", knowledge_dir="/nonexistent")
    return lib["library_id"]


def _make_files(directory: Path, names: list[str]) -> list[str]:
    """Create dummy .md files and return their string paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        f = directory / name
        f.write_text(f"content of {name}")
        paths.append(str(f))
    return paths


def _supported_named_files() -> list[str]:
    return [
        f"f{idx:02d}{suffix}" for idx, suffix in enumerate(SUPPORTED_DOCUMENT_SUFFIXES, start=1)
    ]


# ── Basic ingest scenarios ───────────────────────────────────────


class TestIngestFiles:
    """Core ingest_files entry point (RAG engine mocked away)."""

    @pytest.mark.asyncio
    async def test_suffixes(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        from houyi.rag.config import EmbeddingConfig

        file_names = _supported_named_files()
        paths = _make_files(tmp_path / "all-formats", file_names)

        class _SuccessRAG:
            def __init__(self, *_args, **_kwargs):
                pass

            async def index(self, paths):
                assert len(paths) == 1
                return {"chunks": 1}

        with (
            patch.object(
                ingest_service_module,
                "resolve_embedding_config",
                return_value=(
                    EmbeddingConfig(
                        provider="openai",
                        model="text-embedding-3-small",
                        dimension=1536,
                    ),
                    "openai",
                ),
            ),
            patch.object(rag_module, "RAG", _SuccessRAG),
        ):
            result = await svc.ingest_files(library_id, paths)

        assert result["success"] is True
        assert result["stats"]["files_processed"] == len(file_names)
        assert result["stats"]["files_failed"] == 0
        assert result["stats"]["chunks_created"] == len(file_names)

        lib = repo.get_library(library_id)
        assert lib["doc_count"] == len(file_names)
        assert lib["chunk_count"] == len(file_names)
        indexed_names = {d["file_name"] for d in lib["documents"].values()}
        assert indexed_names == set(file_names)
        assert all(d["status"] == "indexed" for d in lib["documents"].values())

    @pytest.mark.asyncio
    async def test_nonexistent_library(self, svc: IngestService):
        result = await svc.ingest_files("lib_nope", ["/some/file.md"])
        assert result["success"] is False
        assert "Library not found" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_file_list(
        self,
        svc: IngestService,
        library_id: str,
    ):
        result = await svc.ingest_files(library_id, ["/nonexistent/path"])
        assert result["success"] is False
        assert "No valid files" in result["error"]

    @pytest.mark.asyncio
    async def test_fallback_ingest_succeeds(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        paths = _make_files(tmp_path / "docs", ["a.md", "b.txt"])

        with patch.object(svc, "_ingest_with_rag", side_effect=ImportError):
            result = await svc.ingest_files(library_id, paths)

        assert result["success"] is True
        assert result["stats"]["files_processed"] == 2

        lib = repo.get_library(library_id)
        assert lib["doc_count"] == 2

    @pytest.mark.asyncio
    async def test_failed_doc_count(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        from houyi.rag.config import EmbeddingConfig

        paths = _make_files(tmp_path / "failed", ["broken.md"])

        class _BrokenRAG:
            def __init__(self, *_args, **_kwargs):
                pass

            async def index(self, paths):
                raise RuntimeError(f"index failed for {paths[0]}")

        with (
            patch.object(
                ingest_service_module,
                "resolve_embedding_config",
                return_value=(
                    EmbeddingConfig(
                        provider="openai", model="text-embedding-3-small", dimension=1536
                    ),
                    "openai",
                ),
            ),
            patch.object(rag_module, "RAG", _BrokenRAG),
        ):
            result = await svc.ingest_files(library_id, paths)

        assert result["success"] is False
        lib = repo.get_library(library_id)
        assert lib["doc_count"] == 0

    @pytest.mark.asyncio
    async def test_local_missing(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        from houyi.rag.config import EmbeddingConfig

        paths = _make_files(tmp_path / "local", ["a.md"])

        def _block_fastembed(name, *args, **kwargs):
            if name == "fastembed" or name.startswith("fastembed."):
                raise ImportError("blocked for test")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__
        with (
            patch.object(
                ingest_service_module,
                "resolve_embedding_config",
                return_value=(
                    EmbeddingConfig(
                        provider="local", model="BAAI/bge-small-en-v1.5", dimension=384
                    ),
                    "local",
                ),
            ),
            patch("builtins.__import__", side_effect=_block_fastembed),
        ):
            result = await svc.ingest_files(library_id, paths)

        assert result["success"] is True
        lib = repo.get_library(library_id)
        assert lib["doc_count"] == 1
        assert lib["chunk_count"] == 0
        assert lib["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_local_error(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        from houyi.rag.config import EmbeddingConfig

        paths = _make_files(tmp_path / "local-runtime-error", ["broken.md"])

        class _BrokenLocalRAG:
            def __init__(self, *_args, **_kwargs):
                pass

            async def index(self, paths):
                _ = paths
                raise RuntimeError("model_optimized.onnx failed: File doesn't exist")

        with (
            patch.object(
                ingest_service_module,
                "resolve_embedding_config",
                return_value=(
                    EmbeddingConfig(
                        provider="local", model="BAAI/bge-small-en-v1.5", dimension=384
                    ),
                    "local",
                ),
            ),
            patch.object(rag_module, "RAG", _BrokenLocalRAG),
        ):
            result = await svc.ingest_files(library_id, paths)

        assert result["success"] is True
        lib = repo.get_library(library_id)
        assert lib["doc_count"] == 1
        assert lib["chunk_count"] == 0
        assert lib["status"] == "degraded"
        doc = next(iter(lib["documents"].values()))
        assert doc["status"] == "indexed"
        assert "model_optimized.onnx" in str(doc.get("metadata", {}).get("degraded_reason", ""))

    @pytest.mark.asyncio
    async def test_reingest_error(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        from houyi.rag.config import EmbeddingConfig

        paths = _make_files(tmp_path / "retry", ["retry.md"])
        file_name = Path(paths[0]).name

        class _FailRAG:
            def __init__(self, *_args, **_kwargs):
                pass

            async def index(self, paths):
                raise RuntimeError(f"fastembed package required for local embedding: {paths[0]}")

        class _SuccessRAG:
            def __init__(self, *_args, **_kwargs):
                pass

            async def index(self, paths):
                return {"chunks": 3}

        with (
            patch.object(
                ingest_service_module,
                "resolve_embedding_config",
                return_value=(
                    EmbeddingConfig(
                        provider="openai",
                        model="text-embedding-3-small",
                        dimension=1536,
                    ),
                    "openai",
                ),
            ),
            patch.object(rag_module, "RAG", _FailRAG),
        ):
            first = await svc.ingest_files(library_id, paths)
        assert first["success"] is False

        with (
            patch.object(
                ingest_service_module,
                "resolve_embedding_config",
                return_value=(
                    EmbeddingConfig(
                        provider="openai",
                        model="text-embedding-3-small",
                        dimension=1536,
                    ),
                    "openai",
                ),
            ),
            patch.object(rag_module, "RAG", _SuccessRAG),
        ):
            second = await svc.ingest_files(library_id, paths)
        assert second["success"] is True

        lib = repo.get_library(library_id)
        doc = next(d for d in lib["documents"].values() if d["file_name"] == file_name)
        assert doc["status"] == "indexed"
        assert doc["chunk_count"] == 3
        assert "error_message" not in doc
        assert not doc.get("metadata", {}).get("error")


# ── Cancellation ─────────────────────────────────────────────────


class TestCancellation:
    """Ingest respects cancel_flags mid-loop."""

    @pytest.mark.asyncio
    async def test_cancel_stops_processing(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        paths = _make_files(tmp_path / "cancel", [f"f{i}.md" for i in range(5)])

        call_count = 0

        async def cancelling_fallback(library, lid, files, total, stats, cb):
            nonlocal call_count
            for _ in files:
                call_count += 1
                if call_count >= 2:
                    repo.cancel_ingest(lid)
                stats["files_processed"] += 1
                if repo.cancel_flags.get(lid):
                    repo.cancel_flags.pop(lid, None)
                    library["status"] = "ready" if stats["files_processed"] > 0 else "empty"
                    repo.save()
                    return

        with (
            patch.object(svc, "_ingest_with_rag", side_effect=ImportError),
            patch.object(svc, "_ingest_fallback", side_effect=cancelling_fallback),
        ):
            result = await svc.ingest_files(library_id, paths)

        assert result["stats"]["files_processed"] < 5


# ── File discovery ───────────────────────────────────────────────


class TestCollectFiles:
    """_collect_files expands directories and filters index paths."""

    def test_expands_directory(self, tmp_path: Path):
        expected = _supported_named_files()
        _make_files(tmp_path, expected)
        files = IngestService._collect_files([str(tmp_path)])
        names = {f.name for f in files}
        assert names == set(expected)

    def test_single_file(self, tmp_path: Path):
        paths = _make_files(tmp_path, ["solo.md"])
        files = IngestService._collect_files(paths)
        assert len(files) == 1
        assert files[0].name == "solo.md"

    def test_skips_unsupported_extension(self, tmp_path: Path):
        (tmp_path / "code.py").write_text("print(1)")
        files = IngestService._collect_files([str(tmp_path / "code.py")])
        assert len(files) == 1  # single file path is kept regardless

    def test_nonexistent_path_ignored(self, tmp_path: Path):
        files = IngestService._collect_files([str(tmp_path / "missing.md")])
        assert files == []


# ── Incremental config hash ──────────────────────────────────────


class TestIncrementalConfigHash:
    """_filter_incremental detects config changes via hash."""

    def _config_hash(self, **overrides) -> str:
        cfg = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "chunking_strategy": "recursive",
            "embedding_provider": "none",
            "embedding_model": "none",
            "embedding_dimension": 0,
        }
        cfg.update(overrides)
        return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

    @pytest.mark.asyncio
    async def test_config_change_forces_rebuild(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        paths = _make_files(tmp_path / "inc", ["x.md"])
        lib = repo.get_library(library_id)

        lib["file_index"] = {
            "_config_hash": "old_hash",
            str((tmp_path / "inc" / "x.md").resolve()): {
                "mtime": (tmp_path / "inc" / "x.md").stat().st_mtime,
                "size": (tmp_path / "inc" / "x.md").stat().st_size,
            },
        }
        repo.save()

        with (
            patch.object(svc, "_ingest_with_rag", side_effect=ImportError),
            patch.object(
                ingest_service_module,
                "resolve_embedding_config",
                return_value=(None, "no_provider"),
            ),
        ):
            result = await svc.ingest_files(library_id, paths, incremental=True)

        assert result["stats"]["files_processed"] >= 1

    @pytest.mark.asyncio
    async def test_unchanged_files_skipped(
        self,
        svc: IngestService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        paths = _make_files(tmp_path / "skip", ["y.md"])
        fp = tmp_path / "skip" / "y.md"
        key = str(fp.resolve())
        stat = fp.stat()

        lib = repo.get_library(library_id)
        current_hash = self._config_hash()
        lib["file_index"] = {
            "_config_hash": current_hash,
            key: {"mtime": stat.st_mtime, "size": stat.st_size},
        }
        lib["documents"] = {
            "doc_1": {
                "doc_id": "doc_1",
                "file_path": key,
                "status": "indexed",
            },
        }
        repo.save()

        with (
            patch.object(svc, "_ingest_with_rag", side_effect=ImportError),
            patch(
                "houyi_studio.server.rag.ingest_service.resolve_embedding_config",
                return_value=(None, "no_provider"),
            ),
        ):
            result = await svc.ingest_files(library_id, paths, incremental=True)

        assert result["success"] is True
        assert result["stats"]["files_processed"] == 0
        assert result["stats"]["files_skipped"] >= 1
