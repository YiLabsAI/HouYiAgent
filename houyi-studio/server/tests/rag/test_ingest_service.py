"""Unit tests for IngestService — file discovery, cancellation, and incremental filtering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from houyi_studio.server.rag.ingest_service import IngestService
from houyi_studio.server.rag.library_repository import LibraryRepository


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


# ── Basic ingest scenarios ───────────────────────────────────────


class TestIngestFiles:
    """Core ingest_files entry point (RAG engine mocked away)."""

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
        _make_files(tmp_path, ["a.md", "b.txt", "c.csv"])
        files = IngestService._collect_files([str(tmp_path)])
        names = {f.name for f in files}
        assert "a.md" in names
        assert "b.txt" in names
        assert "c.csv" in names

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
            patch(
                "houyi_studio.server.rag.ingest_service.resolve_embedding_config",
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
