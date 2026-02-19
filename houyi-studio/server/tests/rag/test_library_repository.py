"""Unit tests for LibraryRepository — library CRUD, persistence, migration, and safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from houyi_studio.server.rag.library_repository import LibraryRepository


@pytest.fixture
def repo(tmp_path: Path) -> LibraryRepository:
    """Fresh repository backed by an isolated tmp directory."""
    return LibraryRepository(storage_dir=tmp_path)


# ── Storage directory & path helpers ──────────────────────────────


class TestStoragePaths:
    """Path helpers must resolve under the injected storage root."""

    def test_storage_dir_created_on_init(self, tmp_path: Path):
        subdir = tmp_path / "nested" / "storage"
        repo = LibraryRepository(storage_dir=subdir)
        assert repo.storage_dir == subdir.resolve()
        assert repo.storage_dir.exists()

    def test_library_subdirectories(self, repo: LibraryRepository):
        lid = "lib_abc12345"
        assert repo.library_storage_dir(lid) == repo.storage_dir / lid
        assert repo.library_upload_dir(lid).name == "uploads"
        assert repo.library_index_dir(lid).name == "index"


# ── CRUD operations ──────────────────────────────────────────────


class TestCRUD:
    """Create, read, update, delete on libraries."""

    def test_create_and_get(self, repo: LibraryRepository):
        lib = repo.create_library(name="Alpha", description="desc", mode="indexed")
        assert lib is not None
        assert lib["name"] == "Alpha"
        assert lib["status"] == "empty"

        fetched = repo.get_library(lib["library_id"])
        assert fetched is not None
        assert fetched["library_id"] == lib["library_id"]

    def test_create_duplicate_name_returns_none(self, repo: LibraryRepository):
        repo.create_library(name="Dup")
        assert repo.create_library(name="Dup") is None

    def test_list_libraries(self, repo: LibraryRepository):
        repo.create_library(name="A")
        repo.create_library(name="B")
        names = {lib["name"] for lib in repo.list_libraries()}
        assert names == {"A", "B"}

    def test_update_basic_fields(self, repo: LibraryRepository):
        lib = repo.create_library(name="Orig")
        lid = lib["library_id"]
        updated = repo.update_library(lid, {"name": "New", "mode": "agentic"})
        assert updated["name"] == "New"
        assert updated["mode"] == "agentic"

    def test_update_merges_metadata(self, repo: LibraryRepository):
        lib = repo.create_library(name="Meta", metadata={"a": 1})
        lid = lib["library_id"]
        repo.update_library(lid, {"metadata": {"b": 2}})
        fetched = repo.get_library(lid)
        assert fetched["metadata"] == {"a": 1, "b": 2}

    def test_update_nonexistent_returns_none(self, repo: LibraryRepository):
        assert repo.update_library("lib_missing", {"name": "X"}) is None

    def test_delete_removes_library_and_disk(self, repo: LibraryRepository):
        lib = repo.create_library(name="Del")
        lid = lib["library_id"]
        storage = repo.library_storage_dir(lid)
        assert storage.exists()

        assert repo.delete_library(lid) is True
        assert repo.get_library(lid) is None
        assert not storage.exists()

    def test_delete_nonexistent_returns_false(self, repo: LibraryRepository):
        assert repo.delete_library("lib_nonexistent") is False

    def test_get_nonexistent_returns_none(self, repo: LibraryRepository):
        assert repo.get_library("lib_nope") is None

    def test_create_counts_existing_files(self, tmp_path: Path):
        kdir = tmp_path / "kdata"
        kdir.mkdir()
        (kdir / "readme.md").write_text("hello")
        (kdir / "notes.txt").write_text("world")

        repo = LibraryRepository(storage_dir=tmp_path / "store")
        lib = repo.create_library(name="WithFiles", knowledge_dir=str(kdir))
        assert lib["doc_count"] == 2


# ── Path-safety checks ──────────────────────────────────────────


class TestPathSafety:
    """delete_library must refuse unsafe IDs."""

    def test_rejects_bad_id_prefix(self, repo: LibraryRepository):
        repo._libraries["evil"] = {"library_id": "evil", "name": "bad"}
        assert repo.delete_library("evil") is False
        assert "evil" in repo._libraries

    def test_rejects_path_traversal(self, repo: LibraryRepository):
        bad_id = "lib_x/../../etc"
        repo._libraries[bad_id] = {"library_id": bad_id, "name": "bad"}
        assert repo.delete_library(bad_id) is False

    def test_rejects_deeply_nested_id(self, repo: LibraryRepository):
        nested_id = "lib_a/sub"
        nested_dir = repo.storage_dir / "lib_a" / "sub"
        nested_dir.mkdir(parents=True, exist_ok=True)
        repo._libraries[nested_id] = {"library_id": nested_id, "name": "nested"}
        assert repo.delete_library(nested_id) is False


# ── JSON persistence ─────────────────────────────────────────────


class TestPersistence:
    """Round-trip through libraries.json."""

    def test_save_and_reload(self, tmp_path: Path):
        repo1 = LibraryRepository(storage_dir=tmp_path)
        repo1.create_library(name="Persist")
        del repo1

        repo2 = LibraryRepository(storage_dir=tmp_path)
        libs = repo2.list_libraries()
        assert len(libs) == 1
        assert libs[0]["name"] == "Persist"

    def test_corrupt_json_falls_back_to_empty(self, tmp_path: Path):
        (tmp_path / "libraries.json").write_text("{bad json")
        repo = LibraryRepository(storage_dir=tmp_path)
        assert repo.list_libraries() == []


# ── Migration ────────────────────────────────────────────────────


class TestMigration:
    """_migrate_library_data fixes legacy data on load."""

    def _seed(self, tmp_path: Path, libraries: dict[str, Any]) -> LibraryRepository:
        tmp_path.mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "libraries.json", "w") as f:
            json.dump(libraries, f)
        return LibraryRepository(storage_dir=tmp_path)

    def test_missing_status_set_to_empty(self, tmp_path: Path):
        repo = self._seed(
            tmp_path,
            {
                "lib_old": {
                    "library_id": "lib_old",
                    "name": "Old",
                    "mode": "auto",
                    "doc_count": 0,
                },
            },
        )
        lib = repo.get_library("lib_old")
        assert lib["status"] == "empty"
        assert "documents" in lib

    def test_duplicate_docs_removed(self, tmp_path: Path):
        repo = self._seed(
            tmp_path,
            {
                "lib_dup": {
                    "library_id": "lib_dup",
                    "name": "Dup",
                    "mode": "auto",
                    "status": "ready",
                    "doc_count": 2,
                    "documents": {
                        "doc_1": {"doc_id": "doc_1", "file_path": "/same.md", "status": "indexed"},
                        "doc_2": {"doc_id": "doc_2", "file_path": "/same.md", "status": "indexed"},
                    },
                },
            },
        )
        lib = repo.get_library("lib_dup")
        assert lib["doc_count"] == 1
        assert len(lib["documents"]) == 1

    def test_degraded_status_when_indexed_but_zero_chunks(self, tmp_path: Path):
        repo = self._seed(
            tmp_path,
            {
                "lib_deg": {
                    "library_id": "lib_deg",
                    "name": "Degraded",
                    "mode": "auto",
                    "status": "ready",
                    "doc_count": 1,
                    "chunk_count": 0,
                    "documents": {
                        "doc_1": {
                            "doc_id": "doc_1",
                            "file_path": "/a.md",
                            "status": "indexed",
                            "chunk_count": 0,
                        },
                    },
                },
            },
        )
        assert repo.get_library("lib_deg")["status"] == "degraded"

    def test_error_status_when_all_docs_error(self, tmp_path: Path):
        repo = self._seed(
            tmp_path,
            {
                "lib_err": {
                    "library_id": "lib_err",
                    "name": "Errors",
                    "mode": "auto",
                    "status": "ready",
                    "doc_count": 1,
                    "documents": {
                        "doc_1": {"doc_id": "doc_1", "file_path": "/x.md", "status": "error"},
                    },
                },
            },
        )
        assert repo.get_library("lib_err")["status"] == "error"


# ── Cancel-flag management ───────────────────────────────────────


class TestCancelFlags:
    """cancel_ingest / cancel_flags bookkeeping."""

    def test_cancel_sets_flag(self, repo: LibraryRepository):
        repo.cancel_ingest("lib_xyz")
        assert repo.cancel_flags["lib_xyz"] is True

    def test_cancel_flag_cleared_after_pop(self, repo: LibraryRepository):
        repo.cancel_ingest("lib_abc")
        repo.cancel_flags.pop("lib_abc", None)
        assert "lib_abc" not in repo.cancel_flags
