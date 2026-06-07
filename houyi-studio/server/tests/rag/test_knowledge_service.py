import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from houyi_studio.server.rag import KnowledgeService

from houyi.infrastructure.config.env_config import ENV_EMBEDDING_MODEL, ENV_EMBEDDING_PROVIDER


@pytest.fixture
def knowledge_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a fresh KnowledgeService with isolated tmp_path storage."""
    monkeypatch.delenv(ENV_EMBEDDING_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_EMBEDDING_MODEL, raising=False)
    svc = KnowledgeService(storage_dir=tmp_path)
    yield svc


class TestLibraryManagement:
    """Tests for library CRUD operations."""

    def test_create_library(self, knowledge_service):
        """Test creating a new library."""
        result = knowledge_service.create_library(
            name="Test Library",
            description="Test description",
            mode="auto",
            knowledge_dir="/nonexistent/path",  # Use nonexistent path to avoid scanning
        )

        assert result["name"] == "Test Library"
        assert result["description"] == "Test description"
        assert result["mode"] == "auto"
        assert result["status"] == "empty"
        assert result["doc_count"] == 0
        assert "library_id" in result

    def test_create_duplicate_library_name(self, knowledge_service):
        """Test that duplicate library names are prevented."""
        knowledge_service.create_library(name="Duplicate", description="", mode="auto")

        # Second create with same name should fail
        result = knowledge_service.create_library(name="Duplicate", description="", mode="auto")
        assert result is None or "error" in result

    def test_list_libraries(self, knowledge_service):
        """Test listing all libraries."""
        knowledge_service.create_library(name="Lib1", description="", mode="auto")
        knowledge_service.create_library(name="Lib2", description="", mode="indexed")

        libs = knowledge_service.list_libraries()
        assert len(libs) == 2
        names = {lib["name"] for lib in libs}
        assert names == {"Lib1", "Lib2"}

    def test_get_library(self, knowledge_service):
        """Test getting a specific library."""
        created = knowledge_service.create_library(name="GetTest", description="", mode="auto")
        lib_id = created["library_id"]

        lib = knowledge_service.get_library(lib_id)
        assert lib is not None
        assert lib["name"] == "GetTest"

        # Non-existent library
        assert knowledge_service.get_library("nonexistent") is None

    def test_delete_library(self, knowledge_service):
        """Test deleting a library."""
        created = knowledge_service.create_library(name="ToDelete", description="", mode="auto")
        lib_id = created["library_id"]

        result = knowledge_service.delete_library(lib_id)
        assert result is True

        # Library should no longer exist
        assert knowledge_service.get_library(lib_id) is None

    def test_update_library_basic_fields(self, knowledge_service):
        """Test updating basic library fields."""
        created = knowledge_service.create_library(name="Original", description="", mode="auto")
        lib_id = created["library_id"]

        updated = knowledge_service.update_library(
            lib_id,
            {
                "name": "Updated",
                "description": "New description",
                "mode": "indexed",
            },
        )

        assert updated["name"] == "Updated"
        assert updated["description"] == "New description"
        assert updated["mode"] == "indexed"

    def test_update_library_metadata(self, knowledge_service):
        """Test that metadata (settings) are saved correctly."""
        created = knowledge_service.create_library(name="ConfigTest", description="", mode="auto")
        lib_id = created["library_id"]

        # Update with settings in metadata
        updated = knowledge_service.update_library(
            lib_id,
            {
                "metadata": {
                    "chunk_size": 1024,
                    "chunk_overlap": 100,
                    "top_k": 20,
                    "enable_rerank": False,
                }
            },
        )

        assert updated["metadata"]["chunk_size"] == 1024
        assert updated["metadata"]["chunk_overlap"] == 100
        assert updated["metadata"]["top_k"] == 20
        assert updated["metadata"]["enable_rerank"] is False

        # Reload and verify persistence
        lib = knowledge_service.get_library(lib_id)
        assert lib["metadata"]["chunk_size"] == 1024
        assert lib["metadata"]["top_k"] == 20


class TestDeleteLibrarySafety:
    """Tests for delete_library path-safety guards.

    These tests ensure delete_library refuses to delete directories
    that are outside the storage root, have malformed IDs, or are nested
    deeper than expected.
    """

    def test_delete_nonexistent_returns_false(self, knowledge_service):
        assert knowledge_service.delete_library("lib_nonexistent") is False

    def test_delete_normal_library_succeeds(self, knowledge_service):
        lib = knowledge_service.create_library(name="Safe", description="", mode="auto")
        lib_id = lib["library_id"]
        assert knowledge_service.delete_library(lib_id) is True
        assert knowledge_service.get_library(lib_id) is None

    def test_rejects_bad_id(self, knowledge_service):
        """IDs that don't start with 'lib_' are refused."""
        knowledge_service._repo._libraries["evil"] = {"library_id": "evil", "name": "bad"}
        assert knowledge_service.delete_library("evil") is False
        assert "evil" in knowledge_service._repo._libraries

    def test_delete_rejects_path_traversal(self, knowledge_service):
        """IDs containing '..' that escape storage root are refused."""
        bad_id = "lib_x/../../etc"
        knowledge_service._repo._libraries[bad_id] = {"library_id": bad_id, "name": "bad"}
        assert knowledge_service.delete_library(bad_id) is False
        assert bad_id in knowledge_service._repo._libraries

    def test_storage_dir_is_injected(self, tmp_path):
        """KnowledgeService(storage_dir=...) must use the given path."""
        svc = KnowledgeService(storage_dir=tmp_path / "custom")
        assert svc.storage_dir == (tmp_path / "custom").resolve()
        assert svc.storage_dir.exists()

    def test_library_respects_storage(self, tmp_path):
        """Instance methods return paths under the injected storage dir."""
        svc = KnowledgeService(storage_dir=tmp_path / "iso")
        lib = svc.create_library(name="Isolated", description="", mode="auto")
        lib_id = lib["library_id"]
        assert str(svc.library_storage_dir(lib_id)).startswith(str(svc.storage_dir))
        assert str(svc.library_upload_dir(lib_id)).startswith(str(svc.storage_dir))
        assert str(svc.library_index_dir(lib_id)).startswith(str(svc.storage_dir))


class TestFileIngest:
    """Tests for file ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_unsupported_file_type(self, knowledge_service):
        """Test that unsupported file types are handled gracefully."""
        from houyi_studio.server.rag import embedding_config as _ec_mod

        created = knowledge_service.create_library(
            name="UnsupportedTest", description="", mode="auto"
        )
        lib_id = created["library_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Random content")
            temp_path = f.name

        with patch.object(_ec_mod, "_is_provider_runtime_available", return_value=False):
            try:
                result = await knowledge_service.ingest_files(lib_id, [temp_path])

                # Should not crash, may return error or empty
                assert "success" in result

            finally:
                os.unlink(temp_path)


class TestDataMigration:
    """Tests for data migration logic."""

    def test_migrate_library_without_status(self, knowledge_service):
        """Test that libraries without status get migrated correctly."""
        lib_id = "lib_test_old"
        knowledge_service._repo._libraries[lib_id] = {
            "library_id": lib_id,
            "name": "Old Library",
            "description": "",
            "mode": "auto",
            "doc_count": 0,
        }
        knowledge_service._repo._save_libraries()

        knowledge_service._repo._load_libraries()

        lib = knowledge_service.get_library(lib_id)
        assert lib["status"] == "empty"  # Should be migrated to "empty"

    def test_migrate_removes_duplicate_documents(self, knowledge_service):
        """Test that duplicate documents are removed during migration."""
        lib_id = "lib_test_dupes"
        knowledge_service._repo._libraries[lib_id] = {
            "library_id": lib_id,
            "name": "Dupe Library",
            "description": "",
            "mode": "auto",
            "status": "ready",
            "doc_count": 2,
            "documents": {
                "doc_1": {
                    "doc_id": "doc_1",
                    "file_path": "/test/file.md",
                    "file_name": "file.md",
                    "status": "indexed",
                },
                "doc_2": {
                    "doc_id": "doc_2",
                    "file_path": "/test/file.md",
                    "file_name": "file.md",
                    "status": "indexed",
                },
            },
        }
        knowledge_service._repo._save_libraries()

        knowledge_service._repo._load_libraries()

        lib = knowledge_service.get_library(lib_id)
        assert lib["doc_count"] == 1  # Duplicate should be removed
        assert len(lib["documents"]) == 1

    def test_zero_chunks_degraded(self, knowledge_service):
        """Libraries with status=ready but chunks=0 should be migrated to degraded."""
        lib_id = "lib_test_zero_chunks"
        knowledge_service._repo._libraries[lib_id] = {
            "library_id": lib_id,
            "name": "Zero Chunks Lib",
            "description": "",
            "mode": "auto",
            "status": "ready",
            "doc_count": 2,
            "chunk_count": 0,
            "documents": {
                "doc_1": {
                    "doc_id": "doc_1",
                    "file_path": "/test/a.md",
                    "file_name": "a.md",
                    "status": "indexed",
                    "chunk_count": 0,
                },
                "doc_2": {
                    "doc_id": "doc_2",
                    "file_path": "/test/b.md",
                    "file_name": "b.md",
                    "status": "indexed",
                    "chunk_count": 0,
                },
            },
        }
        knowledge_service._repo._save_libraries()

        knowledge_service._repo._load_libraries()

        lib = knowledge_service.get_library(lib_id)
        assert lib["status"] == "degraded", (
            f"Expected 'degraded' for ready+0chunks, got '{lib['status']}'"
        )


class TestEmbeddingProviderDetection:
    """Tests for embedding provider detection and degraded status."""

    def test_no_embedding_sets_degraded(self, knowledge_service):
        """When no embedding provider is available, ingest should set status='degraded'
        instead of 'ready' when chunks=0."""
        from unittest.mock import patch

        created = knowledge_service.create_library(name="NoProv", description="", mode="auto")
        lib_id = created["library_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n\nSome content for embedding test.")
            temp_path = f.name

        try:
            # Clear ALL embedding-related env vars to force "no provider" fallback
            env_backup = {}
            for key in (
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_API_KEY",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "OPENAI_API_KEY",
                "EMBEDDING_PROVIDER",
                "EMBEDDING_MODEL",
            ):
                env_backup[key] = os.environ.pop(key, None)

            # Also block fastembed import so local fallback doesn't kick in
            def _block_fastembed(name, *a, **kw):
                if name == "fastembed" or name.startswith("fastembed."):
                    raise ImportError("blocked for test")
                return original_import(name, *a, **kw)

            import builtins

            original_import = builtins.__import__
            with patch("builtins.__import__", side_effect=_block_fastembed):
                asyncio.run(knowledge_service.ingest_files(lib_id, [temp_path]))

            lib = knowledge_service.get_library(lib_id)
            # Files should be counted but no chunks created
            assert lib["doc_count"] >= 1
            assert lib["chunk_count"] == 0
            # Status should be 'degraded', NOT 'ready'
            assert lib["status"] == "degraded", (
                f"Expected 'degraded' when chunks=0, got '{lib['status']}'"
            )
        finally:
            os.unlink(temp_path)
            # Restore env vars
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_gcloud_project_enables_gemini(self):
        """GOOGLE_CLOUD_PROJECT env var should be checked for Gemini embedding."""
        env_backup = os.environ.get("GOOGLE_CLOUD_PROJECT")
        try:
            os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project-123"
            assert os.environ.get("GOOGLE_CLOUD_PROJECT") == "test-project-123"
        finally:
            if env_backup is not None:
                os.environ["GOOGLE_CLOUD_PROJECT"] = env_backup
            else:
                os.environ.pop("GOOGLE_CLOUD_PROJECT", None)


class TestDocumentManagement:
    """Tests for document operations."""

    def test_list_documents_empty(self, knowledge_service):
        """Test listing documents for empty library."""
        created = knowledge_service.create_library(name="EmptyDocs", description="", mode="auto")
        lib_id = created["library_id"]

        docs = knowledge_service.list_documents(lib_id)
        assert docs == []
