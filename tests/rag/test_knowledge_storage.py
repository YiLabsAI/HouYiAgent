"""Tests for knowledge library storage isolation and file handling."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "houyi-studio/server")

# Use isolated temp directory for tests
_TEST_STORAGE_DIR = tempfile.mkdtemp()
os.environ["HOUYI_KNOWLEDGE_STORAGE"] = _TEST_STORAGE_DIR


class TestStorageConstants:
    """Test that storage paths use constants, not hardcoded strings."""

    def test_constants_defined(self):
        """Verify all path constants are defined."""
        from houyi_studio.server.rag import (
            INDEX_SUBDIR,
            KNOWLEDGE_STORAGE_DIR,
            UPLOADS_SUBDIR,
        )

        assert UPLOADS_SUBDIR == "uploads"
        assert INDEX_SUBDIR == "index"
        assert KNOWLEDGE_STORAGE_DIR.is_absolute()

    def test_helper_functions_use_constants(self):
        """Verify helper functions use constants."""
        from houyi_studio.server.rag import (
            INDEX_SUBDIR,
            UPLOADS_SUBDIR,
            get_library_index_dir,
            get_library_upload_dir,
        )

        lib_id = "test_lib"
        upload_dir = get_library_upload_dir(lib_id)
        index_dir = get_library_index_dir(lib_id)

        assert upload_dir.name == UPLOADS_SUBDIR
        assert index_dir.name == INDEX_SUBDIR


class TestFileFiltering:
    """Test that ingest correctly filters files."""

    @pytest.fixture
    def knowledge_service(self):
        """Create a KnowledgeService instance."""
        from houyi_studio.server.rag import KnowledgeService

        return KnowledgeService()

    @pytest.fixture
    def test_library(self, knowledge_service):
        """Create a test library and clean up after."""
        lib = knowledge_service.create_library(
            name="Test Library",
            description="For testing",
            mode="indexed",
        )
        yield lib
        knowledge_service.delete_library(lib["library_id"])

    @pytest.mark.asyncio
    async def test_uploaded_files_not_skipped(self, knowledge_service, test_library):
        """Files in uploads/ directory should NOT be skipped during ingest.

        This test verifies the *file-filtering* logic only: the file must not
        be classified as ``skipped``.
        """
        from houyi_studio.server.rag import get_library_upload_dir

        lib_id = test_library["library_id"]

        upload_dir = get_library_upload_dir(lib_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        test_file = upload_dir / "test.md"
        test_file.write_text("# Test\n\nThis is test content.")

        assert "/uploads/" in str(test_file) or "\\uploads\\" in str(test_file)

        result = await knowledge_service.ingest_files(
            library_id=lib_id,
            paths=[str(test_file)],
        )

        stats = result.get("stats", {})
        assert stats.get("files_skipped", 0) == 0, (
            f"Uploaded file should not be skipped. Result: {result}"
        )
        files_seen = stats.get("files_processed", 0) + stats.get("files_failed", 0)
        assert files_seen > 0, (
            f"Uploaded file was not seen by the ingest pipeline. Result: {result}"
        )

    @pytest.mark.asyncio
    async def test_index_files_are_skipped(self, knowledge_service, test_library):
        """Files in production .houyi/*/index/ directory SHOULD be skipped.

        Note: This test verifies the is_index_path function correctly identifies
        index paths. The function uses string matching to handle cross-platform paths.
        """
        import os
        from pathlib import Path

        from houyi_studio.server.rag import is_index_path

        # Test that is_index_path correctly identifies production index paths
        # Use os.sep for cross-platform compatibility
        index_path = Path(
            os.path.join("/home", ".houyi", "knowledge", "lib_001", "index", "vectors.bin")
        )
        assert is_index_path(index_path), f"Should identify index path: {index_path}"

        index_path2 = Path(
            os.path.join("/Users", "test", ".houyi", "knowledge", "lib_001", "index", "meta.json")
        )
        assert is_index_path(index_path2), f"Should identify index path: {index_path2}"

        # Upload paths should NOT be marked as index
        upload_path = Path(
            os.path.join("/home", ".houyi", "knowledge", "lib_001", "uploads", "doc.md")
        )
        assert not is_index_path(upload_path), f"Upload path should not be index: {upload_path}"

        regular_path = Path(os.path.join("/home", "user", "documents", "readme.md"))
        assert not is_index_path(regular_path), f"Regular path should not be index: {regular_path}"

        # Just having /index/ in the path isn't enough - needs .houyi too
        random_index = Path(os.path.join("/some", "random", "index", "file.txt"))
        assert not is_index_path(random_index), f"Random index should not match: {random_index}"


class TestLibraryIsolation:
    """Test that libraries are properly isolated."""

    def test_each_library_has_own_storage(self):
        """Each library should have its own storage directory."""
        from houyi_studio.server.rag import (
            get_library_index_dir,
            get_library_storage_dir,
            get_library_upload_dir,
        )

        lib1_storage = get_library_storage_dir("lib_001")
        lib2_storage = get_library_storage_dir("lib_002")

        # Different libraries, different paths
        assert lib1_storage != lib2_storage
        assert "lib_001" in str(lib1_storage)
        assert "lib_002" in str(lib2_storage)

        # Uploads and index are under library storage
        assert get_library_upload_dir("lib_001").parent == lib1_storage
        assert get_library_index_dir("lib_001").parent == lib1_storage


class TestLibraryDeletion:
    """Test that library deletion cleans up all files."""

    def test_delete_library_removes_all_storage(self):
        """Deleting a library should remove uploads, index, and all storage."""
        from houyi_studio.server.rag import (
            KnowledgeService,
            get_library_index_dir,
            get_library_storage_dir,
            get_library_upload_dir,
        )

        service = KnowledgeService()

        # Create library
        lib = service.create_library(
            name="Delete Test",
            description="Test deletion cleanup",
            mode="indexed",
        )
        lib_id = lib["library_id"]

        # Create files in both uploads and index directories
        upload_dir = get_library_upload_dir(lib_id)
        index_dir = get_library_index_dir(lib_id)
        storage_dir = get_library_storage_dir(lib_id)

        upload_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        (upload_dir / "test.md").write_text("# Test")
        (index_dir / "test.json").write_text("{}")

        # Verify files exist
        assert (upload_dir / "test.md").exists()
        assert (index_dir / "test.json").exists()
        assert storage_dir.exists()

        # Delete library
        result = service.delete_library(lib_id)
        assert result is True

        # Verify ALL storage is deleted
        assert not storage_dir.exists(), f"Storage dir should be deleted: {storage_dir}"
        assert not upload_dir.exists(), f"Upload dir should be deleted: {upload_dir}"
        assert not index_dir.exists(), f"Index dir should be deleted: {index_dir}"

        print(f"✓ Library {lib_id} and all its storage deleted successfully")

    def test_create_library_creates_directories(self):
        """Creating a library should immediately create storage directories."""
        from houyi_studio.server.rag import (
            KnowledgeService,
            get_library_index_dir,
            get_library_storage_dir,
            get_library_upload_dir,
        )

        service = KnowledgeService()

        # Create library
        lib = service.create_library(
            name="Dir Test",
            description="Test directory creation",
            mode="indexed",
        )
        lib_id = lib["library_id"]

        try:
            # Directories should exist immediately after creation
            storage_dir = get_library_storage_dir(lib_id)
            upload_dir = get_library_upload_dir(lib_id)
            index_dir = get_library_index_dir(lib_id)

            assert storage_dir.exists(), f"Storage dir should exist: {storage_dir}"
            assert upload_dir.exists(), f"Upload dir should exist: {upload_dir}"
            assert index_dir.exists(), f"Index dir should exist: {index_dir}"

            print("✓ Library directories created on creation")
        finally:
            service.delete_library(lib_id)
