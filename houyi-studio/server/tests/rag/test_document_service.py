"""Tests for DocumentService — document CRUD, status transitions, and chunk preview."""

from __future__ import annotations

from pathlib import Path

import pytest
from houyi_studio.server.rag.document_service import DocumentService
from houyi_studio.server.rag.library_repository import LibraryRepository


@pytest.fixture
def repo(tmp_path: Path) -> LibraryRepository:
    """Isolated repository for document tests."""
    return LibraryRepository(storage_dir=tmp_path)


@pytest.fixture
def library_id(repo: LibraryRepository) -> str:
    """Pre-populated library to host documents."""
    lib = repo.create_library(name="DocTests", knowledge_dir="/nonexistent")
    return lib["library_id"]


@pytest.fixture
def svc(repo: LibraryRepository) -> DocumentService:
    """DocumentService wired to the test repository."""
    return DocumentService(repo)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A small temp file that add_document can stat."""
    f = tmp_path / "sample.md"
    f.write_text("# Hello\nWorld.")
    return f


# ── List / get / add ─────────────────────────────────────────────


class TestDocumentCRUD:
    """Basic document creation and retrieval."""

    def test_list_empty(self, svc: DocumentService, library_id: str):
        assert svc.list_documents(library_id) == []

    def test_list_unknown_library(self, svc: DocumentService):
        assert svc.list_documents("lib_missing") == []

    def test_add_and_get(
        self,
        svc: DocumentService,
        library_id: str,
        sample_file: Path,
    ):
        doc = svc.add_document(library_id, str(sample_file))
        assert doc is not None
        assert doc["filename"] == "sample.md"
        assert doc["status"] == "pending"
        assert doc["file_size"] > 0

        fetched = svc.get_document(library_id, doc["doc_id"])
        assert fetched["doc_id"] == doc["doc_id"]

    def test_add_nonexistent_file_returns_none(
        self,
        svc: DocumentService,
        library_id: str,
    ):
        assert svc.add_document(library_id, "/no/such/file.md") is None

    def test_add_to_nonexistent_library_returns_none(
        self,
        svc: DocumentService,
        sample_file: Path,
    ):
        assert svc.add_document("lib_nope", str(sample_file)) is None

    def test_get_nonexistent_doc_returns_none(
        self,
        svc: DocumentService,
        library_id: str,
    ):
        assert svc.get_document(library_id, "doc_nope") is None

    def test_list_after_add(
        self,
        svc: DocumentService,
        library_id: str,
        sample_file: Path,
    ):
        svc.add_document(library_id, str(sample_file))
        docs = svc.list_documents(library_id)
        assert len(docs) == 1


# ── Status updates ───────────────────────────────────────────────


class TestStatusUpdates:
    """update_document_status, enable, disable."""

    @pytest.fixture(autouse=True)
    def _add_doc(
        self,
        svc: DocumentService,
        library_id: str,
        sample_file: Path,
    ):
        doc = svc.add_document(library_id, str(sample_file))
        self.doc_id = doc["doc_id"]

    def test_update_status(self, svc: DocumentService, library_id: str):
        updated = svc.update_document_status(library_id, self.doc_id, "indexed", chunk_count=42)
        assert updated["status"] == "indexed"
        assert updated["chunk_count"] == 42

    def test_error_message_attached(self, svc: DocumentService, library_id: str):
        updated = svc.update_document_status(
            library_id,
            self.doc_id,
            "error",
            error_message="boom",
        )
        assert updated["error_message"] == "boom"

    def test_error_message_cleared_on_non_error(self, svc: DocumentService, library_id: str):
        svc.update_document_status(library_id, self.doc_id, "error", error_message="fail")
        updated = svc.update_document_status(library_id, self.doc_id, "indexed")
        assert "error_message" not in updated

    def test_disable_and_enable(self, svc: DocumentService, library_id: str):
        disabled = svc.disable_document(library_id, self.doc_id)
        assert disabled["status"] == "disabled"

        enabled = svc.enable_document(library_id, self.doc_id)
        assert enabled["status"] == "indexed"

    def test_update_nonexistent_doc_returns_none(self, svc: DocumentService, library_id: str):
        assert svc.update_document_status(library_id, "doc_nope", "indexed") is None


# ── Delete and stats recalculation ───────────────────────────────


class TestDeleteDocument:
    """delete_document removes the doc and recalculates library stats."""

    def test_delete_last_doc_sets_empty(
        self,
        svc: DocumentService,
        repo: LibraryRepository,
        library_id: str,
        sample_file: Path,
    ):
        doc = svc.add_document(library_id, str(sample_file))
        assert svc.delete_document(library_id, doc["doc_id"]) is True

        lib = repo.get_library(library_id)
        assert lib["status"] == "empty"
        assert lib["doc_count"] == 0

    def test_delete_nonexistent_returns_false(self, svc: DocumentService, library_id: str):
        assert svc.delete_document(library_id, "doc_ghost") is False

    def test_stats_recalculated(
        self,
        svc: DocumentService,
        repo: LibraryRepository,
        library_id: str,
        tmp_path: Path,
    ):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("aaa")
        f2.write_text("bbb")

        d1 = svc.add_document(library_id, str(f1))
        d2 = svc.add_document(library_id, str(f2))
        svc.update_document_status(library_id, d1["doc_id"], "indexed", chunk_count=5)
        svc.update_document_status(library_id, d2["doc_id"], "indexed", chunk_count=3)

        svc.delete_document(library_id, d1["doc_id"])
        lib = repo.get_library(library_id)
        assert lib["doc_count"] == 1
        assert lib["chunk_count"] == 3
        assert lib["status"] == "ready"


# ── Retrieval counter ────────────────────────────────────────────


class TestRetrievalCount:
    """increment_retrieval_count bumps the counter."""

    def test_increment(
        self,
        svc: DocumentService,
        library_id: str,
        sample_file: Path,
    ):
        doc = svc.add_document(library_id, str(sample_file))
        assert doc["retrieval_count"] == 0

        svc.increment_retrieval_count(library_id, doc["doc_id"])
        svc.increment_retrieval_count(library_id, doc["doc_id"])

        fetched = svc.get_document(library_id, doc["doc_id"])
        assert fetched["retrieval_count"] == 2

    def test_increment_unknown_library_is_noop(self, svc: DocumentService):
        svc.increment_retrieval_count("lib_nope", "doc_nope")

    def test_increment_unknown_doc_is_noop(
        self,
        svc: DocumentService,
        library_id: str,
    ):
        svc.increment_retrieval_count(library_id, "doc_nope")


# ── Chunk helpers ────────────────────────────────────────────────


class TestListChunks:
    """list_chunks returns stored chunk arrays."""

    def test_empty_when_no_chunks(
        self,
        svc: DocumentService,
        library_id: str,
        sample_file: Path,
    ):
        doc = svc.add_document(library_id, str(sample_file))
        assert svc.list_chunks(library_id, doc["doc_id"]) == []

    def test_unknown_library_returns_empty(self, svc: DocumentService):
        assert svc.list_chunks("lib_nope", "doc_x") == []

    def test_unknown_doc_returns_empty(self, svc: DocumentService, library_id: str):
        assert svc.list_chunks(library_id, "doc_nope") == []


class TestPreviewChunks:
    """preview_chunks splits content for the UI."""

    def test_recursive_strategy(self, svc: DocumentService):
        content = "a" * 100
        chunks = svc.preview_chunks(content, chunk_size=40, chunk_overlap=10)
        assert len(chunks) >= 2
        assert all("index" in c and "char_count" in c for c in chunks)

    def test_sentence_strategy(self, svc: DocumentService):
        content = "First sentence. Second sentence. Third sentence."
        chunks = svc.preview_chunks(content, chunk_size=30, strategy="sentence")
        assert len(chunks) >= 2

    def test_truncation_at_200_chars(self, svc: DocumentService):
        content = "x" * 500
        chunks = svc.preview_chunks(content, chunk_size=500)
        assert chunks[0]["content"].endswith("...")
        assert chunks[0]["char_count"] == 500

    def test_short_content_no_truncation(self, svc: DocumentService):
        content = "tiny"
        chunks = svc.preview_chunks(content, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "tiny"
        assert not chunks[0]["content"].endswith("...")
