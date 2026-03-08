from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from houyi_studio.server.rag import KnowledgeService

from houyi.infrastructure.config.env_config import (
    ENV_EMBEDDING_MODEL,
    ENV_EMBEDDING_PROVIDER,
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_APPLICATION_CREDENTIALS,
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_OPENAI_API_KEY,
)


@pytest.fixture
def knowledge_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_EMBEDDING_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_EMBEDDING_MODEL, raising=False)
    monkeypatch.delenv(ENV_GOOGLE_API_KEY, raising=False)
    monkeypatch.delenv(ENV_GOOGLE_APPLICATION_CREDENTIALS, raising=False)
    monkeypatch.delenv(ENV_GOOGLE_CLOUD_PROJECT, raising=False)
    monkeypatch.delenv(ENV_OPENAI_API_KEY, raising=False)
    svc = KnowledgeService(storage_dir=tmp_path)
    yield svc


class TestFileIngest:
    @pytest.mark.asyncio
    async def test_uploaded_files_are_not_skipped(self, knowledge_service):
        from houyi_studio.server.rag import get_library_upload_dir

        created = knowledge_service.create_library(
            name="UploadFilter",
            description="",
            mode="indexed",
        )
        lib_id = created["library_id"]

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
        assert stats.get("files_skipped", 0) == 0
        files_seen = stats.get("files_processed", 0) + stats.get("files_failed", 0)
        assert files_seen > 0

    def test_ingest_single_file(self, knowledge_service):
        created = knowledge_service.create_library(name="IngestTest", description="", mode="auto")
        lib_id = created["library_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n\nThis is test content.")
            temp_path = f.name

        try:
            result = asyncio.run(knowledge_service.ingest_files(lib_id, [temp_path]))

            assert (
                result["success"] is True
                or result["stats"]["files_processed"] > 0
                or result["stats"]["files_failed"] > 0
            )

            docs = knowledge_service.list_documents(lib_id)
            assert len(docs) >= 1

            doc = next((d for d in docs if d["file_name"] == os.path.basename(temp_path)), None)
            assert doc is not None

        finally:
            os.unlink(temp_path)

    def test_ingest_duplicate_file_no_duplication(self, knowledge_service):
        created = knowledge_service.create_library(name="DedupTest", description="", mode="auto")
        lib_id = created["library_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Duplicate Test\n\nContent.")
            temp_path = f.name

        try:
            asyncio.run(knowledge_service.ingest_files(lib_id, [temp_path]))
            docs_after_first = knowledge_service.list_documents(lib_id)

            asyncio.run(knowledge_service.ingest_files(lib_id, [temp_path]))
            docs_after_second = knowledge_service.list_documents(lib_id)

            assert len(docs_after_second) == len(docs_after_first)

        finally:
            os.unlink(temp_path)

    def test_ingest_multiple_files(self, knowledge_service):
        created = knowledge_service.create_library(name="MultiTest", description="", mode="auto")
        lib_id = created["library_id"]

        temp_files = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                    f.write(f"# Doc {i}\n\nContent {i}.")
                    temp_files.append(f.name)

            asyncio.run(knowledge_service.ingest_files(lib_id, temp_files))

            docs = knowledge_service.list_documents(lib_id)
            assert len(docs) == 3

        finally:
            for f in temp_files:
                os.unlink(f)

    def test_library_status_after_ingest(self, knowledge_service):
        created = knowledge_service.create_library(name="StatusTest", description="", mode="auto")
        lib_id = created["library_id"]

        lib = knowledge_service.get_library(lib_id)
        assert lib["status"] == "empty"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n\nContent.")
            temp_path = f.name

        try:
            asyncio.run(knowledge_service.ingest_files(lib_id, [temp_path]))

            lib = knowledge_service.get_library(lib_id)
            assert lib["status"] in ["ready", "degraded", "error", "partial"]
            assert lib["doc_count"] >= 1

        finally:
            os.unlink(temp_path)


class TestDocumentManagement:
    def test_list_documents_with_files(self, knowledge_service):
        created = knowledge_service.create_library(name="DocsTest", description="", mode="auto")
        lib_id = created["library_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n\nContent.")
            temp_path = f.name

        try:
            asyncio.run(knowledge_service.ingest_files(lib_id, [temp_path]))

            docs = knowledge_service.list_documents(lib_id)
            assert len(docs) >= 1

            doc = docs[0]
            assert "doc_id" in doc
            assert "file_name" in doc
            assert "file_path" in doc
            assert "status" in doc

        finally:
            os.unlink(temp_path)
