"""Tests for knowledge file upload API."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from houyi_studio.server.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_library():
    """Create a mock library with knowledge_dir."""
    return {
        "library_id": "lib_test123",
        "name": "Test Library",
        "knowledge_dir": tempfile.mkdtemp(),
        "mode": "indexed",
    }


class TestKnowledgeUpload:
    """Tests for /api/knowledge/{library_id}/upload endpoint."""

    def test_upload_returns_404_for_nonexistent_library(self, client):
        """Should return 404 when library doesn't exist."""
        with patch("houyi_studio.server.app.get_knowledge_service") as mock_service:
            mock_service.return_value.get_library.return_value = None

            response = client.post(
                "/api/knowledge/nonexistent/upload",
                files={"files": ("test.md", b"# Test content", "text/markdown")},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["error"]

    def test_upload_saves_file_to_knowledge_dir(self, client, mock_library):
        """Should save uploaded file to library's knowledge_dir/uploads/."""
        with patch("houyi_studio.server.app.get_knowledge_service") as mock_service:
            mock_service.return_value.get_library.return_value = mock_library

            response = client.post(
                f"/api/knowledge/{mock_library['library_id']}/upload",
                files={"files": ("test.md", b"# Test content", "text/markdown")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["library_id"] == mock_library["library_id"]
            assert len(data["uploaded_paths"]) == 1
            assert "test.md" in data["uploaded_paths"][0]

            # Verify file was actually saved
            upload_path = Path(data["uploaded_paths"][0])
            assert upload_path.exists()
            assert upload_path.read_text() == "# Test content"

    def test_upload_creates_upload_directory(self, client, mock_library):
        """Should create uploads directory if it doesn't exist."""
        with patch("houyi_studio.server.app.get_knowledge_service") as mock_service:
            mock_service.return_value.get_library.return_value = mock_library

            upload_dir = Path(mock_library["knowledge_dir"]) / "uploads"
            assert not upload_dir.exists()

            response = client.post(
                f"/api/knowledge/{mock_library['library_id']}/upload",
                files={"files": ("test.md", b"content", "text/markdown")},
            )

            assert response.status_code == 200
            assert upload_dir.exists()

    def test_upload_multiple_files(self, client, mock_library):
        """Should handle multiple file uploads."""
        with patch("houyi_studio.server.app.get_knowledge_service") as mock_service:
            mock_service.return_value.get_library.return_value = mock_library

            response = client.post(
                f"/api/knowledge/{mock_library['library_id']}/upload",
                files=[
                    ("files", ("file1.md", b"# File 1", "text/markdown")),
                    ("files", ("file2.md", b"# File 2", "text/markdown")),
                    ("files", ("file3.txt", b"File 3 content", "text/plain")),
                ],
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["uploaded_paths"]) == 3

    def test_upload_sanitizes_filename(self, client, mock_library):
        """Should sanitize filename to prevent path traversal."""
        with patch("houyi_studio.server.app.get_knowledge_service") as mock_service:
            mock_service.return_value.get_library.return_value = mock_library

            response = client.post(
                f"/api/knowledge/{mock_library['library_id']}/upload",
                files={"files": ("../../../etc/passwd", b"malicious", "text/plain")},
            )

            assert response.status_code == 200
            data = response.json()
            # Should only use basename, not full path
            assert len(data["uploaded_paths"]) == 1
            assert "passwd" in data["uploaded_paths"][0]
            assert "../" not in data["uploaded_paths"][0]
