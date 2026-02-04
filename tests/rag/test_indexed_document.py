"""Tests for indexed document loaders and splitters."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from houyi.rag.indexed.document.loaders import (
    _load_html_file,
    _load_single_file,
    _load_text_file,
    load_documents,
)
from houyi.rag.indexed.document.splitters import (
    _recursive_split,
    _sentence_split,
    split_documents,
)
from houyi.rag.types import Document


class TestDocumentLoaders:
    """Tests for document loading functions."""

    @pytest.mark.asyncio
    async def test_load_text_file(self) -> None:
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello, World!")
            f.flush()
            path = Path(f.name)

        try:
            doc = _load_text_file(path)
            assert doc.content == "Hello, World!"
            assert doc.doc_type == "text"
            assert doc.source == str(path)
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_markdown_file(self) -> None:
        """Test loading a markdown file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Title\n\nContent here")
            f.flush()
            path = Path(f.name)

        try:
            doc = await _load_single_file(path)
            assert doc is not None
            assert "# Title" in doc.content
            assert doc.doc_type == "text"
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_html_file(self) -> None:
        """Test loading and parsing HTML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as f:
            f.write("<html><body><h1>Title</h1><p>Content</p></body></html>")
            f.flush()
            path = Path(f.name)

        try:
            doc = _load_html_file(path)
            assert "Title" in doc.content
            assert "Content" in doc.content
            assert "<h1>" not in doc.content  # Tags should be removed
            assert doc.doc_type == "html"
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_html_with_script(self) -> None:
        """Test that script tags are removed from HTML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as f:
            f.write(
                "<html><script>alert('bad')</script>"
                "<body>Safe content</body></html>"
            )
            f.flush()
            path = Path(f.name)

        try:
            doc = _load_html_file(path)
            assert "alert" not in doc.content
            assert "Safe content" in doc.content
        finally:
            path.unlink()

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self) -> None:
        """Test loading a file that doesn't exist."""
        doc = await _load_single_file(Path("/nonexistent/file.txt"))
        assert doc is None

    @pytest.mark.asyncio
    async def test_load_documents_from_directory(self) -> None:
        """Test loading documents from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "doc1.txt").write_text("Document 1")
            (Path(tmpdir) / "doc2.md").write_text("# Document 2")

            # Create subdirectory with files
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "doc3.txt").write_text("Document 3")

            docs = await load_documents([tmpdir])

            assert len(docs) == 3
            contents = [d.content for d in docs]
            assert any("Document 1" in c for c in contents)
            assert any("# Document 2" in c for c in contents)
            assert any("Document 3" in c for c in contents)

    @pytest.mark.asyncio
    async def test_load_documents_from_files(self) -> None:
        """Test loading documents from specific file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "doc1.txt"
            file2 = Path(tmpdir) / "doc2.txt"
            file1.write_text("Content 1")
            file2.write_text("Content 2")

            docs = await load_documents([str(file1), str(file2)])

            assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_load_unsupported_file_type(self) -> None:
        """Test that unsupported file types return None."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("Unknown format")
            f.flush()
            path = Path(f.name)

        try:
            doc = await _load_single_file(path)
            assert doc is None
        finally:
            path.unlink()


class TestDocumentSplitters:
    """Tests for document splitting functions."""

    @pytest.mark.asyncio
    async def test_recursive_split_basic(self) -> None:
        """Test basic recursive splitting."""
        doc = Document(
            doc_id="test-doc",
            content="Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            source="test.txt",
        )

        chunks = _recursive_split(doc, chunk_size=50, chunk_overlap=10)

        assert len(chunks) >= 1
        assert all(c.doc_id == "test-doc" for c in chunks)
        # All content should be preserved
        combined = " ".join(c.content for c in chunks)
        assert "Paragraph one" in combined
        assert "Paragraph two" in combined

    @pytest.mark.asyncio
    async def test_recursive_split_small_chunk(self) -> None:
        """Test splitting with small chunk size."""
        doc = Document(
            doc_id="test-doc",
            content="This is a test sentence. Another sentence here.",
            source="test.txt",
        )

        chunks = _recursive_split(doc, chunk_size=20, chunk_overlap=5)

        # Should create multiple small chunks
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_recursive_split_large_content(self) -> None:
        """Test splitting large content."""
        content = "Word " * 1000  # Large content
        doc = Document(doc_id="test-doc", content=content, source="test.txt")

        chunks = _recursive_split(doc, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1
        # All chunks should have content
        assert all(c.content.strip() for c in chunks)

    @pytest.mark.asyncio
    async def test_sentence_split_basic(self) -> None:
        """Test basic sentence splitting."""
        doc = Document(
            doc_id="test-doc",
            content="First sentence. Second sentence. Third sentence.",
            source="test.txt",
        )

        chunks = _sentence_split(doc, chunk_size=40, chunk_overlap=5)

        assert len(chunks) >= 1
        # Content should be preserved
        combined = " ".join(c.content for c in chunks)
        assert "First sentence" in combined
        assert "Second sentence" in combined

    @pytest.mark.asyncio
    async def test_split_documents_recursive(self) -> None:
        """Test split_documents with recursive strategy."""
        docs = [
            Document(doc_id="doc1", content="Content one.\n\nMore content.", source="1.txt"),
            Document(doc_id="doc2", content="Content two.", source="2.txt"),
        ]

        chunks = await split_documents(docs, chunk_size=100, strategy="recursive")

        assert len(chunks) >= 2
        doc_ids = {c.doc_id for c in chunks}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    @pytest.mark.asyncio
    async def test_split_documents_sentence(self) -> None:
        """Test split_documents with sentence strategy."""
        docs = [
            Document(
                doc_id="doc1",
                content="Sentence one. Sentence two! Sentence three?",
                source="1.txt",
            )
        ]

        chunks = await split_documents(docs, chunk_size=30, strategy="sentence")

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_chunk_metadata(self) -> None:
        """Test that chunk metadata is properly set."""
        doc = Document(doc_id="test-doc", content="Test content here.", source="test.txt")

        chunks = _recursive_split(doc, chunk_size=100, chunk_overlap=10)

        assert len(chunks) >= 1
        chunk = chunks[0]
        assert "chunk_index" in chunk.metadata
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test.txt"

    @pytest.mark.asyncio
    async def test_chunk_positions(self) -> None:
        """Test that chunk positions are tracked."""
        doc = Document(
            doc_id="test-doc",
            content="First part. Second part. Third part.",
            source="test.txt",
        )

        chunks = _recursive_split(doc, chunk_size=15, chunk_overlap=2)

        # First chunk should start at 0
        assert chunks[0].start_idx == 0
        # All chunks should have valid positions
        for chunk in chunks:
            assert chunk.start_idx >= 0
            assert chunk.end_idx > chunk.start_idx

    @pytest.mark.asyncio
    async def test_empty_document(self) -> None:
        """Test splitting empty document."""
        doc = Document(doc_id="test-doc", content="", source="test.txt")

        chunks = _recursive_split(doc, chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_split_preserves_content(self) -> None:
        """Test that splitting preserves all original content."""
        original = "This is the complete original content that should be preserved."
        doc = Document(doc_id="test-doc", content=original, source="test.txt")

        chunks = _recursive_split(doc, chunk_size=30, chunk_overlap=5)

        # Join all chunks and check key words are present
        combined = " ".join(c.content for c in chunks)
        for word in ["complete", "original", "content", "preserved"]:
            assert word in combined
