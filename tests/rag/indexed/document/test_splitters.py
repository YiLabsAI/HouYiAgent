from __future__ import annotations

import pytest

from houyi.rag.indexed.document.splitters import (
    MAX_CHUNK_CHARS,
    _recursive_split,
    _sentence_split,
    split_documents,
)
from houyi.rag.types import Document


class TestDocumentSplitters:
    @pytest.mark.asyncio
    async def test_recursive_split_basic(self) -> None:
        doc = Document(
            doc_id="test-doc",
            content="Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            source="test.txt",
        )

        chunks = _recursive_split(doc, chunk_size=50, chunk_overlap=10)

        assert len(chunks) >= 1
        assert all(c.doc_id == "test-doc" for c in chunks)
        combined = " ".join(c.content for c in chunks)
        assert "Paragraph one" in combined
        assert "Paragraph two" in combined

    @pytest.mark.asyncio
    async def test_recursive_split_small_chunk(self) -> None:
        doc = Document(
            doc_id="test-doc",
            content="This is a test sentence. Another sentence here.",
            source="test.txt",
        )

        chunks = _recursive_split(doc, chunk_size=20, chunk_overlap=5)

        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_recursive_split_large_content(self) -> None:
        content = "Word " * 1000
        doc = Document(doc_id="test-doc", content=content, source="test.txt")

        chunks = _recursive_split(doc, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1
        assert all(c.content.strip() for c in chunks)

    @pytest.mark.asyncio
    async def test_sentence_split_basic(self) -> None:
        doc = Document(
            doc_id="test-doc",
            content="First sentence. Second sentence. Third sentence.",
            source="test.txt",
        )

        chunks = _sentence_split(doc, chunk_size=40, chunk_overlap=5)

        assert len(chunks) >= 1
        combined = " ".join(c.content for c in chunks)
        assert "First sentence" in combined
        assert "Second sentence" in combined

    @pytest.mark.asyncio
    async def test_split_documents_recursive(self) -> None:
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
        doc = Document(doc_id="test-doc", content="Test content here.", source="test.txt")

        chunks = _recursive_split(doc, chunk_size=100, chunk_overlap=10)

        assert len(chunks) >= 1
        chunk = chunks[0]
        assert "chunk_index" in chunk.metadata
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test.txt"

    @pytest.mark.asyncio
    async def test_chunk_positions(self) -> None:
        doc = Document(
            doc_id="test-doc",
            content="First part. Second part. Third part.",
            source="test.txt",
        )

        chunks = _recursive_split(doc, chunk_size=15, chunk_overlap=2)

        assert chunks[0].start_idx == 0
        for chunk in chunks:
            assert chunk.start_idx >= 0
            assert chunk.end_idx > chunk.start_idx

    @pytest.mark.asyncio
    async def test_empty_document(self) -> None:
        doc = Document(doc_id="test-doc", content="", source="test.txt")

        chunks = _recursive_split(doc, chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_split_preserves_content(self) -> None:
        original = "This is the complete original content that should be preserved."
        doc = Document(doc_id="test-doc", content=original, source="test.txt")

        chunks = _recursive_split(doc, chunk_size=30, chunk_overlap=5)

        combined = " ".join(c.content for c in chunks)
        for word in ["complete", "original", "content", "preserved"]:
            assert word in combined

    def test_max_chunk_chars_is_safe(self) -> None:
        expected_max_tokens = MAX_CHUNK_CHARS / 4

        assert expected_max_tokens <= 2048, (
            f"MAX_CHUNK_CHARS={MAX_CHUNK_CHARS} may produce ~{expected_max_tokens} tokens, "
            f"exceeding the 2048 token limit"
        )
