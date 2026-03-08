"""Tests for indexed sparse index."""

from __future__ import annotations

import importlib.util
import tempfile

import pytest

from houyi.rag.types import Chunk

BM25S_AVAILABLE = importlib.util.find_spec("bm25s") is not None


class TestSparseIndex:
    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_basic(self) -> None:
        from houyi.rag.indexed.index.sparse import SparseIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = SparseIndex(knowledge_dir=tmpdir)
            await index.load()

            chunks = [
                Chunk(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    content="Python is a programming language",
                    metadata={"source": "python.txt", "chunk_index": 0},
                ),
            ]
            await index.add_batch(chunks)

            assert index.count() == 1

    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_search(self) -> None:
        from houyi.rag.indexed.index.sparse import SparseIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = SparseIndex(knowledge_dir=tmpdir)
            await index.load()

            chunks = [
                Chunk(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    content="Python is a great programming language for beginners",
                    metadata={"source": "python.txt", "chunk_index": 0},
                ),
                Chunk(
                    chunk_id="chunk-2",
                    doc_id="doc-2",
                    content="Java is widely used in enterprise applications",
                    metadata={"source": "java.txt", "chunk_index": 0},
                ),
                Chunk(
                    chunk_id="chunk-3",
                    doc_id="doc-3",
                    content="Python has excellent libraries for machine learning",
                    metadata={"source": "ml.txt", "chunk_index": 0},
                ),
            ]
            await index.add_batch(chunks)

            results = await index.search("Python programming", k=3)

            assert len(results) >= 1
            assert any("Python" in r.content for r in results[:2])

    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_persistence(self) -> None:
        from houyi.rag.indexed.index.sparse import SparseIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index1 = SparseIndex(knowledge_dir=tmpdir)
            await index1.load()

            chunks = [
                Chunk(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    content="Persistent search content",
                    metadata={"source": "test.txt", "chunk_index": 0},
                ),
            ]
            await index1.add_batch(chunks)
            await index1.save()

            index2 = SparseIndex(knowledge_dir=tmpdir)
            await index2.load()

            assert index2.count() == 1

    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_empty_search(self) -> None:
        from houyi.rag.indexed.index.sparse import SparseIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = SparseIndex(knowledge_dir=tmpdir)
            await index.load()

            results = await index.search("test query", k=10)
            assert results == []

    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_incremental_add(self) -> None:
        from houyi.rag.indexed.index.sparse import SparseIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = SparseIndex(knowledge_dir=tmpdir)
            await index.load()

            await index.add_batch(
                [
                    Chunk(
                        chunk_id="chunk-1",
                        doc_id="doc-1",
                        content="First document",
                        metadata={},
                    ),
                ]
            )
            assert index.count() == 1

            await index.add_batch(
                [
                    Chunk(
                        chunk_id="chunk-2",
                        doc_id="doc-2",
                        content="Second document",
                        metadata={},
                    ),
                ]
            )
            assert index.count() == 2
