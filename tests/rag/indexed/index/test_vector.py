"""Tests for indexed vector index."""

from __future__ import annotations

import importlib.util
import json
import tempfile

import pytest

from houyi.rag.types import Chunk

HNSWLIB_AVAILABLE = importlib.util.find_spec("hnswlib") is not None


class TestVectorIndex:
    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_basic(self) -> None:
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index.load()

            chunk = Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                content="Test content",
                metadata={"source": "test.txt", "chunk_index": 0},
            )

            identifier = await index.add(chunk, [0.1] * 16)

            assert identifier == 0
            assert index.count() == 1

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_batch_add(self) -> None:
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index.load()

            chunks = [
                Chunk(
                    chunk_id=f"chunk-{i}",
                    doc_id="doc-1",
                    content=f"Content {i}",
                    metadata={"source": "test.txt", "chunk_index": i},
                )
                for i in range(5)
            ]
            embeddings = [[float(i) * 0.1] * 16 for i in range(5)]

            ids = await index.add_batch(chunks, embeddings)

            assert len(ids) == 5
            assert index.count() == 5

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_search(self) -> None:
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index.load()

            chunks = [
                Chunk(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    content="Python programming language",
                    metadata={"source": "python.txt", "chunk_index": 0},
                ),
                Chunk(
                    chunk_id="chunk-2",
                    doc_id="doc-2",
                    content="Java programming language",
                    metadata={"source": "java.txt", "chunk_index": 0},
                ),
            ]
            embeddings = [[0.9] * 16, [0.1] * 16]
            await index.add_batch(chunks, embeddings)

            results = await index.search([0.85] * 16, k=2)

            assert len(results) == 2
            assert results[0].chunk_id == "chunk-1"
            assert "Python" in results[0].content

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_persistence(self) -> None:
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index1 = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index1.load()

            chunk = Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                content="Persistent content",
                metadata={"source": "test.txt", "chunk_index": 0},
            )
            await index1.add(chunk, [0.5] * 16)
            await index1.save()

            index2 = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index2.load()

            assert index2.count() == 1

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_empty_search(self) -> None:
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index.load()

            results = await index.search([0.5] * 16, k=10)
            assert results == []

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_delete(self) -> None:
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index.load()

            chunk = Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                content="Delete me",
                metadata={"source": "test.txt", "chunk_index": 0},
            )
            identifier = await index.add(chunk, [0.5] * 16)

            await index.delete([identifier])

            assert identifier not in index._id_to_chunk

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_dimension_saved_in_metadata(self, tmp_path) -> None:
        from houyi.rag.indexed.index.vector import VectorIndex

        index = VectorIndex(dimension=768, knowledge_dir=str(tmp_path))

        chunk = Chunk(
            chunk_id="test_0",
            doc_id="test",
            content="Test content",
            start_idx=0,
            end_idx=12,
        )
        await index.add(chunk, [0.1] * 768)
        await index.save()

        meta_path = tmp_path / ".houyi" / "vector_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta.get("dimension") == 768
