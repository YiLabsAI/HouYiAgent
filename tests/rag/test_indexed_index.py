"""Tests for indexed embedding and index components."""

from __future__ import annotations

import importlib.util
import tempfile

import pytest

from houyi.rag.config import EmbeddingConfig
from houyi.rag.indexed.embedding.base import (
    APIEmbedder,
    BaseEmbedder,
    LocalEmbedder,
    create_embedder,
)
from houyi.rag.types import Chunk

# Check if optional dependencies are available
HNSWLIB_AVAILABLE = importlib.util.find_spec("hnswlib") is not None
BM25S_AVAILABLE = importlib.util.find_spec("bm25s") is not None


class TestEmbedding:
    """Tests for embedding components."""

    def test_create_embedder_api(self) -> None:
        """Test creating API embedder."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )
        embedder = create_embedder(config)
        assert isinstance(embedder, APIEmbedder)
        assert embedder.provider == "openai"
        assert embedder.dimension == 1536

    def test_create_embedder_local(self) -> None:
        """Test creating local embedder configuration."""
        config = EmbeddingConfig(
            provider="local",
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        embedder = create_embedder(config)
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert embedder.dimension == 384

    def test_api_embedder_unknown_provider(self) -> None:
        """Test that unknown provider raises error."""
        embedder = APIEmbedder(
            provider="unknown",
            model="test",
            dimension=768,
        )
        with pytest.raises(ValueError, match="Unknown provider"):
            import asyncio
            asyncio.run(embedder.embed("test"))


class FakeEmbedder(BaseEmbedder):
    """Fake embedder for testing."""

    async def embed(self, text: str) -> list[float]:
        """Return a simple hash-based embedding."""
        # Create a simple deterministic embedding
        hash_val = hash(text) % 10000
        return [float(hash_val % 100) / 100.0] * self.dimension


class TestVectorIndex:
    """Tests for vector index (hnswlib)."""

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_basic(self) -> None:
        """Test basic vector index operations."""
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index.load()

            # Create test chunk
            chunk = Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                content="Test content",
                metadata={"source": "test.txt", "chunk_index": 0},
            )

            # Add to index
            embedding = [0.1] * 16
            id_ = await index.add(chunk, embedding)

            assert id_ == 0
            assert index.count() == 1

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_batch_add(self) -> None:
        """Test batch adding to vector index."""
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
        """Test vector index search."""
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index.load()

            # Add test chunks
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
            embeddings = [
                [0.9] * 16,  # Python
                [0.1] * 16,  # Java
            ]
            await index.add_batch(chunks, embeddings)

            # Search with embedding similar to Python
            query_embedding = [0.85] * 16
            results = await index.search(query_embedding, k=2)

            assert len(results) == 2
            # First result should be Python (most similar)
            assert results[0].chunk_id == "chunk-1"
            assert "Python" in results[0].content

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_persistence(self) -> None:
        """Test vector index save and load."""
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate index
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

            # Load in new instance
            index2 = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index2.load()

            assert index2.count() == 1

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_empty_search(self) -> None:
        """Test searching empty index."""
        from houyi.rag.indexed.index.vector import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(dimension=16, knowledge_dir=tmpdir)
            await index.load()

            results = await index.search([0.5] * 16, k=10)
            assert results == []

    @pytest.mark.skipif(not HNSWLIB_AVAILABLE, reason="hnswlib not installed")
    @pytest.mark.asyncio
    async def test_vector_index_delete(self) -> None:
        """Test deleting from vector index."""
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
            id_ = await index.add(chunk, [0.5] * 16)

            await index.delete([id_])

            # Note: hnswlib marks as deleted but count may not reflect immediately
            # The important thing is the chunk is no longer in the mapping
            assert id_ not in index._id_to_chunk


class TestSparseIndex:
    """Tests for sparse index (BM25)."""

    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_basic(self) -> None:
        """Test basic sparse index operations."""
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
        """Test sparse index search."""
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

            # Search for Python
            results = await index.search("Python programming", k=3)

            assert len(results) >= 1
            # Python-related chunks should rank higher
            assert any("Python" in r.content for r in results[:2])

    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_persistence(self) -> None:
        """Test sparse index save and load."""
        from houyi.rag.indexed.index.sparse import SparseIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate index
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

            # Load in new instance
            index2 = SparseIndex(knowledge_dir=tmpdir)
            await index2.load()

            assert index2.count() == 1

    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_empty_search(self) -> None:
        """Test searching empty sparse index."""
        from houyi.rag.indexed.index.sparse import SparseIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = SparseIndex(knowledge_dir=tmpdir)
            await index.load()

            results = await index.search("test query", k=10)
            assert results == []

    @pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
    @pytest.mark.asyncio
    async def test_sparse_index_incremental_add(self) -> None:
        """Test adding chunks incrementally."""
        from houyi.rag.indexed.index.sparse import SparseIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = SparseIndex(knowledge_dir=tmpdir)
            await index.load()

            # First batch
            await index.add_batch([
                Chunk(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    content="First document",
                    metadata={},
                ),
            ])
            assert index.count() == 1

            # Second batch
            await index.add_batch([
                Chunk(
                    chunk_id="chunk-2",
                    doc_id="doc-2",
                    content="Second document",
                    metadata={},
                ),
            ])
            assert index.count() == 2
