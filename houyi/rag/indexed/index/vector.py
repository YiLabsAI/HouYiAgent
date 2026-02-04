"""Vector index using hnswlib."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from houyi.rag.types import Chunk, SearchResult, Source


class VectorIndex:
    """Vector index using hnswlib for approximate nearest neighbor search.

    hnswlib is a lightweight, header-only C++ library with Python bindings.
    Package size: ~2.5MB wheel, no external dependencies.

    Reference: https://github.com/nmslib/hnswlib
    """

    def __init__(
        self,
        dimension: int = 1536,
        knowledge_dir: str = "knowledge/",
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
    ) -> None:
        """Initialize vector index.

        Args:
            dimension: Vector dimension
            knowledge_dir: Knowledge base directory (for persistence)
            max_elements: Maximum number of elements
            ef_construction: HNSW construction parameter
            M: HNSW connections parameter
        """
        self.dimension = dimension
        self.knowledge_dir = Path(knowledge_dir)
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M

        self._index: Any = None
        self._id_to_chunk: dict[int, Chunk] = {}
        self._next_id = 0
        self._index_path = self.knowledge_dir / ".houyi" / "vector_index.bin"
        self._meta_path = self.knowledge_dir / ".houyi" / "vector_meta.json"

    def _ensure_index(self) -> None:
        """Ensure index is initialized."""
        if self._index is not None:
            return

        try:
            import hnswlib
        except ImportError as err:
            raise ImportError(
                "hnswlib package required for vector index. "
                "Install with: pip install hnswlib"
            ) from err

        self._index = hnswlib.Index(space="cosine", dim=self.dimension)
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.set_ef(50)  # Search parameter

    async def load(self) -> None:
        """Load index from disk."""
        self._ensure_index()

        if self._index_path.exists() and self._meta_path.exists():
            self._index.load_index(str(self._index_path))

            with open(self._meta_path, encoding="utf-8") as f:
                meta = json.load(f)
                self._next_id = meta.get("next_id", 0)
                # Load chunk metadata
                for id_str, chunk_data in meta.get("chunks", {}).items():
                    self._id_to_chunk[int(id_str)] = Chunk(**chunk_data)

    async def save(self) -> None:
        """Save index to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        self._index.save_index(str(self._index_path))

        meta = {
            "next_id": self._next_id,
            "chunks": {
                str(id_): chunk.model_dump()
                for id_, chunk in self._id_to_chunk.items()
            },
        }
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    async def add(self, chunk: Chunk, embedding: list[float]) -> int:
        """Add a single chunk to index.

        Args:
            chunk: Chunk to add
            embedding: Vector embedding

        Returns:
            Assigned ID
        """
        self._ensure_index()

        id_ = self._next_id
        self._next_id += 1

        self._index.add_items([embedding], [id_])
        self._id_to_chunk[id_] = chunk

        return id_

    async def add_batch(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> list[int]:
        """Add multiple chunks to index.

        Args:
            chunks: Chunks to add
            embeddings: Corresponding embeddings

        Returns:
            Assigned IDs
        """
        self._ensure_index()

        ids = list(range(self._next_id, self._next_id + len(chunks)))
        self._next_id += len(chunks)

        self._index.add_items(embeddings, ids)
        for id_, chunk in zip(ids, chunks, strict=True):
            self._id_to_chunk[id_] = chunk

        return ids

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[SearchResult]:
        """Search for similar chunks.

        Args:
            query_embedding: Query vector
            k: Number of results

        Returns:
            List of search results
        """
        self._ensure_index()

        if self._index.get_current_count() == 0:
            return []

        labels, distances = self._index.knn_query([query_embedding], k=min(k, self._index.get_current_count()))

        results = []
        for label, distance in zip(labels[0], distances[0], strict=True):
            chunk = self._id_to_chunk.get(label)
            if chunk:
                # Convert cosine distance to similarity score
                score = 1.0 - distance

                results.append(SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=score,
                    source=Source(
                        file_path=chunk.metadata.get("source", ""),
                        location=f"chunk {chunk.metadata.get('chunk_index', 0)}",
                        snippet=chunk.content[:200],
                        score=score,
                    ),
                    metadata=chunk.metadata,
                ))

        return results

    async def delete(self, ids: list[int]) -> None:
        """Delete chunks from index.

        Note: hnswlib supports marking elements for deletion.

        Args:
            ids: IDs to delete
        """
        self._ensure_index()

        for id_ in ids:
            if id_ in self._id_to_chunk:
                self._index.mark_deleted(id_)
                del self._id_to_chunk[id_]

    def count(self) -> int:
        """Get number of indexed elements."""
        if self._index is None:
            return 0
        return self._index.get_current_count()
