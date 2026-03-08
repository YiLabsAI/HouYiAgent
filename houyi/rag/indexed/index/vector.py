"""Vector index using hnswlib."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from houyi.rag.indexed.index.results import build_chunk_search_result
from houyi.rag.types import Chunk, SearchResult


class VectorIndex:
    """Vector index using hnswlib for approximate nearest neighbor search.

    hnswlib is a lightweight, header-only C++ library with Python bindings.
    Package size: ~2.5MB wheel, no external dependencies.

    Reference: https://github.com/nmslib/hnswlib
    """

    def __init__(
        self,
        dimension: int = 1536,
        knowledge_dir: str | None = None,
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
        self._loaded = False  # Track if index was loaded from disk
        self._index_path = self.knowledge_dir / ".houyi" / "vector_index.bin"
        self._meta_path = self.knowledge_dir / ".houyi" / "vector_meta.json"

    def _ensure_index(self) -> None:
        """Ensure index is initialized."""
        if self._index is not None:
            return

        try:
            hnswlib = importlib.import_module("hnswlib")
        except ImportError as err:
            raise ImportError(
                "hnswlib package required for vector index. Install with: pip install hnswlib"
            ) from err

        index_cls = hnswlib.Index
        self._index = index_cls(space="cosine", dim=self.dimension)
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.set_ef(50)  # Search parameter

    async def load(self) -> None:
        """Load index from disk."""
        # Skip if already loaded
        if self._loaded:
            return

        self._ensure_index()

        if self._index_path.exists() and self._meta_path.exists():
            with open(self._meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            # Validate dimension compatibility
            saved_dim = meta.get("dimension")
            if saved_dim and saved_dim != self.dimension:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "Index dimension mismatch: saved=%d, requested=%d. "
                    "Delete %s to rebuild with new dimension.",
                    saved_dim,
                    self.dimension,
                    self._index_path.parent,
                )
                raise ValueError(
                    f"Vector index dimension mismatch: existing index has dimension "
                    f"{saved_dim}, but embedder uses dimension {self.dimension}. "
                    f"Delete the .houyi directory to rebuild with new embedding model."
                )

            self._index.load_index(str(self._index_path))
            self._next_id = meta.get("next_id", 0)
            # Load chunk metadata
            for id_str, chunk_data in meta.get("chunks", {}).items():
                self._id_to_chunk[int(id_str)] = Chunk(**chunk_data)

        self._loaded = True

    async def save(self) -> None:
        """Save index to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        self._index.save_index(str(self._index_path))

        meta = {
            "dimension": self.dimension,  # Save dimension for validation
            "next_id": self._next_id,
            "chunks": {str(id_): chunk.model_dump() for id_, chunk in self._id_to_chunk.items()},
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

        labels, distances = self._index.knn_query(
            [query_embedding], k=min(k, self._index.get_current_count())
        )

        results = []
        for label, distance in zip(labels[0], distances[0], strict=True):
            chunk = self._id_to_chunk.get(label)
            if chunk:
                score = 1.0 - distance
                results.append(build_chunk_search_result(chunk=chunk, score=score))

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
