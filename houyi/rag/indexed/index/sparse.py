"""Sparse index using BM25."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from houyi.rag.types import Chunk, SearchResult, Source


class SparseIndex:
    """Sparse index using BM25 for lexical search.

    Uses bm25s, a high-performance BM25 implementation from HuggingFace.
    Package size: ~100KB, depends on numpy + scipy.

    Reference: https://github.com/xhluca/bm25s
    """

    def __init__(
        self,
        knowledge_dir: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize sparse index.

        Args:
            knowledge_dir: Knowledge base directory (for persistence)
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.k1 = k1
        self.b = b

        self._index: Any = None
        self._chunks: list[Chunk] = []
        self._index_path = self.knowledge_dir / ".houyi" / "sparse_index"

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        self.close()

    def close(self) -> None:
        """Close and cleanup index resources."""
        if self._index is not None:
            # bm25s stores internal state - setting to None allows GC
            self._index = None

    def _ensure_index(self) -> None:
        """Ensure index is initialized."""
        if self._index is not None:
            return

        try:
            import bm25s
        except ImportError as err:
            raise ImportError(
                "bm25s package required for sparse index. Install with: pip install bm25s"
            ) from err

        # Create empty index
        self._index = bm25s.BM25()

    async def load(self) -> None:
        """Load index from disk."""
        if self._index_path.exists():
            try:
                import bm25s

                self._index = bm25s.BM25.load(str(self._index_path))

                # Load chunk metadata
                meta_path = self._index_path / "chunks.json"
                if meta_path.exists():
                    with open(meta_path, encoding="utf-8") as f:
                        chunks_data = json.load(f)
                        self._chunks = [Chunk(**c) for c in chunks_data]
            except Exception:
                self._ensure_index()
        else:
            self._ensure_index()

    async def save(self) -> None:
        """Save index to disk."""
        if self._index is None:
            return

        self._index_path.mkdir(parents=True, exist_ok=True)
        self._index.save(str(self._index_path))

        # Save chunk metadata
        meta_path = self._index_path / "chunks.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self._chunks], f, ensure_ascii=False, indent=2)

    async def add_batch(self, chunks: list[Chunk]) -> None:
        """Add chunks to index.

        Note: BM25 typically requires rebuilding the entire index
        when adding new documents for optimal performance.

        Args:
            chunks: Chunks to add
        """
        self._ensure_index()

        # Add to chunk list
        self._chunks.extend(chunks)

        # Rebuild index with all chunks
        corpus = [c.content for c in self._chunks]

        try:
            import bm25s

            # Tokenize corpus
            corpus_tokens = bm25s.tokenize(corpus)
            # Build index
            self._index = bm25s.BM25()
            self._index.index(corpus_tokens)
        except ImportError as err:
            raise ImportError("bm25s package required for sparse index") from err

    async def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[SearchResult]:
        """Search for relevant chunks.

        Args:
            query: Query string
            k: Number of results

        Returns:
            List of search results
        """
        self._ensure_index()

        if not self._chunks:
            return []

        try:
            import bm25s

            # Tokenize query
            query_tokens = bm25s.tokenize([query])
            # Search
            results, scores = self._index.retrieve(query_tokens, k=min(k, len(self._chunks)))
        except ImportError as err:
            raise ImportError("bm25s package required for sparse index") from err

        search_results = []
        for idx, score in zip(results[0], scores[0], strict=False):
            if idx < len(self._chunks):
                chunk = self._chunks[idx]
                search_results.append(
                    SearchResult(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        score=float(score),
                        source=Source(
                            file_path=chunk.metadata.get("source", ""),
                            location=f"chunk {chunk.metadata.get('chunk_index', 0)}",
                            snippet=chunk.content[:200],
                            score=float(score),
                        ),
                        metadata=chunk.metadata,
                    )
                )

        return search_results

    def count(self) -> int:
        """Get number of indexed chunks."""
        return len(self._chunks)
