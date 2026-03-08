from __future__ import annotations

from houyi.rag.types import Chunk, SearchResult, Source


def build_chunk_search_result(*, chunk: Chunk, score: float) -> SearchResult:
    """Project an indexed chunk into the shared retrieval result shape."""
    return SearchResult(
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
    )
