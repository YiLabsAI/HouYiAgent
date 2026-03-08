"""Protocols for RAG retrieval components."""

from __future__ import annotations

from typing import Any, Protocol

from houyi.rag.types import SearchResult


class Retriever(Protocol):
    """Protocol for retrieval components."""

    async def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Retrieve relevant results for a query."""
        ...


class QueryEmbedder(Protocol):
    """Protocol for components that embed raw text queries for retrieval."""

    async def embed(self, text: str) -> Any:
        """Embed a text query into the backend-specific vector representation."""
        ...


class VectorSearchBackend(Protocol):
    """Protocol for vector-capable retrieval backends."""

    async def search(self, query_embedding: Any, top_k: int = 10) -> list[SearchResult]:
        """Search by query embedding and return ranked results."""
        ...


class TextSearchBackend(Protocol):
    """Protocol for retrieval backends that search directly from raw text queries."""

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search by raw text query and return ranked results."""
        ...


class Reranker(Protocol):
    """Protocol for reranking components."""

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 10
    ) -> list[SearchResult]:
        """Rerank search results."""
        ...


class Generator(Protocol):
    """Protocol for answer generation components."""

    async def generate(self, query: str, results: list[SearchResult]) -> tuple[str, float]:
        """Generate answer from results. Returns (answer, confidence)."""
        ...


class Validator(Protocol):
    """Protocol for result validation components (e.g., CRAG)."""

    async def validate(
        self, query: str, results: list[SearchResult]
    ) -> tuple[list[SearchResult], bool]:
        """Validate and filter results. Returns (filtered_results, needs_web_search)."""
        ...
