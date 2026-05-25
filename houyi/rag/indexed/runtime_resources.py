from __future__ import annotations

from typing import Any

from houyi.rag.indexed.search_backend import (
    ensure_embedder,
    ensure_graph_store,
    ensure_sparse_index,
    ensure_vector_index,
)


class IndexedRuntimeResources:
    """Own lazily initialized runtime resources for indexed retrieval.

    This object centralizes mutable runtime state that is shared by search and
    ingest collaborators. Callers may inject prebuilt resources in tests, but
    production code should prefer the get_* accessors so initialization and
    storage location stay consistent.
    """

    def __init__(
        self,
        *,
        embedding_config: Any,
        graph_config: Any,
        index_dir: str,
    ) -> None:
        self._embedding_config = embedding_config
        self._graph_config = graph_config
        self._index_dir = index_dir

        self._embedder: Any = None
        self._vector_index: Any = None
        self._sparse_index: Any = None
        self._graph_store: Any = None

    @property
    def embedder(self) -> Any:
        return self._embedder

    @embedder.setter
    def embedder(self, value: Any) -> None:
        self._embedder = value

    @property
    def vector_index(self) -> Any:
        return self._vector_index

    @vector_index.setter
    def vector_index(self, value: Any) -> None:
        self._vector_index = value

    @property
    def sparse_index(self) -> Any:
        return self._sparse_index

    @sparse_index.setter
    def sparse_index(self, value: Any) -> None:
        self._sparse_index = value

    @property
    def graph_store(self) -> Any:
        return self._graph_store

    @graph_store.setter
    def graph_store(self, value: Any) -> None:
        self._graph_store = value

    async def get_embedder(self) -> Any:
        """Return the active embedder, creating it on first use."""
        self._embedder = await ensure_embedder(
            embedder=self._embedder,
            embedding_config=self._embedding_config,
        )
        return self._embedder

    async def get_vector_index(self) -> Any:
        """Return the vector index bound to this resource owner's index directory."""
        self._vector_index = await ensure_vector_index(
            vector_index=self._vector_index,
            embedding_config=self._embedding_config,
            index_dir=self._index_dir,
        )
        return self._vector_index

    async def get_sparse_index(self) -> Any:
        """Return the sparse index bound to this resource owner's index directory."""
        self._sparse_index = await ensure_sparse_index(
            sparse_index=self._sparse_index,
            index_dir=self._index_dir,
        )
        return self._sparse_index

    async def get_graph_store(self) -> Any:
        """Return the graph store when graph retrieval is configured for this runtime."""
        self._graph_store = await ensure_graph_store(
            graph_store=self._graph_store,
            index_dir=self._index_dir,
            graph_config=self._graph_config,
        )
        return self._graph_store
