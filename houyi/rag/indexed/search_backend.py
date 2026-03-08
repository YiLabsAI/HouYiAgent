from __future__ import annotations

from typing import Any


async def ensure_embedder(*, embedder: Any, embedding_config: Any):
    """Return an embedder instance, creating one only when the runtime has none."""

    if embedder is not None:
        return embedder

    from houyi.rag.indexed.embedding import create_embedder

    return create_embedder(embedding_config)


async def ensure_vector_index(*, vector_index: Any, embedding_config: Any, index_dir: Any):
    """Return the vector index bound to the indexed runtime storage directory."""

    if vector_index is not None:
        return vector_index

    from houyi.rag.indexed.index.vector import VectorIndex

    vector_index = VectorIndex(
        dimension=embedding_config.dimension,
        knowledge_dir=index_dir,
    )
    await vector_index.load()
    return vector_index


async def ensure_sparse_index(*, sparse_index: Any, index_dir: Any):
    """Return the sparse index bound to the indexed runtime storage directory."""

    if sparse_index is not None:
        return sparse_index

    from houyi.rag.indexed.index.sparse import SparseIndex

    sparse_index = SparseIndex(knowledge_dir=index_dir)
    await sparse_index.load()
    return sparse_index


async def ensure_graph_store(*, graph_store: Any, index_dir: Any, graph_config: Any):
    """Return the graph store for the indexed runtime when graph retrieval is enabled."""

    if graph_store is not None:
        return graph_store

    from houyi.rag.indexed.graph.store import GraphStore

    graph_store = GraphStore(
        knowledge_dir=index_dir,
        config=graph_config,
    )
    await graph_store.load()
    return graph_store


async def vector_search(*, query: str, k: int, resources: Any):
    """Execute vector retrieval using runtime-managed embedder and vector index."""

    embedder = await resources.get_embedder()
    vector_index = await resources.get_vector_index()
    query_embedding = await embedder.embed(query)
    return await vector_index.search(query_embedding, k)


async def sparse_search(*, query: str, k: int, resources: Any):
    """Execute sparse retrieval using the runtime-managed sparse index."""

    sparse_index = await resources.get_sparse_index()
    return await sparse_index.search(query, k)


async def graph_search(*, query: str, k: int, resources: Any):
    """Execute graph retrieval using the runtime-managed graph store."""

    graph_store = await resources.get_graph_store()
    return await graph_store.search(query, k)
