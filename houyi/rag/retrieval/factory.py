"""Factory helpers for retrieval-domain convenience constructors.

The exported factory in this module currently provides a convenience path for
building a HybridRetriever backed by indexed-mode concrete components. The
retrieval domain keeps the reusable retriever abstraction, while the indexed
runtime wiring stays contained in this module instead of leaking into package
callers.
"""

from __future__ import annotations

from typing import Any

from houyi.rag.retrieval.hybrid import HybridRetriever


def _build_indexed_hybrid_components(
    *,
    knowledge_dir: str,
    embedding_dimension: int,
    enable_graph: bool,
) -> dict[str, Any]:
    """Build the indexed-backed collaborators used by the convenience factory.

    This keeps concrete indexed/ wiring in one place so the exported factory
    remains a thin adapter from package-level convenience API to retrieval-local
    HybridRetriever construction.
    """
    from houyi.rag.config import EmbeddingConfig
    from houyi.rag.indexed.embedding import create_embedder
    from houyi.rag.indexed.index.sparse import SparseIndex
    from houyi.rag.indexed.index.vector import VectorIndex

    vector_index = VectorIndex(dimension=embedding_dimension, knowledge_dir=knowledge_dir)
    sparse_index = SparseIndex(knowledge_dir=knowledge_dir)
    embedding_config = EmbeddingConfig(
        provider="local",
        dimension=embedding_dimension,
    )
    embedder = create_embedder(embedding_config)

    graph_store = None
    if enable_graph:
        from houyi.rag.indexed.graph.store import GraphStore

        graph_store = GraphStore(knowledge_dir=knowledge_dir)

    return {
        "vector_index": vector_index,
        "sparse_index": sparse_index,
        "graph_store": graph_store,
        "embedder": embedder,
    }


def create_hybrid_retriever(
    knowledge_dir: str,
    embedding_dimension: int = 384,
    enable_graph: bool = False,
) -> HybridRetriever:
    """Create a HybridRetriever via indexed-backed default collaborators.

    This is a package-level convenience API, not the canonical ownership point
    for retrieval abstractions. Callers that already have protocol-shaped
    backends should construct HybridRetriever directly.
    """
    return HybridRetriever(
        **_build_indexed_hybrid_components(
            knowledge_dir=knowledge_dir,
            embedding_dimension=embedding_dimension,
            enable_graph=enable_graph,
        )
    )
