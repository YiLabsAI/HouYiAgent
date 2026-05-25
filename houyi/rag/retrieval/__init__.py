"""Cross-mode retrieval domain for reusable RAG retrieval abstractions.

This package contains retrieval protocols, shared retriever implementations,
factory/config helpers, and reusable retrieval algorithms that are not owned by
any single RAG mode.

The exported create_hybrid_retriever() helper is currently an indexed-backed
convenience constructor. The core cross-mode boundary in this package remains
the protocol surface plus HybridRetriever itself.
"""

from houyi.rag.retrieval.config import HybridRetrieverConfig
from houyi.rag.retrieval.factory import create_hybrid_retriever
from houyi.rag.retrieval.hybrid import HybridRetriever
from houyi.rag.retrieval.protocols import Generator, Reranker, Retriever, Validator

__all__ = [
    "Generator",
    "HybridRetriever",
    "HybridRetrieverConfig",
    "Reranker",
    "Retriever",
    "Validator",
    "create_hybrid_retriever",
]
