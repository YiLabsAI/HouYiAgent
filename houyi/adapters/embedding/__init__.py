"""Embedding provider abstractions and implementations.

This is the canonical home for embedding providers across HouYi.
Memory, RAG, rerank/classify, and any future module that needs
text-vector embeddings should import from here.

Resolution policy
-----------------
make_embedding_provider selects an implementation from the
EMBEDDING_PROVIDER environment variable (or an explicit override).
Supported values: siliconflow / dashscope / local / noop.

See houyi.infrastructure.config.env_config for the underlying
environment-variable contracts.

Note: houyi.rag.indexed.embedding.Embedder exposes a different
shape (single-text in / vector out) tuned for the RAG ingestion
pipeline; that protocol is scheduled to be consolidated against
this module in a follow-up sprint.
"""

from __future__ import annotations

from houyi.adapters.embedding.factory import (
    DEFAULT_SILICONFLOW_EMBEDDING_MODEL,
    make_embedding_provider,
)
from houyi.adapters.embedding.protocol import (
    EmbeddingProvider,
    EmbeddingProviderError,
    cosine_similarity,
)
from houyi.adapters.embedding.providers import (
    DashScopeEmbeddingProvider,
    NoOpEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    SiliconFlowEmbeddingProvider,
)

__all__ = [
    "DEFAULT_SILICONFLOW_EMBEDDING_MODEL",
    "DashScopeEmbeddingProvider",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "NoOpEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "SiliconFlowEmbeddingProvider",
    "cosine_similarity",
    "make_embedding_provider",
]
