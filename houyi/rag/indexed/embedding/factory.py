from __future__ import annotations

from houyi.rag.config import EmbeddingConfig
from houyi.rag.indexed.embedding.api import APIEmbedder
from houyi.rag.indexed.embedding.base import Embedder
from houyi.rag.indexed.embedding.gemini import GeminiEmbedder
from houyi.rag.indexed.embedding.local import LocalEmbedder


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Create an embedder for the configured indexed embedding provider."""
    if config.provider == "local":
        return LocalEmbedder(
            model=config.model,
            dimension=config.dimension,
        )
    if config.provider in ("gemini", "vertex"):
        return GeminiEmbedder(
            model=config.model,
            dimension=config.dimension,
        )
    return APIEmbedder(
        provider=config.provider,
        model=config.model,
        dimension=config.dimension,
    )
