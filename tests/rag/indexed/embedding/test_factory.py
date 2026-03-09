"""Tests for indexed embedding factory helpers."""

from __future__ import annotations

import pytest

from houyi.rag.config import EmbeddingConfig
from houyi.rag.indexed.embedding import APIEmbedder, LocalEmbedder, create_embedder


class TestEmbeddingFactory:
    def test_create_embedder_api(self) -> None:
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )
        embedder = create_embedder(config)
        assert isinstance(embedder, APIEmbedder)
        assert embedder.provider == "openai"
        assert embedder.dimension == 1536

    def test_create_embedder_local(self) -> None:
        config = EmbeddingConfig(
            provider="local",
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        embedder = create_embedder(config)
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert embedder.dimension == 384

    def test_create_embedder_vertex(self) -> None:
        config = EmbeddingConfig(
            provider="vertex",
            model="text-embedding-004",
            dimension=768,
        )
        embedder = create_embedder(config)
        from houyi.rag.indexed.embedding import GeminiEmbedder

        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.model == "text-embedding-004"
        assert embedder.dimension == 768

    def test_api_embedder_unknown_provider(self) -> None:
        embedder = APIEmbedder(
            provider="unknown",
            model="test",
            dimension=768,
        )
        with pytest.raises(ValueError, match="Unknown provider"):
            import asyncio

            asyncio.run(embedder.embed("test"))
