from __future__ import annotations

from typing import Any

from houyi.rag.indexed.embedding.base import BaseEmbedder, ProgressCallback


class LocalEmbedder(BaseEmbedder):
    """Embedder using a local FastEmbed model."""

    def __init__(self, model: str, dimension: int) -> None:
        super().__init__(dimension)
        self.model = model
        self._encoder: Any = None

    def _ensure_encoder(self) -> None:
        """Load the local encoder on first use."""
        if self._encoder is None:
            try:
                from fastembed import TextEmbedding

                self._encoder = TextEmbedding(model_name=self.model)
            except ImportError as err:
                raise ImportError(
                    "fastembed package required for local embedding. "
                    "Install with: pip install fastembed"
                ) from err

    async def embed(self, text: str) -> list[float]:
        """Embed a single text using the local model."""
        self._ensure_encoder()
        embeddings = list(self._encoder.embed([text]))
        return embeddings[0].tolist()

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Batch embed texts using the local model."""
        self._ensure_encoder()
        embeddings = list(self._encoder.embed(texts))
        if progress_callback:
            progress_callback(len(texts), len(texts), len(texts))
        return [embedding.tolist() for embedding in embeddings]
