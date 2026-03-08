from __future__ import annotations

from houyi.rag.indexed.embedding.base import BaseEmbedder, ProgressCallback


class APIEmbedder(BaseEmbedder):
    """Embedder using remote provider APIs."""

    def __init__(
        self,
        provider: str,
        model: str,
        dimension: int,
    ) -> None:
        super().__init__(dimension)
        self.provider = provider
        self.model = model

    async def embed(self, text: str) -> list[float]:
        """Embed using the configured remote provider."""
        if self.provider == "openai":
            return await self._embed_openai(text)
        if self.provider == "anthropic":
            return await self._embed_openai(text)
        raise ValueError(f"Unknown provider: {self.provider}")

    async def _embed_openai(self, text: str) -> list[float]:
        """Embed using OpenAI-compatible embeddings API."""
        try:
            import openai

            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except ImportError as err:
            raise ImportError("openai package required for API embedding") from err

    async def embed_batch(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Batch embed using the configured remote provider when supported."""
        if self.provider == "openai":
            try:
                import openai

                client = openai.AsyncOpenAI()
                response = await client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                if progress_callback:
                    progress_callback(len(texts), len(texts), len(texts))
                return [d.embedding for d in response.data]
            except ImportError as err:
                raise ImportError("openai package required for API embedding") from err
        return await super().embed_batch(texts, progress_callback)
