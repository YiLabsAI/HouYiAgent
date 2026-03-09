from __future__ import annotations

import pytest

from houyi.rag.indexed.embedding.base import BaseEmbedder, Embedder


class DummyEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        super().__init__(dimension=3)
        self.seen: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        return [float(len(text)), 0.0, 1.0]


class SuperCallingEmbedder(BaseEmbedder):
    async def embed(self, text: str) -> list[float]:
        return await super().embed(text)


class TestBaseEmbedder:
    async def test_embed_batch_runs_sequentially_and_reports_progress(self) -> None:
        embedder = DummyEmbedder()
        progress_calls: list[tuple[int, int, int]] = []

        result = await embedder.embed_batch(
            ["a", "bb", "ccc"],
            progress_callback=lambda current, total, batch: progress_calls.append(
                (current, total, batch)
            ),
        )

        assert embedder.dimension == 3
        assert embedder.seen == ["a", "bb", "ccc"]
        assert result == [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
        assert progress_calls == [(1, 3, 1), (2, 3, 1), (3, 3, 1)]

    async def test_embedder_protocol_runtime_checkable(self) -> None:
        embedder = DummyEmbedder()

        assert isinstance(embedder, Embedder)

    async def test_embedder_protocol_default_methods_raise_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            await Embedder.embed(object(), "hello")

        with pytest.raises(NotImplementedError):
            await Embedder.embed_batch(object(), ["hello"])

    async def test_base_embedder_abstract_super_path_raises_not_implemented(self) -> None:
        embedder = SuperCallingEmbedder(dimension=1)

        with pytest.raises(NotImplementedError):
            await embedder.embed("hello")
