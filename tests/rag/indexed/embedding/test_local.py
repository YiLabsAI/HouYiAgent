"""Tests for indexed local embedders."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from houyi.rag.indexed.embedding.local import LocalEmbedder


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class FakeTextEmbedding:
    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]):
        self.calls.append(list(texts))
        return [FakeVector([float(len(text)), 1.0]) for text in texts]


class TestLocalEmbedder:
    async def test_embed_loads_encoder_once_and_embeds_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[FakeTextEmbedding] = []

        def build_encoder(*, model_name: str) -> FakeTextEmbedding:
            encoder = FakeTextEmbedding(model_name=model_name)
            created.append(encoder)
            return encoder

        monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=build_encoder))
        embedder = LocalEmbedder(model="BAAI/bge-small-en-v1.5", dimension=2)

        first = await embedder.embed("hello")
        second = await embedder.embed("world")

        assert first == [5.0, 1.0]
        assert second == [5.0, 1.0]
        assert len(created) == 1
        assert created[0].calls == [["hello"], ["world"]]

    async def test_embed_batch_reports_progress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created: list[FakeTextEmbedding] = []

        def build_encoder(*, model_name: str) -> FakeTextEmbedding:
            encoder = FakeTextEmbedding(model_name=model_name)
            created.append(encoder)
            return encoder

        monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=build_encoder))
        progress_calls: list[tuple[int, int, int]] = []
        embedder = LocalEmbedder(model="BAAI/bge-small-en-v1.5", dimension=2)

        result = await embedder.embed_batch(
            ["a", "bb"],
            progress_callback=lambda current, total, batch: progress_calls.append(
                (current, total, batch)
            ),
        )

        assert result == [[1.0, 1.0], [2.0, 1.0]]
        assert created[0].calls == [["a", "bb"]]
        assert progress_calls == [(2, 2, 2)]

    def test_ensure_encoder_raises_when_fastembed_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delitem(sys.modules, "fastembed", raising=False)
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("missing fastembed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        embedder = LocalEmbedder(model="BAAI/bge-small-en-v1.5", dimension=2)

        with pytest.raises(ImportError, match="fastembed package required for local embedding"):
            embedder._ensure_encoder()
