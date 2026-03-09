from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from houyi.rag.indexed.embedding.api import APIEmbedder


class FakeAsyncOpenAI:
    def __init__(self, data: list[list[float]]) -> None:
        self._data = data
        self.calls: list[tuple[str, object]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, *, model: str, input: object) -> SimpleNamespace:
        self.calls.append((model, input))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=embedding) for embedding in self._data]
        )


class TestAPIEmbedder:
    async def test_embed_openai_uses_async_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_client = FakeAsyncOpenAI([[0.1, 0.2, 0.3]])
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda: fake_client))

        embedder = APIEmbedder(provider="openai", model="text-embedding-3-small", dimension=3)
        result = await embedder.embed("hello")

        assert result == [0.1, 0.2, 0.3]
        assert fake_client.calls == [("text-embedding-3-small", "hello")]

    async def test_embed_anthropic_reuses_openai_compatible_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = FakeAsyncOpenAI([[0.4, 0.5, 0.6]])
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda: fake_client))

        embedder = APIEmbedder(provider="anthropic", model="anthropic-embed", dimension=3)
        result = await embedder.embed("hello")

        assert result == [0.4, 0.5, 0.6]
        assert fake_client.calls == [("anthropic-embed", "hello")]

    async def test_embed_batch_openai_reports_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = FakeAsyncOpenAI([[1.0], [2.0]])
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda: fake_client))
        progress_calls: list[tuple[int, int, int]] = []

        embedder = APIEmbedder(provider="openai", model="batch-model", dimension=1)
        result = await embedder.embed_batch(
            ["a", "b"],
            progress_callback=lambda current, total, batch: progress_calls.append(
                (current, total, batch)
            ),
        )

        assert result == [[1.0], [2.0]]
        assert fake_client.calls == [("batch-model", ["a", "b"])]
        assert progress_calls == [(2, 2, 2)]

    async def test_embed_batch_non_openai_falls_back_to_base_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = FakeAsyncOpenAI([[0.7], [0.8]])
        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda: fake_client))
        progress_calls: list[tuple[int, int, int]] = []

        embedder = APIEmbedder(provider="anthropic", model="fallback-model", dimension=1)
        result = await embedder.embed_batch(
            ["a", "b"],
            progress_callback=lambda current, total, batch: progress_calls.append(
                (current, total, batch)
            ),
        )

        assert result == [[0.7], [0.7]]
        assert fake_client.calls == [("fallback-model", "a"), ("fallback-model", "b")]
        assert progress_calls == [(1, 2, 1), (2, 2, 1)]

    async def test_embed_raises_when_openai_dependency_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delitem(sys.modules, "openai", raising=False)
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("missing openai")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        embedder = APIEmbedder(provider="openai", model="text-embedding-3-small", dimension=3)

        with pytest.raises(ImportError, match="openai package required for API embedding"):
            await embedder.embed("hello")

    async def test_embed_batch_raises_when_openai_dependency_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delitem(sys.modules, "openai", raising=False)
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("missing openai")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        embedder = APIEmbedder(provider="openai", model="text-embedding-3-small", dimension=3)

        with pytest.raises(ImportError, match="openai package required for API embedding"):
            await embedder.embed_batch(["hello"])
