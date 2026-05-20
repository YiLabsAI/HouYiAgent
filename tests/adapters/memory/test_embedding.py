"""Embedding provider and cosine_similarity unit tests.

Covers NoOpEmbeddingProvider, cosine_similarity edge cases,
EmbeddingProvider protocol compliance, the SiliconFlow remote backend
(mocked), and ``make_embedding_provider`` factory resolution.
"""

from __future__ import annotations

import math

import pytest

from houyi.adapters.memory.embedding import (
    DEFAULT_SILICONFLOW_EMBEDDING_MODEL,
    EmbeddingProviderError,
    NoOpEmbeddingProvider,
    SiliconFlowEmbeddingProvider,
    cosine_similarity,
    make_embedding_provider,
)


class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_different_lengths(self):
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_known_angle(self):
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert abs(cosine_similarity(a, b) - expected) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        result = cosine_similarity(a, b)
        assert abs(result) < 1e-6 or math.isfinite(result)


class TestNoOpProvider:
    async def test_embed_returns_correct_dim(self):
        p = NoOpEmbeddingProvider(dim=64)
        embs = await p.embed(["hello world"])
        assert len(embs) == 1
        assert len(embs[0]) == 64

    async def test_dimension_method(self):
        p = NoOpEmbeddingProvider(dim=128)
        assert p.dimension() == 128

    async def test_batch_embed(self):
        p = NoOpEmbeddingProvider(dim=32)
        embs = await p.embed(["a", "b", "c"])
        assert len(embs) == 3
        for e in embs:
            assert len(e) == 32

    async def test_deterministic(self):
        p = NoOpEmbeddingProvider(dim=32)
        e1 = await p.embed(["hello"])
        e2 = await p.embed(["hello"])
        assert e1[0] == e2[0]

    async def test_different_texts_differ(self):
        p = NoOpEmbeddingProvider(dim=32)
        e1 = await p.embed(["hello"])
        e2 = await p.embed(["world"])
        assert e1[0] != e2[0]

    async def test_normalized(self):
        p = NoOpEmbeddingProvider(dim=64)
        embs = await p.embed(["test normalization"])
        norm = math.sqrt(sum(x * x for x in embs[0]))
        assert abs(norm - 1.0) < 1e-6

    async def test_empty_text(self):
        p = NoOpEmbeddingProvider(dim=16)
        embs = await p.embed([""])
        assert len(embs) == 1
        assert len(embs[0]) == 16

    async def test_empty_batch(self):
        p = NoOpEmbeddingProvider(dim=16)
        embs = await p.embed([])
        assert embs == []

    async def test_custom_dim(self):
        p = NoOpEmbeddingProvider(dim=8)
        embs = await p.embed(["x"])
        assert len(embs[0]) == 8

    async def test_self_similarity_high(self):
        p = NoOpEmbeddingProvider(dim=64)
        embs = await p.embed(["test text"])
        sim = cosine_similarity(embs[0], embs[0])
        assert abs(sim - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# SiliconFlow (mocked transport)
# ---------------------------------------------------------------------------


def _fake_embeddings_response(vectors: list[list[float]]) -> dict:
    return {"data": [{"embedding": v, "index": i} for i, v in enumerate(vectors)]}


class _FakeResponse:
    def __init__(self, status: int, payload: dict | str):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        import httpx

        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status={self.status_code}",
                request=None,
                response=None,  # type: ignore[arg-type]
            )

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("invalid json")
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = responses
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if not self._responses:
            raise AssertionError("unexpected extra POST call")
        return self._responses.pop(0)


class TestSiliconFlowProvider:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            SiliconFlowEmbeddingProvider(api_key="")

    async def test_embed_batches_order(self, monkeypatch):
        # Provider batch=2 → two POST calls for 3 inputs.
        provider = SiliconFlowEmbeddingProvider(api_key="sk-test", dimension=3, max_batch=2)
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        fake = _FakeClient(
            [
                _FakeResponse(200, _fake_embeddings_response(vectors[:2])),
                _FakeResponse(200, _fake_embeddings_response(vectors[2:])),
            ]
        )

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: fake)

        out = await provider.embed(["a", "b", "c"])
        assert out == vectors
        assert len(fake.calls) == 2
        assert fake.calls[0]["json"]["model"] == DEFAULT_SILICONFLOW_EMBEDDING_MODEL
        assert fake.calls[0]["json"]["input"] == ["a", "b"]
        assert fake.calls[0]["headers"]["Authorization"] == "Bearer sk-test"

    async def test_http_error_provider(self, monkeypatch):
        provider = SiliconFlowEmbeddingProvider(api_key="sk", dimension=3, max_batch=4)
        fake = _FakeClient([_FakeResponse(500, {})])
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: fake)
        with pytest.raises(EmbeddingProviderError):
            await provider.embed(["x"])

    async def test_count_mismatch_raises(self, monkeypatch):
        provider = SiliconFlowEmbeddingProvider(api_key="sk", dimension=3, max_batch=4)
        fake = _FakeClient([_FakeResponse(200, _fake_embeddings_response([[1.0, 0.0, 0.0]]))])
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: fake)
        with pytest.raises(EmbeddingProviderError):
            await provider.embed(["a", "b"])

    async def test_empty_batch_short_circuits(self):
        provider = SiliconFlowEmbeddingProvider(api_key="sk", dimension=3)
        assert await provider.embed([]) == []


class TestFactory:
    def test_noop(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        provider = make_embedding_provider(provider="noop")
        assert isinstance(provider, NoOpEmbeddingProvider)

    def test_siliconflow_requires_key(self, monkeypatch):
        monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
        with pytest.raises(EmbeddingProviderError):
            make_embedding_provider(provider="siliconflow")

    def test_siliconflow_uses_env_key(self, monkeypatch):
        monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-from-env")
        provider = make_embedding_provider(provider="siliconflow")
        assert isinstance(provider, SiliconFlowEmbeddingProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(EmbeddingProviderError):
            make_embedding_provider(provider="acme")

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "noop")
        provider = make_embedding_provider()
        assert isinstance(provider, NoOpEmbeddingProvider)
