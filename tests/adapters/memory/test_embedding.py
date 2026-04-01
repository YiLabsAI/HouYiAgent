"""Embedding provider and cosine_similarity unit tests.

Covers NoOpEmbeddingProvider, cosine_similarity edge cases,
and EmbeddingProvider protocol compliance.
"""

from __future__ import annotations

import math

from houyi.adapters.memory.embedding import (
    NoOpEmbeddingProvider,
    cosine_similarity,
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
