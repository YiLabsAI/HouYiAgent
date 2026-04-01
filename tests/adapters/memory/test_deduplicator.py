"""MemoryDeduplicator unit tests.

Covers exact-match dedup, key-based conflict detection,
semantic dedup, and edge cases.
"""

from __future__ import annotations

import pytest

from houyi.adapters.memory.deduplicator import MemoryDeduplicator
from houyi.adapters.memory.embedding import NoOpEmbeddingProvider
from houyi.adapters.memory.types import (
    MemoryCandidate,
    MemoryRecord,
    MemoryScope,
)


@pytest.fixture()
def dedup() -> MemoryDeduplicator:
    return MemoryDeduplicator()


@pytest.fixture()
def dedup_with_emb() -> MemoryDeduplicator:
    return MemoryDeduplicator(
        embedding_provider=NoOpEmbeddingProvider(dim=32),
        similarity_threshold=0.9,
    )


def _candidate(content: str, scope: MemoryScope = MemoryScope.USER) -> MemoryCandidate:
    return MemoryCandidate(content=content, scope=scope)


def _record(
    key: str,
    content: str,
    scope: MemoryScope = MemoryScope.USER,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    return MemoryRecord(key=key, content=content, scope=scope, embedding=embedding)


class TestExactMatch:
    async def test_identical_content(self, dedup):
        cand = _candidate("Python is great")
        existing = [_record("lang", "Python is great")]
        matches = await dedup.check(cand, existing)
        assert len(matches) == 1
        assert matches[0].relation == "duplicate"
        assert matches[0].similarity == 1.0

    async def test_different_content(self, dedup):
        cand = _candidate("Python is great")
        existing = [_record("lang", "Rust is fast")]
        matches = await dedup.check(cand, existing)
        assert len(matches) == 0

    async def test_different_scope_skipped(self, dedup):
        cand = _candidate("data", scope=MemoryScope.SESSION)
        existing = [_record("k", "data", scope=MemoryScope.USER)]
        matches = await dedup.check(cand, existing)
        assert len(matches) == 0


class TestKeyConflict:
    async def test_same_key_different_content(self, dedup):
        cand = _candidate("lang: Rust")
        existing = [_record("lang", "Python", scope=MemoryScope.USER)]
        matches = await dedup.check(cand, existing)
        assert any(m.relation == "conflict" for m in matches)

    async def test_key_match_conflict(self, dedup):
        cand = _candidate("lang: Rust is better")
        existing = [_record("lang", "Python is good", scope=MemoryScope.USER)]
        matches = await dedup.check(cand, existing)
        assert any(m.relation == "conflict" for m in matches)


class TestSemanticDedup:
    async def test_identical_text_semantic(self, dedup_with_emb):
        emb = [0.1] * 32
        cand = _candidate("Python is great for ML")
        existing = [_record("lang", "Python is great for ML", embedding=emb)]
        matches = await dedup_with_emb.check(cand, existing)
        assert len(matches) >= 1

    async def test_no_embedding_no_semantic(self, dedup):
        emb = [0.1] * 32
        cand = _candidate("Python is great for ML")
        existing = [_record("lang", "Rust is fast", embedding=emb)]
        matches = await dedup.check(cand, existing)
        assert len(matches) == 0

    async def test_records_without_embedding(self, dedup_with_emb):
        cand = _candidate("unique statement")
        existing = [_record("k", "something else")]
        matches = await dedup_with_emb.check(cand, existing)
        assert len(matches) == 0


class TestEdgeCases:
    async def test_empty_existing(self, dedup):
        cand = _candidate("anything")
        matches = await dedup.check(cand, [])
        assert matches == []

    async def test_multiple_existing(self, dedup):
        cand = _candidate("data")
        existing = [
            _record("a", "data"),
            _record("b", "data"),
        ]
        matches = await dedup.check(cand, existing)
        assert len(matches) == 2

    async def test_custom_threshold(self):
        dedup = MemoryDeduplicator(similarity_threshold=0.99)
        cand = _candidate("unique content here")
        existing = [_record("k", "different content")]
        matches = await dedup.check(cand, existing)
        assert len(matches) == 0
