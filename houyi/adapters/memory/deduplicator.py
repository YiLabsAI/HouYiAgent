"""Memory deduplication checker.

Detects duplicate, conflicting, and update relationships between
a new MemoryCandidate and existing MemoryRecords.

Strategies:
1. Exact match: same key + scope → duplicate
2. Semantic match: embedding cosine similarity > threshold → duplicate
3. Conflict detection: same key, different content → conflict
4. Update detection: same topic with newer information → update
"""

from __future__ import annotations

from houyi.adapters.memory.embedding import EmbeddingProvider, cosine_similarity
from houyi.adapters.memory.types import (
    DedupMatch,
    MemoryCandidate,
    MemoryRecord,
)


class MemoryDeduplicator:
    """Checks a candidate against existing memories for duplicates."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.9,
    ):
        self._embedding = embedding_provider
        self._threshold = similarity_threshold

    async def check(
        self,
        candidate: MemoryCandidate,
        existing: list[MemoryRecord],
    ) -> list[DedupMatch]:
        """Detect duplicates and conflicts. Target p95 < 30ms.

        Returns a list of matches — empty means no duplicates found.
        """
        if not existing:
            return []

        matches: list[DedupMatch] = []

        for record in existing:
            if record.scope == candidate.scope:
                if record.content == candidate.content:
                    matches.append(
                        DedupMatch(
                            existing_memory_id=record.record_id,
                            similarity=1.0,
                            relation="duplicate",
                        )
                    )
                    continue

                if record.key and record.key == candidate.content.split(":")[0].strip().lower():
                    matches.append(
                        DedupMatch(
                            existing_memory_id=record.record_id,
                            similarity=0.8,
                            relation="conflict"
                            if record.content != candidate.content
                            else "duplicate",
                        )
                    )

        if self._embedding and not matches:
            matches.extend(await self._semantic_check(candidate, existing))

        return matches

    async def _semantic_check(
        self,
        candidate: MemoryCandidate,
        existing: list[MemoryRecord],
    ) -> list[DedupMatch]:
        """Use embeddings to find semantic duplicates."""
        assert self._embedding is not None

        records_with_emb = [r for r in existing if r.embedding]
        if not records_with_emb:
            return []

        candidate_embs = await self._embedding.embed([candidate.content])
        if not candidate_embs:
            return []
        cand_vec = candidate_embs[0]

        matches: list[DedupMatch] = []
        for record in records_with_emb:
            assert record.embedding is not None
            sim = cosine_similarity(cand_vec, record.embedding)
            if sim >= self._threshold:
                matches.append(
                    DedupMatch(
                        existing_memory_id=record.record_id,
                        similarity=sim,
                        relation="duplicate" if sim > 0.95 else "update",
                    )
                )

        return matches
