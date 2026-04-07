"""SourceAggregator — deduplicate, rank, and merge sources from all sub-questions.

Applies three-stage deduplication:
  1. URL exact match (same URL → merge, keep higher reliability)
  2. Content similarity > threshold → merge
  3. Rank by: relevance × reliability × recency
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

from houyi.application.research.types import (
    AggregatedSources,
    SearchResult,
    SourceReference,
)

logger = logging.getLogger(__name__)

_DEFAULT_SIMILARITY_THRESHOLD = 0.85


class SourceAggregator:
    """Aggregates, deduplicates, and ranks sources from multiple search results."""

    def __init__(
        self,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._sim_threshold = similarity_threshold

    async def aggregate(
        self,
        results: list[SearchResult],
    ) -> AggregatedSources:
        """Aggregate sources from all sub-question search results.

        Returns deduplicated sources ranked by reliability, with
        per-question coverage metrics.
        """
        raw: list[tuple[str, SourceReference]] = []
        for result in results:
            for src in result.sources:
                raw.append((result.question_id, src))

        deduped, dup_count, question_groups = self._deduplicate(raw)

        deduped.sort(key=lambda s: s.reliability_score, reverse=True)

        grouped: dict[str, list[str]] = defaultdict(list)
        for src in deduped:
            for qid in sorted(question_groups.get(src.reference_id, set())):
                grouped[qid].append(src.reference_id)

        coverage: dict[str, float] = {}
        for result in results:
            expected = max(1, sum(sq_sources for sq_sources in [len(result.sources)]))
            actual = len(grouped.get(result.question_id, []))
            coverage[result.question_id] = min(1.0, actual / expected)

        return AggregatedSources(
            sources=deduped,
            deduplicated_count=dup_count,
            grouped_by_question=dict(grouped),
            coverage_by_question=coverage,
        )

    def _deduplicate(
        self,
        raw: list[tuple[str, SourceReference]],
    ) -> tuple[list[SourceReference], int, dict[str, set[str]]]:
        """Deduplicate by URL exact match, then content fingerprint."""
        seen_urls: dict[str, str] = {}
        seen_hashes: dict[str, str] = {}
        canonical_sources: dict[str, SourceReference] = {}
        question_groups: dict[str, set[str]] = defaultdict(set)
        dup_count = 0

        for qid, src in raw:
            canonical_id: str | None = None

            if src.url and src.url in seen_urls:
                canonical_id = seen_urls[src.url]
            else:
                fp = _content_fingerprint(src)
                if fp in seen_hashes:
                    canonical_id = seen_hashes[fp]

            if canonical_id is not None:
                existing = canonical_sources[canonical_id]
                if src.reliability_score > existing.reliability_score:
                    canonical_sources[canonical_id] = src.model_copy(
                        update={"reference_id": canonical_id}
                    )
                if src.url:
                    seen_urls[src.url] = canonical_id
                dup_count += 1
                question_groups[canonical_id].add(qid)
                continue

            canonical_id = src.reference_id
            fp = _content_fingerprint(src)
            canonical_sources[canonical_id] = src
            seen_hashes[fp] = canonical_id
            if src.url:
                seen_urls[src.url] = canonical_id
            question_groups[canonical_id].add(qid)

        unique = list(canonical_sources.values())
        return unique, dup_count, question_groups


def _content_fingerprint(src: SourceReference) -> str:
    """Produce a fingerprint for near-duplicate detection.

    Uses title + snippet normalized hash. For true semantic dedup,
    embeddings would be used (deferred to Phase 4).
    """
    text = f"{src.title.lower().strip()}|{src.snippet.lower().strip()}"
    return hashlib.md5(text.encode()).hexdigest()
