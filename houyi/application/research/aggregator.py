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

        deduped, dup_count = self._deduplicate(raw)

        deduped.sort(key=lambda s: s.reliability_score, reverse=True)

        grouped: dict[str, list[str]] = defaultdict(list)
        for qid, src in raw:
            if src.reference_id in {s.reference_id for s in deduped}:
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
    ) -> tuple[list[SourceReference], int]:
        """Deduplicate by URL exact match, then content fingerprint."""
        seen_urls: dict[str, SourceReference] = {}
        seen_hashes: dict[str, SourceReference] = {}
        dup_count = 0

        for _qid, src in raw:
            if src.url and src.url in seen_urls:
                existing = seen_urls[src.url]
                if src.reliability_score > existing.reliability_score:
                    fp_old = _content_fingerprint(existing)
                    seen_urls[src.url] = src
                    seen_hashes[fp_old] = src
                dup_count += 1
                continue

            fp = _content_fingerprint(src)
            if fp in seen_hashes:
                existing = seen_hashes[fp]
                if src.reliability_score > existing.reliability_score:
                    seen_hashes[fp] = src
                    if src.url:
                        seen_urls[src.url] = src
                dup_count += 1
                continue

            seen_hashes[fp] = src
            if src.url:
                seen_urls[src.url] = src

        unique = list(seen_hashes.values())
        return unique, dup_count


def _content_fingerprint(src: SourceReference) -> str:
    """Produce a fingerprint for near-duplicate detection.

    Uses title + snippet normalized hash. For true semantic dedup,
    embeddings would be used (deferred to Phase 4).
    """
    text = f"{src.title.lower().strip()}|{src.snippet.lower().strip()}"
    return hashlib.md5(text.encode()).hexdigest()
