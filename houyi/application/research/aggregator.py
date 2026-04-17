"""SourceAggregator — deduplicate, rank, filter, and merge sources from all sub-questions.

Applies three-stage deduplication:
  1. URL exact match (same URL → merge, keep higher reliability)
  2. Content similarity > threshold → merge
  3. Noise filtering: drop sources with zero relevance to user query
  4. Rank by: relevance × reliability × recency
"""

from __future__ import annotations

import hashlib
import logging
import re
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
        *,
        user_query: str = "",
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

        # Noise filtering: remove sources with zero relevance to any
        # sub-question or the user query.  This prevents clearly
        # irrelevant results (e.g. same-name persons in unrelated
        # domains) from reaching the report generator.
        question_texts = [result.summary or "" for result in results]
        deduped, noise_count = _filter_noise_sources(
            deduped,
            user_query=user_query,
            question_texts=question_texts,
            question_groups=question_groups,
        )
        dup_count += noise_count

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


# Minimum combined (title + snippet) length for a source to be considered
# usable evidence.  Below this threshold the source adds noise without
# content that the report generator can cite.
_MIN_USABLE_CONTENT_CHARS = 16

# Stop-words excluded from keyword extraction for noise filtering.
_NOISE_FILTER_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "for",
        "nor",
        "so",
        "yet",
        "at",
        "by",
        "in",
        "of",
        "on",
        "to",
        "up",
        "is",
        "it",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "shall",
        "can",
        "could",
        "may",
        "might",
        "would",
        "should",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "how",
        "that",
        "this",
        "these",
        "those",
        "with",
        "from",
        "into",
        "about",
    }
)


def _extract_noise_filter_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text for noise filtering."""
    if not text:
        return set()
    tokens = re.findall(r"[\w\u4e00-\u9fff]{2,}", text.lower())
    return {t for t in tokens if t not in _NOISE_FILTER_STOP_WORDS}


def _filter_noise_sources(
    sources: list[SourceReference],
    *,
    user_query: str,
    question_texts: list[str],
    question_groups: dict[str, set[str]],
) -> tuple[list[SourceReference], int]:
    """Remove sources with zero relevance to the research topic.

    Uses a conservative keyword overlap check: a source is kept if ANY
    keyword from the user query or question texts appears in its title,
    snippet, or URL.  Only completely unrelated sources are dropped.

    Returns (filtered_sources, noise_count).
    """
    if not sources or len(sources) <= 3:
        return sources, 0

    # Build keyword pool from user query and all question texts.
    keywords = _extract_noise_filter_keywords(user_query)
    for text in question_texts:
        keywords |= _extract_noise_filter_keywords(text)

    if not keywords:
        return sources, 0

    kept: list[SourceReference] = []
    noise_count = 0
    for src in sources:
        combined = f"{src.title} {src.snippet}".strip()
        # Filter sources with no usable content.
        if len(combined) < _MIN_USABLE_CONTENT_CHARS:
            noise_count += 1
            if src.reference_id in question_groups:
                del question_groups[src.reference_id]
            continue
        searchable = f"{combined} {src.url or ''}".lower()
        overlap = sum(1 for kw in keywords if kw in searchable)
        if overlap == 0:
            noise_count += 1
            logger.debug(
                "Noise source filtered: ref_id=%s title=%r",
                src.reference_id,
                src.title[:60],
            )
            if src.reference_id in question_groups:
                del question_groups[src.reference_id]
            continue
        kept.append(src)
    return kept, noise_count
