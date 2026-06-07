"""Candidate fusion and de-duplication for memory recall.

Retrievers produce local scores with different scales. This module
normalizes and merges those candidates into a single ordered list while
preserving original per-retriever signals for traceability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable

from houyi.adapters.memory.recall.types import RecallCandidate, RetrieverKind

_DEFAULT_KIND_WEIGHTS: dict[RetrieverKind, float] = {
    RetrieverKind.ENTITY_STATE: 10.0,
    RetrieverKind.TIMELINE: 0.6,
    RetrieverKind.ITERATIVE: 1.0,
    RetrieverKind.RAW_TURN: 0.3,
    RetrieverKind.GRAPH: 5.0,
}


class Fuser(ABC):
    """Abstract base for candidate fusion strategies."""

    @abstractmethod
    def fuse(
        self,
        candidates: Iterable[RecallCandidate],
        *,
        top_k: int,
    ) -> list[RecallCandidate]:
        """Return candidates ordered by final fused score."""
        raise NotImplementedError  # pragma: no cover - abstract


class WeightedFuser(Fuser):
    """Weighted score fusion keyed by retriever kind.

    The strategy keeps the highest-scoring candidate per semantic key
    and records all contributing retrievers in signals. A semantic
    key is the normalized subject/predicate/object triple, which avoids
    returning the same fact twice when two retrievers find it through
    different paths.
    """

    def __init__(self, kind_weights: dict[RetrieverKind, float] | None = None) -> None:
        self._weights = dict(kind_weights or _DEFAULT_KIND_WEIGHTS)

    def _normalize_group(self, candidates: list[RecallCandidate]) -> list[RecallCandidate]:
        if not candidates:
            return candidates
        scores = [c.score for c in candidates]
        s_max = max(scores)
        if s_max <= 1.0:
            # Already within the [0.0, 1.0] interval: keep raw scores to avoid inflating weak candidates
            for c in candidates:
                c.signals = dict(c.signals)
                c.signals["raw_score"] = c.score
            return candidates

        valid_candidates = []
        for c in candidates:
            normalized = c.score / s_max
            c.signals = dict(c.signals)
            c.signals["raw_score"] = c.score
            c.score = normalized
            valid_candidates.append(c)
        return valid_candidates

    def fuse(
        self,
        candidates: Iterable[RecallCandidate],
        *,
        top_k: int,
    ) -> list[RecallCandidate]:
        if top_k <= 0:
            return []

        # 1. Group candidates by their retriever kind to perform min-max normalization per-retriever
        by_kind: dict[RetrieverKind, list[RecallCandidate]] = defaultdict(list)
        for cand in candidates:
            by_kind[cand.matched_by].append(cand)

        # 2. Normalize candidates within each retriever kind independently
        normalized_cands: list[RecallCandidate] = []
        for _, kind_list in by_kind.items():
            normalized_cands.extend(self._normalize_group(kind_list))

        # 3. Group normalized candidates by semantic key for deduplication
        grouped: dict[tuple[str, str, str], list[RecallCandidate]] = defaultdict(list)
        for cand in normalized_cands:
            grouped[_semantic_key(cand)].append(cand)

        fused = [self._fuse_group(group) for group in grouped.values()]
        fused.sort(key=lambda c: c.signals.get("fused_score", c.score), reverse=True)
        return fused[:top_k]

    def _fuse_group(self, group: list[RecallCandidate]) -> RecallCandidate:
        best = max(group, key=lambda c: self._weighted_score(c))
        final_score = max(self._weighted_score(c) for c in group)

        # Special merging of contributors list
        all_contributors = []
        for c in group:
            all_contributors.extend(c.signals.get("contributors", [c.matched_by.value]))
        contributors = sorted(set(all_contributors))

        best.signals = dict(best.signals)
        best.signals["fused_score"] = final_score
        best.signals["contributors"] = contributors
        best.signals["duplicate_count"] = len(group)

        # Non-destructively merge other signals from duplicate candidates (like graph relation tags)
        for other in group:
            if other is best:
                continue
            for k, v in other.signals.items():
                if k not in best.signals:
                    best.signals[k] = v

        return best

    def _weighted_score(self, candidate: RecallCandidate) -> float:
        weight = self._weights.get(candidate.matched_by, 1.0)
        return candidate.score * weight


class ReciprocalRankFuser(Fuser):
    """Reciprocal-rank fusion for already-ranked candidate lists.

    This implementation accepts a flat iterable and treats current
    order as the rank order. It is useful when retrievers have already
    done their own normalization and rank is more trustworthy than
    local score magnitude.
    """

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        self._k = k

    def fuse(
        self,
        candidates: Iterable[RecallCandidate],
        *,
        top_k: int,
    ) -> list[RecallCandidate]:
        if top_k <= 0:
            return []

        scores: dict[tuple[str, str, str], float] = defaultdict(float)
        best: dict[tuple[str, str, str], RecallCandidate] = {}
        for idx, cand in enumerate(candidates, start=1):
            key = _semantic_key(cand)
            scores[key] += 1.0 / (self._k + idx)
            if key not in best or cand.score > best[key].score:
                best[key] = cand

        result = []
        for key, cand in best.items():
            cand.signals = dict(cand.signals)
            cand.signals["fused_score"] = scores[key]
            result.append(cand)
        result.sort(key=lambda c: c.signals["fused_score"], reverse=True)
        return result[:top_k]


class MMRDeduplicator:
    """Simple max-marginal-relevance de-duplicator.

    It penalizes candidates whose normalized triple is textually close
    to already-selected candidates. This is intentionally lightweight;
    embedding-based diversity can be added behind the same method later.
    """

    def __init__(self, diversity_weight: float = 0.35) -> None:
        if not 0.0 <= diversity_weight <= 1.0:
            raise ValueError("diversity_weight must be within [0, 1]")
        self._diversity_weight = diversity_weight

    def dedupe(
        self,
        candidates: Iterable[RecallCandidate],
        *,
        top_k: int,
    ) -> list[RecallCandidate]:
        remaining = list(candidates)
        selected: list[RecallCandidate] = []
        while remaining and len(selected) < top_k:
            best = max(remaining, key=lambda c: self._mmr_score(c, selected))
            selected.append(best)
            remaining.remove(best)
        return selected

    def _mmr_score(
        self,
        candidate: RecallCandidate,
        selected: list[RecallCandidate],
    ) -> float:
        fused_score = float(candidate.signals.get("fused_score", candidate.score))
        if not selected:
            return fused_score
        max_similarity = max(
            _jaccard(_candidate_text(candidate), _candidate_text(s)) for s in selected
        )
        return (
            1.0 - self._diversity_weight
        ) * fused_score - self._diversity_weight * max_similarity


def _semantic_key(candidate: RecallCandidate) -> tuple[str, str, str]:
    fact = candidate.fact
    return (
        fact.subject.strip().casefold(),
        fact.predicate.strip().casefold(),
        fact.object.strip().casefold(),
    )


def _candidate_text(candidate: RecallCandidate) -> str:
    fact = candidate.fact
    return f"{fact.subject} {fact.predicate} {fact.object}".casefold()


def _jaccard(left: str, right: str) -> float:
    a = set(left.split())
    b = set(right.split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


__all__ = ["Fuser", "MMRDeduplicator", "ReciprocalRankFuser", "WeightedFuser"]
