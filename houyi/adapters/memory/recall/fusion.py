"""Candidate fusion and de-duplication for memory recall.

Retrievers produce local scores with different scales. This module
normalizes and merges those candidates into a single ordered list while
preserving original per-retriever signals for traceability.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from houyi.adapters.memory.recall.types import RecallCandidate, RetrieverKind

_DEFAULT_KIND_WEIGHTS: dict[RetrieverKind, float] = {
    RetrieverKind.ENTITY_STATE: 1.0,
    RetrieverKind.TIMELINE: 1.0,
    RetrieverKind.ITERATIVE: 1.0,
    RetrieverKind.RAW_TURN: 1.0,
    RetrieverKind.GRAPH: 1.0,
    RetrieverKind.VECTOR: 1.0,
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
        if s_max == 0.0:
            # Prevent division by zero; keep raw scores
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

        # 3. Group normalized candidates by near-duplicate detection
        grouped = _group_near_duplicates(normalized_cands)

        fused = [self._fuse_group(group) for group in grouped]
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
    """Rank-based fusion across retriever kinds.

    Each retriever kind is ranked independently by its own local score,
    and a candidate's fused score is the weighted sum of reciprocal ranks
    over the kinds that surfaced it::

        score(d) = Σ_kind  w(kind) · 1 / (k + rank_kind(d))

    Ranking *per kind* (rather than by global arrival order or by raw
    score magnitude) makes fusion robust to the incomparable score scales
    different retrievers emit: every kind's top hit earns the same
    reciprocal-rank weight, so a numerous low-diversity source (e.g. many
    equal-scored entity_state rows) can no longer monopolise the candidate
    budget purely by arrival order. Near-duplicate candidates surfaced by
    multiple kinds accumulate one term per kind, rewarding cross-source
    agreement. Final scores are min-max normalized to [0, 1] so they stay
    on the same scale as the downstream reranker's coverage bonuses
    (rerank_score = fused_score + coverage + ...).
    """

    def __init__(self, k: int = 60, kind_weights: dict[RetrieverKind, float] | None = None) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        self._k = k
        self._weights = dict(kind_weights or {})

    def _weight(self, kind: RetrieverKind) -> float:
        return self._weights.get(kind, 1.0)

    def fuse(
        self,
        candidates: Iterable[RecallCandidate],
        *,
        top_k: int,
    ) -> list[RecallCandidate]:
        if top_k <= 0:
            return []
        candidates = list(candidates)
        if not candidates:
            return []

        grouped = _group_near_duplicates(candidates)
        cand_to_group_id: dict[int, int] = {}
        for group_id, group in enumerate(grouped):
            for cand in group:
                cand_to_group_id[id(cand)] = group_id

        scores, best = self._rank_groups(candidates, cand_to_group_id)
        max_score = max(scores.values()) if scores else 0.0
        fused = [
            self._build_representative(best[group_id], grouped[group_id], raw, max_score)
            for group_id, raw in scores.items()
        ]
        fused.sort(key=lambda c: c.signals["fused_score"], reverse=True)
        return fused[:top_k]

    def _rank_groups(
        self,
        candidates: list[RecallCandidate],
        cand_to_group_id: dict[int, int],
    ) -> tuple[dict[int, float], dict[int, RecallCandidate]]:
        # Rank within each retriever kind by local score, then accumulate
        # weighted reciprocal-rank contributions onto the near-duplicate group.
        by_kind: dict[RetrieverKind, list[RecallCandidate]] = defaultdict(list)
        for cand in candidates:
            by_kind[cand.matched_by].append(cand)

        scores: dict[int, float] = defaultdict(float)
        best: dict[int, RecallCandidate] = {}
        for kind, kind_list in by_kind.items():
            weight = self._weight(kind)
            ranked = sorted(kind_list, key=lambda c: c.score, reverse=True)
            for rank, cand in enumerate(ranked, start=1):
                group_id = cand_to_group_id[id(cand)]
                scores[group_id] += weight / (self._k + rank)
                if self._prefers(cand, best.get(group_id)):
                    best[group_id] = cand
        return scores, best

    @staticmethod
    def _prefers(cand: RecallCandidate, current: RecallCandidate | None) -> bool:
        # Prefer the candidate carrying event_time (fact-relevant date) over
        # one that only has a system valid_from, even at a lower local score —
        # temporal answers depend on that date. A higher local score only wins
        # when it does not discard an already-dated representative.
        if current is None:
            return True
        cand_dated = bool(cand.fact.event_time)
        if cand_dated and not current.fact.event_time:
            return True
        return (cand_dated or not current.fact.event_time) and cand.score > current.score

    @staticmethod
    def _build_representative(
        rep: RecallCandidate,
        members: list[RecallCandidate],
        raw: float,
        max_score: float,
    ) -> RecallCandidate:
        all_contributors: list[str] = []
        for member in members:
            all_contributors.extend(member.signals.get("contributors", [member.matched_by.value]))
        rep.signals = dict(rep.signals)
        rep.signals["raw_score"] = rep.score
        rep.signals["rrf_score"] = raw
        rep.signals["fused_score"] = raw / max_score if max_score > 0 else 0.0
        rep.signals["contributors"] = sorted(set(all_contributors))
        rep.signals["duplicate_count"] = len(members)
        for other in members:
            if other is rep:
                continue
            for key, value in other.signals.items():
                rep.signals.setdefault(key, value)
        return rep


# Source-anchor overlap redundancy thresholds.
#
# When two facts derive evidence from the same conversation turns they are
# cross-predicate re-statements of one underlying event/state: schema-on-read
# splits one event into several compounds that still share turn sources. A
# high Jaccard over the source-anchor set flags this without any text/semantic
# heuristics, and does NOT penalise facts that merely share a subject but
# come from different turns (a genuine multi-place enumeration stays intact).
# Threshold chosen from empirical evidence: intra-topic compound pairs
# cluster at Jaccard 0.3-0.5, cross-topic pairs at <=0.2.
_SOURCE_OVERLAP_THRESHOLD = 0.3
_SOURCE_OVERLAP_REDUNDANCY = 0.85


def _fact_sources(candidate: RecallCandidate) -> set[str]:
    """Return the set of conversation-turn source anchors backing a candidate.

    Compound facts carry their merged members in the ``compound_source_anchors``
    signal; single-source facts fall back to their own ``source_anchor``.
    """
    sig = candidate.signals
    if sig:
        anchors = sig.get("compound_source_anchors")
        if anchors:
            return set(anchors)
    src = candidate.fact.source_anchor
    return {src} if src else set()


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
        diversity: float | None = None,
    ) -> list[RecallCandidate]:
        """Select top_k via max-marginal-relevance.

        The 'diversity' argument overrides the instance diversity weight for this
        call. Enumeration/aggregation queries ("what activities/items
        has X") pass a high value so the budget spreads across distinct
        facts (coverage) instead of being decided by arbitrary tie-order
        among same-scored candidates. Single-answer lookups keep the low
        default so relevance dominates.
        """
        weight = self._diversity_weight if diversity is None else diversity
        weight = min(1.0, max(0.0, weight))
        remaining = list(candidates)
        if not remaining:
            return []
        # Normalize relevance to [0, 1] across the candidate pool so the
        # diversity penalty (also in [0, 1]) is on a comparable scale. With
        # raw rerank/fused scores (range ~0-15) the penalty was negligible,
        # making MMR a near-pure score sort that let many rephrasings of one
        # fact (or one entity) crowd out diverse evidence.
        relevance = {id(c): self._relevance(c) for c in remaining}
        lo = min(relevance.values())
        hi = max(relevance.values())
        span = hi - lo
        norm = {cid: ((val - lo) / span if span > 0 else 1.0) for cid, val in relevance.items()}
        # Pre-compute each candidate's source-anchor set once. _redundancy is
        # called O(N*K) times in the selection loop below (every remaining
        # candidate vs every selected candidate); rebuilding the frozenset on
        # each call makes long-conversation cases with large compound anchors
        # pathologically slow. Cache by object identity to keep the hot path
        # cheap without polluting signals.
        src_cache: dict[int, frozenset[str]] = {
            id(c): frozenset(_fact_sources(c)) for c in remaining
        }
        selected: list[RecallCandidate] = []
        while remaining and len(selected) < top_k:
            best = max(
                remaining,
                key=lambda c: self._mmr_score(c, selected, norm, weight, src_cache),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    @staticmethod
    def _relevance(candidate: RecallCandidate) -> float:
        return float(
            candidate.signals.get(
                "rerank_score", candidate.signals.get("fused_score", candidate.score)
            )
        )

    def _mmr_score(
        self,
        candidate: RecallCandidate,
        selected: list[RecallCandidate],
        norm: dict[int, float],
        diversity_weight: float,
        src_cache: dict[int, frozenset[str]],
    ) -> float:
        relevance = norm[id(candidate)]
        if not selected:
            return relevance
        redundancy = max(self._redundancy(candidate, s, src_cache) for s in selected)
        return (1.0 - diversity_weight) * relevance - diversity_weight * redundancy

    @staticmethod
    def _redundancy(
        a: RecallCandidate,
        b: RecallCandidate,
        src_cache: dict[int, frozenset[str]] | None = None,
    ) -> float:
        # Check compound group key equality
        cgk_a = a.signals and a.signals.get("compound_group_key")
        cgk_b = b.signals and b.signals.get("compound_group_key")
        if cgk_a and cgk_b and cgk_a == cgk_b:
            return 1.0

        # Source-anchor overlap: two facts that derive evidence from the same
        # conversation turns are cross-predicate re-statements of one
        # underlying event/state (schema-on-read splits one event across
        # multiple predicates into several compounds that still share turn
        # sources). A high Jaccard over the source-anchor set flags this
        # without any text/semantic heuristics, and crucially does NOT
        # penalise facts that merely share a subject but come from different
        # turns (e.g. a genuine multi-place enumeration).
        if src_cache is not None:
            sources_a = src_cache.get(id(a))
            sources_b = src_cache.get(id(b))
        else:
            sources_a = frozenset(_fact_sources(a))
            sources_b = frozenset(_fact_sources(b))
        if sources_a and sources_b:
            union = sources_a | sources_b
            overlap = len(sources_a & sources_b)
            if overlap and overlap / len(union) >= _SOURCE_OVERLAP_THRESHOLD:
                return _SOURCE_OVERLAP_REDUNDANCY

        subj_a = a.fact.subject.strip().casefold()
        subj_b = b.fact.subject.strip().casefold()
        pred_a = a.fact.predicate.strip().casefold()
        pred_b = b.fact.predicate.strip().casefold()
        obj_a = a.fact.object.strip().casefold()
        obj_b = b.fact.object.strip().casefold()

        # If they share subject & predicate and one of them is a compound,
        # treat as partially redundant.
        if subj_a == subj_b and pred_a == pred_b:
            if cgk_a or cgk_b:
                return 0.7
            if obj_a == obj_b:
                # Same (subject, predicate, object) but different qualifiers
                # are NOT fully redundant. Enumeration members (e.g. "shares
                # activity with GF" where qualifier distinguishes boardgames
                # vs wine tasting) carry distinct answers. Without this check
                # MMR treats qualifier-bearing facts as perfect duplicates and
                # collapses enumeration members to a single representative.
                quals_a = getattr(a.fact, "qualifiers", None) or {}
                quals_b = getattr(b.fact, "qualifiers", None) or {}
                if quals_a and quals_b and quals_a != quals_b:
                    # Different qualifier values -> partially redundant only
                    return 0.5
                # Truly identical triple (no distinguishing qualifiers)
                return 1.0
        return 0.0


def _light_stem(word: str) -> str:
    w = word.strip().lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        w = w[:-3] + "y"
    elif w.endswith("es"):
        w = w[:-2]
    elif w.endswith("ing"):
        w = w[:-3]
    elif w.endswith("ed"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def _stemmed_words(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {_light_stem(w) for w in cleaned.split() if w}


def _object_head(obj: str | None) -> str:
    """Normalize a fact object to an order-independent identity key.

    Two facts about the same subject that resolve to the same concrete
    object are one evidence point regardless of predicate wording, so
    their objects must compare equal. Light-stemming plus a sorted token
    join makes "board games" == "board game" and "watching movies" ==
    "watch movie" while keeping distinct members ("wine tasting") apart.
    Returns "" for empty objects so callers can skip the equality test.
    """
    words = _stemmed_words(obj or "")
    return " ".join(sorted(words))


def _valid_day(valid_from: float | None) -> str:
    """Reduce an epoch valid_from to day granularity for identity.

    Day granularity separates genuine event occurrences (distinct
    dates) while merging re-extractions of the same fact: overlapping
    ingestion windows re-extract identical facts seconds apart, and a
    sub-day key component would let those duplicates crowd the budget.
    """
    if not valid_from:
        return ""
    try:
        return datetime.fromtimestamp(float(valid_from), tz=UTC).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _semantic_key(candidate: RecallCandidate) -> tuple[str, str, str, str]:
    """Identity key for cross-retriever fusion.

    Includes temporal validity at day granularity: the same triple
    asserted on different days is a *different event occurrence*
    (e.g. one tournament entry per date). Collapsing those breaks
    counting and enumeration questions, while same-day duplicates
    (same row via different retrievers, or window re-extractions)
    still merge.
    """
    fact = candidate.fact
    return (
        fact.subject.strip().casefold(),
        fact.predicate.strip().casefold(),
        fact.object.strip().casefold(),
        _valid_day(fact.valid_from),
    )


def _group_near_duplicates(candidates: Iterable[RecallCandidate]) -> list[list[RecallCandidate]]:
    """Group candidates by exact proposition identity (subject, predicate, object) and day, or by compound key."""
    grouped_map: dict[tuple[Any, ...], list[RecallCandidate]] = defaultdict(list)
    for cand in candidates:
        if cand.signals and cand.signals.get("compound_group_key"):
            key = cand.signals["compound_group_key"]
            grouped_map[("compound", key[0], key[1])].append(cand)
        else:
            key = (
                cand.fact.subject.strip().casefold(),
                cand.fact.predicate.strip().casefold(),
                cand.fact.object.strip().casefold(),
                _valid_day(cand.fact.valid_from),
            )
            grouped_map[("exact", *key)].append(cand)
    return list(grouped_map.values())


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
