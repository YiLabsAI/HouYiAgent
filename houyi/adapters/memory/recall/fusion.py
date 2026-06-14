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

        # Group candidates by near-duplicate detection
        grouped = _group_near_duplicates(candidates)

        scores: dict[int, float] = defaultdict(float)
        best: dict[int, RecallCandidate] = {}
        cand_to_group_id = {}
        for group_id, group in enumerate(grouped):
            for cand in group:
                cand_to_group_id[id(cand)] = group_id

        for idx, cand in enumerate(candidates, start=1):
            mapped_group_id = cand_to_group_id.get(id(cand))
            if mapped_group_id is not None:
                scores[mapped_group_id] += 1.0 / (self._k + idx)
                # When multiple retrievers return the same fact, prefer the
                # candidate that carries event_time (fact-relevant time) over
                # one that only has valid_from (system timestamp). event_time
                # contains the human-readable date the LLM needs for temporal
                # questions. Without this tie-breaker, the fuser may pick a
                # Timeline candidate (event_time=None) over an EntityState
                # candidate (event_time="2020-03") solely because the
                # Timeline score is marginally higher, discarding the answer-
                # relevant date information.
                should_replace = False
                if (
                    mapped_group_id not in best
                    or (cand.fact.event_time and not best[mapped_group_id].fact.event_time)
                    or cand.score > best[mapped_group_id].score
                ):
                    should_replace = True
                if should_replace:
                    best[mapped_group_id] = cand

        fused = []
        for group_id, score in scores.items():
            cand = best[group_id]
            cand.signals = dict(cand.signals)
            cand.signals["fused_score"] = score
            fused.append(cand)

        fused.sort(key=lambda c: c.signals["fused_score"], reverse=True)
        return fused[:top_k]


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
        selected: list[RecallCandidate] = []
        while remaining and len(selected) < top_k:
            best = max(remaining, key=lambda c: self._mmr_score(c, selected, norm, weight))
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
    ) -> float:
        relevance = norm[id(candidate)]
        if not selected:
            return relevance
        redundancy = max(self._redundancy(candidate, s) for s in selected)
        return (1.0 - diversity_weight) * relevance - diversity_weight * redundancy

    @staticmethod
    def _redundancy(a: RecallCandidate, b: RecallCandidate) -> float:
        # Redundancy is keyed on *semantic fact identity*, not source
        # anchor. Two facts about the same subject that point at the same
        # concrete object ("Andrew has_relationship girlfriend" vs
        # "Andrew shares_activity_with girlfriend") rephrase one evidence
        # point, so they are fully redundant and one must yield its budget
        # slot. Distinct objects ("played board games" vs "went_to wine
        # tasting") are different members of the asked-about category and
        # must both survive — exactly what enumeration coverage needs.
        # Anchor-based keys failed here: vector candidates carry a UUID
        # anchor (not a turn id), and same-turn facts are often distinct
        # members. Fall back to lexical overlap when objects are absent.
        subj_a = a.fact.subject.strip().casefold()
        subj_b = b.fact.subject.strip().casefold()
        obj_a = _object_head(a.fact.object)
        obj_b = _object_head(b.fact.object)
        if subj_a == subj_b and obj_a and obj_a == obj_b:
            return 1.0
        return _jaccard(_candidate_text(a), _candidate_text(b))


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
    """Group candidates by (subject, valid_day) exact matches, then Jaccard similarity."""
    by_subj_day: dict[tuple[str, str], list[RecallCandidate]] = defaultdict(list)
    for cand in candidates:
        key = (cand.fact.subject.strip().casefold(), _valid_day(cand.fact.valid_from))
        by_subj_day[key].append(cand)

    grouped: list[list[RecallCandidate]] = []
    for subj_day_list in by_subj_day.values():
        sub_groups: list[list[RecallCandidate]] = []
        for cand in subj_day_list:
            found_group = False
            for group in sub_groups:
                rep = group[0]
                words_cand = _stemmed_words(f"{cand.fact.predicate} {cand.fact.object}")
                words_rep = _stemmed_words(f"{rep.fact.predicate} {rep.fact.object}")
                intersection = words_cand.intersection(words_rep)
                union = words_cand.union(words_rep)
                jaccard = len(intersection) / len(union) if union else 1.0

                if jaccard >= 0.6:
                    group.append(cand)
                    found_group = True
                    break
            if not found_group:
                sub_groups.append([cand])
        grouped.extend(sub_groups)
    return grouped


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
