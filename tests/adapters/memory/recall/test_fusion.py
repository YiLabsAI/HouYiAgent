from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.fusion import (
    MMRDeduplicator,
    ReciprocalRankFuser,
    WeightedFuser,
)
from houyi.adapters.memory.recall.types import RecallCandidate, RetrieverKind
from houyi.adapters.memory.types import AtomicFact, Certainty


def make_candidate(
    subject: str,
    predicate: str,
    obj: str,
    score: float,
    kind: RetrieverKind = RetrieverKind.ENTITY_STATE,
) -> RecallCandidate:
    return RecallCandidate(
        fact=AtomicFact(
            subject=subject,
            predicate=predicate,
            object=obj,
            certainty=Certainty.CERTAIN,
            source_anchor=f"{subject}-{predicate}-{obj}",
        ),
        score=score,
        matched_by=kind,
        retriever_name=kind.value,
    )


class TestWeightedFuser:
    def test_empty_top_k(self) -> None:
        assert WeightedFuser().fuse([make_candidate("a", "b", "c", 1.0)], top_k=0) == []

    def test_merges_duplicates(self) -> None:
        hits = [
            make_candidate("User", "likes", "Coffee", 0.4, RetrieverKind.RAW_TURN),
            make_candidate("user", "likes", "coffee", 0.2, RetrieverKind.ENTITY_STATE),
        ]

        fused = WeightedFuser().fuse(hits, top_k=5)

        assert len(fused) == 1
        assert fused[0].signals["duplicate_count"] == 2
        assert fused[0].signals["contributors"] == ["entity_state", "raw_turn"]

    def test_orders_by_weight(self) -> None:
        hits = [
            make_candidate("a", "p", "low", 1.0, RetrieverKind.RAW_TURN),
            make_candidate("b", "p", "high", 0.2, RetrieverKind.ENTITY_STATE),
        ]

        fused = WeightedFuser().fuse(hits, top_k=2)

        assert [hit.fact.object for hit in fused] == ["high", "low"]


class TestReciprocalRankFuser:
    def test_rejects_bad_k(self) -> None:
        with pytest.raises(ValueError):
            ReciprocalRankFuser(k=0)

    def test_merges_by_rank(self) -> None:
        hits = [
            make_candidate("a", "p", "one", 0.1),
            make_candidate("a", "p", "one", 0.9),
            make_candidate("b", "p", "two", 0.5),
        ]

        fused = ReciprocalRankFuser(k=10).fuse(hits, top_k=2)

        assert len(fused) == 2
        assert fused[0].signals["fused_score"] > 0
        assert fused[0].fact.object == "one"


class TestMMRDeduplicator:
    def test_rejects_weight(self) -> None:
        with pytest.raises(ValueError):
            MMRDeduplicator(diversity_weight=1.1)

    def test_respects_top_k(self) -> None:
        hits = [
            make_candidate("a", "p", "one", 3.0),
            make_candidate("b", "p", "two", 2.0),
        ]

        selected = MMRDeduplicator().dedupe(hits, top_k=1)

        assert len(selected) == 1
        assert selected[0].fact.object == "one"

    def test_penalizes_similarity(self) -> None:
        hits = [
            make_candidate("alpha", "likes", "coffee", 10.0),
            make_candidate("alpha", "likes", "coffee", 9.9),
            make_candidate("beta", "owns", "laptop", 9.0),
        ]

        selected = MMRDeduplicator(diversity_weight=0.9).dedupe(hits, top_k=2)

        assert [hit.fact.subject for hit in selected] == ["alpha", "beta"]
