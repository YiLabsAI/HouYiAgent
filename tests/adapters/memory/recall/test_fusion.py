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

    def test_orders_by_normalized_score(self) -> None:
        hits = [
            make_candidate("a", "p", "low", 1.0, RetrieverKind.RAW_TURN),
            make_candidate("b", "p", "high", 0.2, RetrieverKind.ENTITY_STATE),
        ]

        fused = WeightedFuser().fuse(hits, top_k=2)

        # Uniform weights mean raw_turn (1.0) beats entity_state (0.2)
        assert [hit.fact.object for hit in fused] == ["low", "high"]

    def test_mild_weight_preference(self) -> None:
        weights = {
            RetrieverKind.RAW_TURN: 1.0,
            RetrieverKind.ENTITY_STATE: 1.2,
        }
        hits = [
            make_candidate("a", "p", "first", 1.0, RetrieverKind.RAW_TURN),
            make_candidate("b", "p", "second", 1.0, RetrieverKind.ENTITY_STATE),
        ]

        fused = WeightedFuser(kind_weights=weights).fuse(hits, top_k=2)

        # Same base score, ENTITY_STATE weight (1.2) beats RAW_TURN weight (1.0)
        assert [hit.fact.object for hit in fused] == ["second", "first"]

    def test_cross_retriever_competition(self) -> None:
        hits = [
            make_candidate("a", "p", "timeline_best", 3.8, RetrieverKind.TIMELINE),
            make_candidate("b", "p", "entity_weak", 0.1, RetrieverKind.ENTITY_STATE),
        ]

        fused = WeightedFuser().fuse(hits, top_k=2)

        # Timeline normalizes to 1.0. EntityState normalizes to 1.0 (it's the only one).
        # Tie breaker falls to order of iteration / stability, but we can test
        # that it doesn't just blindly prefer entity_state like before.
        # Actually, let's make timeline's max = 3.8, and another timeline = 2.0.
        hits = [
            make_candidate("a", "p", "timeline_best", 3.8, RetrieverKind.TIMELINE),
            make_candidate("c", "p", "timeline_mid", 2.0, RetrieverKind.TIMELINE),
            make_candidate("b", "p", "entity_weak", 0.1, RetrieverKind.ENTITY_STATE),
        ]

        fused = WeightedFuser().fuse(hits, top_k=3)

        # timeline_best becomes 1.0, entity_weak becomes 1.0.
        # timeline_mid becomes 2.0 / 3.8 = ~0.52.
        # timeline_best and entity_weak should both be > timeline_mid.
        objects = [hit.fact.object for hit in fused]
        assert "timeline_best" in objects[:2]
        assert "entity_weak" in objects[:2]
        assert objects[2] == "timeline_mid"

    def test_normalizes_unit_range(self) -> None:
        hits = [
            make_candidate("a", "p", "best", 0.5, RetrieverKind.ENTITY_STATE),
            make_candidate("b", "p", "worst", 0.1, RetrieverKind.ENTITY_STATE),
        ]

        fused = WeightedFuser().fuse(hits, top_k=2)

        # s_max is 0.5. Instead of skipping normalization because s_max <= 1.0,
        # it should normalize by / 0.5.
        scores = {hit.fact.object: hit.score for hit in fused}
        assert scores["best"] == 1.0
        assert scores["worst"] == 0.2


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


class TestReciprocalRankFuserRanking:
    def test_keeps_event_candidate(self) -> None:
        # 23 equal-scored entity_state rows (the dateless majority) followed
        # by one high-value event row last in arrival order. min-max fusion
        # buries the event below the 23 ties and cuts it; RRF ranks per kind
        # so the event (rank-1 in its own kind) survives the top_k cut.
        hits = [
            make_candidate("james", "likes", f"thing{i}", 5.0, RetrieverKind.ENTITY_STATE)
            for i in range(23)
        ]
        hits.append(make_candidate("james", "went_bowling", "bowling", 7.0, RetrieverKind.EVENT))

        rrf = ReciprocalRankFuser().fuse(hits, top_k=20)
        weighted = WeightedFuser().fuse(hits, top_k=20)

        rrf_objects = {c.fact.object for c in rrf}
        weighted_objects = {c.fact.object for c in weighted}
        assert "bowling" in rrf_objects
        assert "bowling" not in weighted_objects

    def test_multi_source_accumulates(self) -> None:
        shared_es = make_candidate("u", "visited", "tokyo", 5.0, RetrieverKind.ENTITY_STATE)
        shared_event = make_candidate("u", "visited", "tokyo", 5.0, RetrieverKind.EVENT)
        single = make_candidate("u", "visited", "osaka", 5.0, RetrieverKind.ENTITY_STATE)

        fused = ReciprocalRankFuser().fuse([shared_es, shared_event, single], top_k=5)
        by_obj = {c.fact.object: c for c in fused}

        assert by_obj["tokyo"].signals["fused_score"] > by_obj["osaka"].signals["fused_score"]
        assert by_obj["tokyo"].signals["contributors"] == ["entity_state", "event"]

    def test_weight_prioritizes_kind(self) -> None:
        timeline = make_candidate("u", "born", "1990", 1.0, RetrieverKind.TIMELINE)
        entity = make_candidate("u", "lives", "paris", 1.0, RetrieverKind.ENTITY_STATE)

        fused = ReciprocalRankFuser(
            kind_weights={RetrieverKind.TIMELINE: 1.3, RetrieverKind.ENTITY_STATE: 1.0}
        ).fuse([entity, timeline], top_k=2)

        assert fused[0].matched_by == RetrieverKind.TIMELINE

    def test_event_time_tiebreak(self) -> None:
        dated = RecallCandidate(
            fact=AtomicFact(
                subject="u",
                predicate="visited",
                object="tokyo",
                certainty=Certainty.CERTAIN,
                source_anchor="d1",
                event_time="2022-03-16",
            ),
            score=5.0,
            matched_by=RetrieverKind.ENTITY_STATE,
            retriever_name="entity_state",
        )
        undated_higher = RecallCandidate(
            fact=AtomicFact(
                subject="u",
                predicate="visited",
                object="tokyo",
                certainty=Certainty.CERTAIN,
                source_anchor="d2",
            ),
            score=7.0,
            matched_by=RetrieverKind.TIMELINE,
            retriever_name="timeline",
        )

        fused = ReciprocalRankFuser().fuse([undated_higher, dated], top_k=1)

        assert len(fused) == 1
        assert fused[0].fact.event_time == "2022-03-16"


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
