from __future__ import annotations

from houyi.adapters.memory.recall.idk_guard import IDKGuard, IDKGuardConfig
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallReason,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, Certainty


def make_candidate(score: float, signals: dict[str, object] | None = None) -> RecallCandidate:
    return RecallCandidate(
        fact=AtomicFact(
            subject="user",
            predicate="likes",
            object="coffee",
            certainty=Certainty.CERTAIN,
            source_anchor="s1",
        ),
        score=score,
        matched_by=RetrieverKind.ENTITY_STATE,
        retriever_name="fake",
        signals=signals or {},
    )


def test_no_candidates() -> None:
    result = IDKGuard().evaluate(query_type=QueryType.FACTUAL_LOOKUP, candidates=[])

    assert result.reason == RecallReason.NO_CANDIDATES
    assert result.suggested_action == "admit_unknown"
    assert result.trace["guard"]["signal"] == "no_candidates"


def test_negation_absence() -> None:
    result = IDKGuard().evaluate(query_type=QueryType.NEGATION_CHECK, candidates=[])

    assert result.reason == RecallReason.EXPLICIT_ABSENCE
    assert result.suggested_action == "state_absence"


def test_low_evidence() -> None:
    result = IDKGuard().evaluate(
        query_type=QueryType.FACTUAL_LOOKUP,
        candidates=[make_candidate(0.4)],
    )

    assert result.reason == RecallReason.LOW_EVIDENCE
    assert result.trace["guard"]["top_score"] == 0.4


def test_fused_score_used() -> None:
    result = IDKGuard(IDKGuardConfig(evidence_threshold=0.8)).evaluate(
        query_type=QueryType.FACTUAL_LOOKUP,
        candidates=[make_candidate(0.1, {"fused_score": 0.9})],
    )

    assert result.reason == RecallReason.SUFFICIENT


def test_rerank_score_used() -> None:
    result = IDKGuard(IDKGuardConfig(evidence_threshold=0.8)).evaluate(
        query_type=QueryType.FACTUAL_LOOKUP,
        candidates=[make_candidate(0.1, {"fused_score": 0.2, "rerank_score": 0.9})],
    )

    assert result.reason == RecallReason.SUFFICIENT


def test_low_coverage() -> None:
    result = IDKGuard().evaluate(
        query_type=QueryType.RELATIONAL_CHAIN,
        candidates=[make_candidate(1.0, {"evidence_coverage": 0.1})],
    )

    assert result.reason == RecallReason.LOW_EVIDENCE
    assert result.trace["guard"]["evidence_coverage"] == 0.1


def test_contradiction_blocks() -> None:
    result = IDKGuard().evaluate(
        query_type=QueryType.FACTUAL_LOOKUP,
        candidates=[make_candidate(1.0, {"contradicts": True})],
    )

    assert result.reason == RecallReason.CONTRADICTING_EVIDENCE
    assert result.suggested_action == "ask_user_clarify"


def test_recency_resolves() -> None:
    result = IDKGuard().evaluate(
        query_type=QueryType.FACTUAL_LOOKUP,
        candidates=[make_candidate(1.0, {"contradicts": True, "recency_winner": True})],
    )

    assert result.reason == RecallReason.SUFFICIENT
