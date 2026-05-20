from __future__ import annotations

from houyi.adapters.memory.recall.rerank import EvidenceAwareReranker
from houyi.adapters.memory.recall.types import QueryType, RecallCandidate, RetrieverKind
from houyi.adapters.memory.types import AtomicFact, Certainty


def make_candidate(
    subject: str,
    predicate: str,
    obj: str,
    *,
    score: float = 1.0,
    signals: dict[str, object] | None = None,
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
        matched_by=RetrieverKind.ITERATIVE,
        retriever_name="fake",
        signals=signals or {},
    )


def test_rerank_source_evidence() -> None:
    weak = make_candidate("user", "likes", "tea", score=1.0)
    sourced = make_candidate(
        "user",
        "likes",
        "coffee",
        score=1.0,
        signals={"source_rehydrated": True},
    )

    ranked = EvidenceAwareReranker().rerank(
        query_type=QueryType.FACTUAL_LOOKUP,
        candidates=[weak, sourced],
        top_k=2,
    )

    assert [hit.fact.object for hit in ranked] == ["coffee", "tea"]
    assert ranked[0].signals["evidence_coverage"] > ranked[1].signals["evidence_coverage"]


def test_rerank_chain_complete() -> None:
    first = make_candidate(
        "Martin",
        "manager",
        "Alice",
        signals={"iteration_round": 1},
    )
    second = make_candidate(
        "Alice",
        "email",
        "a@example.com",
        signals={"iteration_round": 2},
    )

    ranked = EvidenceAwareReranker().rerank(
        query_type=QueryType.RELATIONAL_CHAIN,
        candidates=[first, second],
        top_k=2,
    )

    assert all(hit.signals["chain_evidence_complete"] is True for hit in ranked)
    assert all(hit.signals["evidence_coverage"] >= 1.0 for hit in ranked)


def test_rerank_chain_partial() -> None:
    partial = make_candidate(
        "Martin",
        "manager",
        "Alice",
        signals={"iteration_round": 1},
    )

    ranked = EvidenceAwareReranker().rerank(
        query_type=QueryType.RELATIONAL_CHAIN,
        candidates=[partial],
        top_k=1,
    )

    assert ranked[0].signals["chain_evidence_complete"] is False
    assert ranked[0].signals["evidence_coverage"] < 0.5


def test_rerank_respects_top_k() -> None:
    hits = [make_candidate("u", "p", str(i), score=float(i)) for i in range(4)]

    ranked = EvidenceAwareReranker().rerank(
        query_type=QueryType.FACTUAL_LOOKUP,
        candidates=hits,
        top_k=2,
    )

    assert len(ranked) == 2
    assert [hit.fact.object for hit in ranked] == ["3", "2"]
