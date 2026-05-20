from __future__ import annotations

import pytest
from pydantic import ValidationError

from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallQuery,
    RecallReason,
    RecallResult,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, Certainty


def make_candidate(score: float = 1.0) -> RecallCandidate:
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
    )


def test_query_requires_text() -> None:
    with pytest.raises(ValidationError):
        RecallQuery(text=" ")


def test_query_requires_top_k() -> None:
    with pytest.raises(ValidationError):
        RecallQuery(text="x", top_k=0)


def test_candidate_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        make_candidate(float("nan"))


def test_result_helpers() -> None:
    candidate = make_candidate()
    result = RecallResult(
        candidates=[candidate],
        query_type=QueryType.FACTUAL_LOOKUP,
        reason=RecallReason.SUFFICIENT,
    )

    assert result.is_sufficient() is True
    assert result.top() is candidate


def test_result_top_empty() -> None:
    result = RecallResult(
        candidates=[],
        query_type=QueryType.FACTUAL_LOOKUP,
        reason=RecallReason.NO_CANDIDATES,
    )

    assert result.is_sufficient() is False
    assert result.top() is None


def test_context_requires_reads() -> None:
    with pytest.raises(ValidationError):
        RetrieverContext(max_source_reads=0)
