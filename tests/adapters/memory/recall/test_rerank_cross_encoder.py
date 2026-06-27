from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.rerank import EvidenceAwareReranker
from houyi.adapters.memory.recall.rerank_cross_encoder import (
    CrossEncoderReranker,
    FallbackReranker,
)
from houyi.adapters.memory.recall.types import QueryType, RecallCandidate, RetrieverKind
from houyi.adapters.memory.types import AtomicFact, Certainty


def _candidate(subject: str, predicate: str, obj: str, score: float = 0.0) -> RecallCandidate:
    fact = AtomicFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        certainty=Certainty.CERTAIN,
        source_anchor="t1",
    )
    return RecallCandidate(
        fact=fact,
        score=score,
        matched_by=RetrieverKind.VECTOR,
        retriever_name="test",
    )


class _MockCrossEncoder:
    """Fake model whose predict returns a score per (query, doc) pair.

    Scores are driven by how many query tokens appear in the document text,
    so the test can assert ordering without a real model download.
    """

    def __init__(self) -> None:
        self.loaded = True

    def predict(self, pairs, batch_size=32):
        return [self._score(q, d) for q, d in pairs]

    @staticmethod
    def _score(query: str, doc: str) -> float:
        qt = set(query.lower().split())
        dt = set(doc.lower().split())
        return float(len(qt & dt))


class TestCrossEncoderReranker:
    """Cross-encoder reranker with a mock model."""

    @pytest.fixture
    def reranker(self):
        r = CrossEncoderReranker()
        r._model = _MockCrossEncoder()
        return r

    async def test_ranks_by_query_relevance(self, reranker) -> None:
        """Candidates sharing more query tokens score higher."""
        candidates = [
            _candidate("Evan", "hobby", "watercolor painting"),
            _candidate("Evan", "trip", "road trip to Jasper with family"),
        ]
        result = await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=candidates,
            top_k=2,
            query="where has Evan been on roadtrips with his family",
        )
        assert result[0].fact.object == "road trip to Jasper with family"

    async def test_requires_query(self, reranker) -> None:
        """Without query text the cross-encoder cannot score."""
        with pytest.raises(ValueError):
            await reranker.arerank(
                query_type=QueryType.FACTUAL_LOOKUP,
                candidates=[_candidate("a", "b", "c")],
                top_k=1,
                query=None,
            )

    async def test_preserves_unscored_tail(self, reranker) -> None:
        """Candidates beyond max_candidates are kept in the tail.

        Gold that sits in the fused tail (>max_candidates) must stay visible
        downstream so boost + MMR can still lift it; drop-tail lost it
        (regressed 8 cases). The scored window leads, the unscored tail
        follows.
        """
        many = [_candidate("s", "p", str(i)) for i in range(10)]
        reranker._max_candidates = 3
        result = await reranker.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=many,
            top_k=10,
            query="s p",
        )
        assert len(result) == 10


class TestFallbackReranker:
    """Chain rerankers so a failure degrades to the next tier."""

    async def test_first_tier_wins(self) -> None:
        class _Ok:
            async def arerank(self, *, query_type, candidates, top_k, query=None):
                return list(candidates)[:top_k]

            def rerank(self, **kw):
                raise RuntimeError

        chain = FallbackReranker([_Ok(), EvidenceAwareReranker()])
        result = await chain.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[_candidate("a", "b", "c")],
            top_k=1,
            query="q",
        )
        assert len(result) == 1

    async def test_falls_through_on_error(self) -> None:
        class _Boom:
            async def arerank(self, *, query_type, candidates, top_k, query=None):
                raise RuntimeError("model down")

            def rerank(self, **kw):
                raise RuntimeError

        chain = FallbackReranker([_Boom(), EvidenceAwareReranker()])
        result = await chain.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[_candidate("a", "b", "c")],
            top_k=1,
            query="q",
        )
        # Heuristic tier caught the failure and produced a result.
        assert len(result) == 1

    async def test_all_fail_returns_input(self) -> None:
        class _Boom:
            async def arerank(self, *, query_type, candidates, top_k, query=None):
                raise RuntimeError

            def rerank(self, **kw):
                raise RuntimeError

        chain = FallbackReranker([_Boom()])
        result = await chain.arerank(
            query_type=QueryType.FACTUAL_LOOKUP,
            candidates=[_candidate("a", "b", "c")],
            top_k=1,
            query="q",
        )
        assert len(result) == 1
