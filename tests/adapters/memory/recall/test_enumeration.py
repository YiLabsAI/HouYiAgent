"""Tests for enumeration-aware recall boosting."""

from __future__ import annotations

import pytest

from houyi.adapters.memory.recall.enumeration import (
    EnumerationBooster,
    detect_enumeration_category,
)
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact


def _candidate(predicate: str, obj: str, *, subject: str = "Tim") -> RecallCandidate:
    return RecallCandidate(
        fact=AtomicFact(
            subject=subject,
            predicate=predicate,
            object=obj,
            certainty="certain",
            source_anchor="anchor",
        ),
        score=5.0,
        matched_by=RetrieverKind.ENTITY_STATE,
        retriever_name="entity_state",
    )


class TestDetectCategory:
    """Category head-noun extraction from enumeration intents."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("What books has Tim read?", "book"),
            ("How many video game tournaments has Nate participated in?", "tournament"),
            ("What kind of places have Andrew and his girlfriend checked out?", "place"),
            ("Which movies did Maria watch?", "movie"),
            ("What activities did Sam enjoy?", "activitie"),
        ],
    )
    def test_detects(self, query: str, expected: str) -> None:
        """Enumeration intents resolve to the stemmed head noun."""
        assert detect_enumeration_category(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "When did Jon lose his job?",
            "Who did Maria have dinner with?",
            "What might John's financial status be?",
            "",
        ],
    )
    def test_rejects(self, query: str) -> None:
        """Non-enumeration questions yield no category."""
        assert detect_enumeration_category(query) is None


class TestBoosterApply:
    """Family detection and in-place score boosting."""

    @pytest.mark.asyncio
    async def test_anchor_boost(self) -> None:
        """Candidates mentioning the category stem are boosted."""
        cands = [
            _candidate("reads_book", "Harry Potter"),
            _candidate("lives_in", "city"),
        ]
        boosted = await EnumerationBooster().apply("What books has Tim read?", cands)
        assert boosted == 1
        assert cands[0].score == pytest.approx(13.0)
        assert cands[0].signals.get("enumeration_family") == "book"
        assert cands[0].signals.get("enumeration_tier") == "instance"
        assert cands[1].score == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_closure_boost(self) -> None:
        """Object co-occurring with an anchored text joins the family."""
        cands = [
            _candidate("reads_book", "The Hobbit", subject="John"),
            _candidate("likes", "The Hobbit"),
            _candidate("likes", "tea"),
        ]
        boosted = await EnumerationBooster().apply("What books has Tim read?", cands)
        assert boosted == 2
        assert cands[0].score == pytest.approx(7.0)
        assert cands[0].signals.get("enumeration_tier") == "mention"
        assert cands[1].score == pytest.approx(13.0)
        assert cands[1].signals.get("enumeration_tier") == "instance"
        assert cands[2].score == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_no_intent_noop(self) -> None:
        """Non-enumeration query leaves all scores untouched."""
        cands = [_candidate("reads_book", "Harry Potter")]
        boosted = await EnumerationBooster().apply("Who is Tim?", cands)
        assert boosted == 0
        assert cands[0].score == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_fts_closure(self) -> None:
        """Backend FTS anchors widen the closure beyond candidates."""

        class _Record:
            content = "John reads book The Hobbit"

        class _Backend:
            def search_fts(self, query: str, limit: int = 50) -> list[tuple[object, float]]:
                return [(_Record(), 1.0)]

        cands = [_candidate("likes", "The Hobbit")]
        boosted = await EnumerationBooster(_Backend()).apply("What books has Tim read?", cands)
        assert boosted == 1
        assert cands[0].signals.get("enumeration_family") == "book"
