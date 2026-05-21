from __future__ import annotations

from houyi.adapters.memory.reasoner import (
    DeterministicReasoningPolicy,
    MemoryReasoner,
    TemporalTurn,
    answer_from_turn_evidence,
)
from houyi.adapters.memory.types import MemoryRecall, MemoryRecord


async def test_match_returns_content():
    policy = DeterministicReasoningPolicy(min_overlap_ratio=0.5)
    reasoner = MemoryReasoner([policy])
    records = [MemoryRecord(key="pref", content="User preference: python and uv")]
    recalls = [MemoryRecall(memory_id=records[0].record_id, score=0.9)]

    result = await reasoner.answer("what is my python preference", recalls, records)

    assert result.abstained is False
    assert "python" in result.answer.lower()


async def test_empty_returns_idk():
    reasoner = MemoryReasoner([])
    result = await reasoner.answer("anything", [], [])

    assert result.abstained is True
    assert result.reason == "no_policy_match"


def test_turn_rule_yesterday():
    turns = [
        TemporalTurn(
            turn_id="D1:3",
            speaker_id="Caroline",
            text="I went to a support group yesterday.",
            occurred_at="1:56 pm on 8 May, 2023",
        )
    ]

    answer = answer_from_turn_evidence(
        query="When did Caroline go to the support group?",
        turns=turns,
        evidence_ids=("D1:3",),
    )

    assert answer == "7 May 2023"
