from __future__ import annotations

from houyi.adapters.memory.reasoner import (
    DeterministicReasoningPolicy,
    LLMMemoryReasoningPolicy,
    MemoryReasoner,
    MemoryReasoningInput,
    TemporalTurn,
    answer_from_turn_evidence,
)
from houyi.adapters.memory.types import MemoryRecall, MemoryRecord


class MockLLMAdapter:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    async def chat(self, messages, temperature=0.0, max_tokens=1024):
        self.calls.append(messages)

        class Response:
            content = self.response_text

        return Response()


async def test_llm_reasoning_success():
    llm = MockLLMAdapter("John's suspected health issue is obesity.")
    policy = LLMMemoryReasoningPolicy(llm)
    records = [
        MemoryRecord(key="health", content="John has had busy week and struggles with obesity.")
    ]
    recalls = [MemoryRecall(memory_id=records[0].record_id, score=0.9)]
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=recalls, records=records
    )

    result = await policy.answer(request)
    assert result.abstained is False
    assert "obesity" in result.answer


async def test_llm_reasoning_timeout():
    class TimeoutLLM:
        async def chat(self, messages, temperature=0.0, max_tokens=1024):
            raise TimeoutError()

    policy = LLMMemoryReasoningPolicy(TimeoutLLM())
    records = [MemoryRecord(key="health", content="John has had busy week.")]
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=[], records=records
    )
    result = await policy.answer(request)
    assert result.abstained is True
    assert result.reason == "timeout"


async def test_llm_reasoning_exception():
    class FailedLLM:
        async def chat(self, messages, temperature=0.0, max_tokens=1024):
            raise RuntimeError("API error")

    policy = LLMMemoryReasoningPolicy(FailedLLM())
    records = [MemoryRecord(key="health", content="John has had busy week.")]
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=[], records=records
    )
    result = await policy.answer(request)
    assert result.abstained is True
    assert result.reason == "llm_failed"


async def test_llm_reasoning_idk():
    llm = MockLLMAdapter("[IDK]")
    policy = LLMMemoryReasoningPolicy(llm)
    records = [MemoryRecord(key="health", content="John has had busy week.")]
    recalls = [MemoryRecall(memory_id=records[0].record_id, score=0.9)]
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=recalls, records=records
    )

    result = await policy.answer(request)
    assert result.abstained is True
    assert result.reason == "llm_idk"


async def test_llm_reasoning_empty():
    llm = MockLLMAdapter("doesn't matter")
    policy = LLMMemoryReasoningPolicy(llm)
    request = MemoryReasoningInput(
        query="What is John's suspected health issue?", recalls=[], records=[]
    )

    result = await policy.answer(request)
    assert result.abstained is True
    assert result.reason == "no_candidates"


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


class TestTemporalRules:
    def test_yesterday(self):
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

    def test_relative_year(self):
        turns = [
            TemporalTurn(
                turn_id="D1:5",
                speaker_id="Audrey",
                text="I adopted my first dog 3 years ago.",
                occurred_at="2023-05-11",
            )
        ]
        answer = answer_from_turn_evidence(
            query="Which year did Audrey adopt her dog?",
            turns=turns,
            evidence_ids=("D1:5",),
        )
        assert answer == "2020"

    def test_first_trip(self):
        turns = [
            TemporalTurn(
                turn_id="D1:3",
                speaker_id="calvin",
                text="I had a great time in Tokyo.",
                occurred_at="26 March 2023",
            )
        ]
        answer = answer_from_turn_evidence(
            query="When did calvin first travel to tokyo?",
            turns=turns,
            evidence_ids=("D1:3",),
        )
        assert answer == "26 March 2023"

    def test_first_trip_range(self):
        turns = [
            TemporalTurn(
                turn_id="D1:2",
                speaker_id="dave",
                text="Hi calvin",
                occurred_at="22 March, 2023",
            ),
            TemporalTurn(
                turn_id="D1:3",
                speaker_id="calvin",
                text="I am traveling to Tokyo.",
                occurred_at="26 March, 2023",
            ),
        ]
        answer = answer_from_turn_evidence(
            query="When did calvin first travel to tokyo?",
            turns=turns,
            evidence_ids=("D1:3",),
        )
        assert answer in (
            "between 22 March 2023 and 26 March 2023",
            "between 22 March and 26 March 2023",
        )


class TestDeterministicPolicy:
    async def test_wh_question_skips(self):
        policy = DeterministicReasoningPolicy(min_overlap_ratio=0.5)
        records = [MemoryRecord(key="pref", content="John lives in Beijing.")]
        request = MemoryReasoningInput(query="Where does John live?", recalls=[], records=records)
        result = await policy.answer(request)
        assert result is None

    async def test_empty_tokens(self):
        policy = DeterministicReasoningPolicy(min_overlap_ratio=0.5)
        records = [MemoryRecord(key="pref", content="John lives in Beijing.")]
        request = MemoryReasoningInput(query="???", recalls=[], records=records)
        result = await policy.answer(request)
        assert result is None

    async def test_low_overlap(self):
        policy = DeterministicReasoningPolicy(min_overlap_ratio=0.8)
        records = [MemoryRecord(key="pref", content="Apple fruit details.")]
        request = MemoryReasoningInput(query="Banana and orange", recalls=[], records=records)
        result = await policy.answer(request)
        assert result is None
