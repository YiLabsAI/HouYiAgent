from __future__ import annotations

from types import SimpleNamespace

import pytest

from houyi.rag.indexed.models import RetrievalTaskResult
from houyi.rag.indexed.result_processing import (
    adjust_confidence,
    apply_crag,
    build_answer_simple,
    collect_sources,
    generate_answer,
    process_retrieval_results,
)
from houyi.rag.types import RetrievalStrategy, SearchResult, Source


class _Quality:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeValidator:
    def __init__(self, result) -> None:
        self._result = result

    async def validate(self, query: str, results: list[SearchResult]):
        return self._result


class FakeAnswerGenerator:
    def __init__(self, answer: str, confidence: float) -> None:
        self._answer = answer
        self._confidence = confidence

    async def generate(self, *, query: str, results: list[SearchResult], include_sources: bool):
        return self._answer, self._confidence


class FailingAnswerGenerator:
    async def generate(self, *, query: str, results: list[SearchResult], include_sources: bool):
        raise RuntimeError("boom")


def test_process_retrieval_results_success_and_timeout() -> None:
    task_results = [
        RetrievalTaskResult(
            strategy=RetrievalStrategy.VECTOR,
            strategy_name="vector",
            results=[SearchResult(chunk_id="c1", content="a", score=0.9)],
            success=True,
            duration_ms=10.0,
        ),
        RetrievalTaskResult(
            strategy=RetrievalStrategy.BM25,
            strategy_name="bm25",
            success=False,
            timed_out=True,
            error="Timeout after 10s",
            duration_ms=10000.0,
        ),
    ]
    strategies_used: list[RetrievalStrategy] = []
    all_results: list[tuple[str, list[SearchResult]]] = []

    metadata = process_retrieval_results(
        task_results=task_results,
        strategies_used=strategies_used,
        all_results=all_results,
        fallback_on_timeout=True,
    )

    assert metadata["successful_count"] == 1
    assert metadata["failed_count"] == 1
    assert metadata["timed_out_count"] == 1
    assert strategies_used == [RetrievalStrategy.VECTOR]
    assert len(all_results) == 1


def test_process_retrieval_results_clears_results_when_fallback_disabled() -> None:
    task_results = [
        RetrievalTaskResult(
            strategy=RetrievalStrategy.BM25,
            strategy_name="bm25",
            success=False,
            timed_out=True,
            error="Timeout",
            duration_ms=1000.0,
        )
    ]
    strategies_used = [RetrievalStrategy.BM25]
    all_results = [("bm25", [SearchResult(chunk_id="c1", content="x", score=0.1)])]

    metadata = process_retrieval_results(
        task_results=task_results,
        strategies_used=strategies_used,
        all_results=all_results,
        fallback_on_timeout=False,
    )

    assert metadata["timed_out_count"] == 1
    assert strategies_used == []
    assert all_results == []


@pytest.mark.asyncio
async def test_apply_crag_updates_metadata_and_filters_results() -> None:
    results = [SearchResult(chunk_id="c1", content="Python", score=0.9)]
    validator_result = SimpleNamespace(
        quality=_Quality("correct"),
        confidence=0.8,
        reasoning="good",
        relevant_results=results,
    )
    metadata: dict[str, object] = {}

    filtered, quality = await apply_crag(
        validator=FakeValidator(validator_result),
        query="What is Python?",
        fused_results=results,
        enable_crag=True,
        metadata=metadata,
    )

    assert filtered == results
    assert quality == "correct"
    assert metadata["crag_quality"] == "correct"
    assert metadata["crag_confidence"] == 0.8


def test_adjust_confidence_caps_by_crag_and_timeout() -> None:
    assert adjust_confidence(confidence=0.9, crag_quality="incorrect", retrieval_metadata={}) == 0.3
    assert adjust_confidence(confidence=0.9, crag_quality="ambiguous", retrieval_metadata={}) == 0.6
    assert (
        adjust_confidence(
            confidence=0.9, crag_quality=None, retrieval_metadata={"timed_out_count": 1}
        )
        == 0.7
    )


def test_collect_sources_limits_and_filters_empty() -> None:
    results = [
        SearchResult(
            chunk_id=f"c{i}",
            content="x",
            score=0.1,
            source=Source(file_path=f"s{i}"),
        )
        for i in range(12)
    ] + [SearchResult(chunk_id="cx", content="y", score=0.2, source=None)]

    assert [source.file_path for source in collect_sources(results)] == [f"s{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_generate_answer_empty_results() -> None:
    answer, confidence = await generate_answer(answer_generator=None, query="q", results=[])
    assert answer == "No relevant information found."
    assert confidence == 0.0


@pytest.mark.asyncio
async def test_generate_answer_uses_generator_when_available() -> None:
    results = [SearchResult(chunk_id="c1", content="Python", score=0.9)]
    answer, confidence = await generate_answer(
        answer_generator=FakeAnswerGenerator("Generated", 0.8),
        query="q",
        results=results,
    )
    assert answer == "Generated"
    assert confidence == 0.8


@pytest.mark.asyncio
async def test_generate_answer_falls_back_to_simple_builder() -> None:
    results = [SearchResult(chunk_id="c1", content="Python", score=0.9)]
    answer, confidence = await generate_answer(
        answer_generator=FailingAnswerGenerator(),
        query="q",
        results=results,
    )
    assert answer == build_answer_simple(results)
    assert confidence > 0
