from __future__ import annotations

import asyncio

import pytest

from houyi.rag.indexed.models import RetrievalTaskResult
from houyi.rag.indexed.retrieval.execution import (
    RetrievalExecutionRequest,
    append_failed_results,
    append_timed_out_results,
    build_retrieval_plan,
    collect_parallel_results,
    execute_parallel_retrieval,
    execute_sequential_retrieval,
)
from houyi.rag.types import RetrievalStrategy


async def _return_values(values: list[str]) -> list[str]:
    return values


def test_build_retrieval_plan_filters_none_tasks() -> None:
    task_payload = object()

    tasks, task_info = build_retrieval_plan(
        strategies=[RetrievalStrategy.BM25, RetrievalStrategy.GRAPH],
        create_retrieval_task=lambda strategy: None
        if strategy == RetrievalStrategy.GRAPH
        else ("bm25", task_payload),
    )

    assert tasks == [task_payload]
    assert task_info == [(RetrievalStrategy.BM25, "bm25")]


@pytest.mark.asyncio
async def test_execute_sequential_retrieval_success_timeout_and_failure() -> None:
    async def slow() -> list[str]:
        await asyncio.sleep(0.01)
        return ["late"]

    mapping = {
        RetrievalStrategy.BM25: ("bm25", _return_values(["ok"])),
        RetrievalStrategy.GRAPH: ("graph", slow()),
    }

    results = await execute_sequential_retrieval(
        RetrievalExecutionRequest(
            strategies=[RetrievalStrategy.BM25, RetrievalStrategy.GRAPH],
            timeout=0.0001,
            create_retrieval_task=lambda strategy: mapping[strategy],
            result_factory=RetrievalTaskResult,
        )
    )

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].timed_out is True


@pytest.mark.asyncio
async def test_execute_parallel_retrieval_no_tasks() -> None:
    results = await execute_parallel_retrieval(
        RetrievalExecutionRequest(
            strategies=[],
            timeout=1.0,
            create_retrieval_task=lambda strategy: None,
            result_factory=RetrievalTaskResult,
        )
    )
    assert results == []


def test_collect_parallel_results_success_and_failure() -> None:
    async def ok() -> list[str]:
        return ["ok"]

    async def bad() -> list[str]:
        raise RuntimeError("boom")

    loop = asyncio.new_event_loop()
    try:
        ok_task = loop.create_task(ok())
        bad_task = loop.create_task(bad())
        loop.run_until_complete(asyncio.gather(ok_task, bad_task, return_exceptions=True))
        results: list[RetrievalTaskResult] = []
        completed = collect_parallel_results(
            done={ok_task, bad_task},
            task_info=[(RetrievalStrategy.BM25, "bm25"), (RetrievalStrategy.GRAPH, "graph")],
            results=results,
            elapsed=5.0,
            result_factory=RetrievalTaskResult,
        )
    finally:
        loop.close()

    assert completed == {0, 1}
    assert len(results) == 2
    assert {result.success for result in results} == {True, False}


def test_append_timeout_and_failed_results() -> None:
    results: list[RetrievalTaskResult] = []
    task_info = [(RetrievalStrategy.BM25, "bm25"), (RetrievalStrategy.GRAPH, "graph")]

    append_timed_out_results(
        results=results,
        task_info=task_info,
        completed_indices={0},
        timeout=2.0,
        result_factory=RetrievalTaskResult,
    )
    append_failed_results(
        results=results,
        task_info=[(RetrievalStrategy.BM25, "bm25")],
        error="boom",
        result_factory=RetrievalTaskResult,
    )

    assert results[0].timed_out is True
    assert results[0].strategy == RetrievalStrategy.GRAPH
    assert results[1].error == "boom"
