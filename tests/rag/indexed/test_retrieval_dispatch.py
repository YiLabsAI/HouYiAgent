from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from houyi.rag.indexed.retrieval.dispatch import (
    RetrievalDispatchContext,
    create_retrieval_task,
)
from houyi.rag.types import RetrievalStrategy


def _build_context(*, graph_enabled: bool = True) -> RetrievalDispatchContext:
    return RetrievalDispatchContext(
        query="python",
        k=3,
        graph_enabled=graph_enabled,
        vector_search=AsyncMock(return_value=["vector"]),
        sparse_search=AsyncMock(return_value=["bm25"]),
        graph_search=AsyncMock(return_value=["graph"]),
    )


@pytest.mark.asyncio
async def test_create_retrieval_task_for_vector_strategy() -> None:
    context = _build_context()

    name, task = create_retrieval_task(
        strategy=RetrievalStrategy.VECTOR,
        context=context,
    )

    assert name == "vector"
    assert await task == ["vector"]


@pytest.mark.asyncio
async def test_create_retrieval_task_for_bm25_strategy() -> None:
    context = _build_context()

    name, task = create_retrieval_task(
        strategy=RetrievalStrategy.BM25,
        context=context,
    )

    assert name == "bm25"
    assert await task == ["bm25"]


def test_create_retrieval_task_skips_graph_when_disabled() -> None:
    context = _build_context(graph_enabled=False)

    task = create_retrieval_task(
        strategy=RetrievalStrategy.GRAPH,
        context=context,
    )

    assert task is None


@pytest.mark.asyncio
async def test_create_retrieval_task_for_graph_strategy_when_enabled() -> None:
    context = _build_context(graph_enabled=True)

    name, task = create_retrieval_task(
        strategy=RetrievalStrategy.GRAPH,
        context=context,
    )

    assert name == "graph"
    assert await task == ["graph"]
