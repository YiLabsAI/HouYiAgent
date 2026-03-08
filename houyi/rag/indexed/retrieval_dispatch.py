from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from houyi.rag.types import RetrievalStrategy


@dataclass(frozen=True)
class RetrievalDispatchContext:
    query: str
    k: int
    graph_enabled: bool
    vector_search: Callable[[str, int], Awaitable[list[Any]]]
    sparse_search: Callable[[str, int], Awaitable[list[Any]]]
    graph_search: Callable[[str, int], Awaitable[list[Any]]]


def create_retrieval_task(
    *,
    strategy: RetrievalStrategy,
    context: RetrievalDispatchContext,
) -> tuple[str, Awaitable[list[Any]]] | None:
    if strategy == RetrievalStrategy.VECTOR:
        return "vector", context.vector_search(context.query, context.k)
    if strategy == RetrievalStrategy.BM25:
        return "bm25", context.sparse_search(context.query, context.k)
    if strategy == RetrievalStrategy.GRAPH and context.graph_enabled:
        return "graph", context.graph_search(context.query, context.k)
    return None
