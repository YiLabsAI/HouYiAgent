from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from houyi.rag.config import EmbeddingConfig, GraphConfig
from houyi.rag.indexed.runtime_resources import IndexedRuntimeResources
from houyi.rag.indexed.search_backend import (
    ensure_embedder,
    ensure_graph_store,
    ensure_sparse_index,
    ensure_vector_index,
    graph_search,
    sparse_search,
    vector_search,
)


@pytest.mark.asyncio
async def test_embedder_returns_existing() -> None:
    embedder = object()
    resolved = await ensure_embedder(
        embedder=embedder,
        embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
    )
    assert resolved is embedder


@pytest.mark.asyncio
async def test_sparse_creates_index() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        index = await ensure_sparse_index(sparse_index=None, index_dir=tmpdir)
        assert index is not None


@pytest.mark.asyncio
async def test_vector_creates_index() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        index = await ensure_vector_index(
            vector_index=None,
            embedding_config=EmbeddingConfig(provider="openai", model="test", dimension=16),
            index_dir=tmpdir,
        )
        assert index is not None


@pytest.mark.asyncio
async def test_graph_creates_store() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = await ensure_graph_store(
            graph_store=None,
            index_dir=tmpdir,
            graph_config=GraphConfig(enabled=True),
        )
        assert store is not None
        store.close()


@pytest.mark.asyncio
async def test_vector_uses_embedder_index() -> None:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2])
    vector_index = MagicMock()
    vector_index.search = AsyncMock(return_value=["hit"])
    resources = MagicMock(spec=IndexedRuntimeResources)
    resources.get_embedder = AsyncMock(return_value=embedder)
    resources.get_vector_index = AsyncMock(return_value=vector_index)

    results = await vector_search(
        query="python",
        k=5,
        resources=resources,
    )

    assert results == ["hit"]
    resources.get_embedder.assert_awaited_once()
    resources.get_vector_index.assert_awaited_once()
    embedder.embed.assert_awaited_once_with("python")
    vector_index.search.assert_awaited_once_with([0.1, 0.2], 5)


@pytest.mark.asyncio
async def test_sparse_graph_delegate() -> None:
    sparse_index = MagicMock()
    sparse_index.search = AsyncMock(return_value=["sparse"])
    graph_store = MagicMock()
    graph_store.search = AsyncMock(return_value=["graph"])
    sparse_resources = MagicMock(spec=IndexedRuntimeResources)
    sparse_resources.get_sparse_index = AsyncMock(return_value=sparse_index)
    graph_resources = MagicMock(spec=IndexedRuntimeResources)
    graph_resources.get_graph_store = AsyncMock(return_value=graph_store)

    sparse_results = await sparse_search(
        query="python",
        k=3,
        resources=sparse_resources,
    )
    graph_results = await graph_search(
        query="python",
        k=3,
        resources=graph_resources,
    )

    assert sparse_results == ["sparse"]
    assert graph_results == ["graph"]
    sparse_resources.get_sparse_index.assert_awaited_once()
    graph_resources.get_graph_store.assert_awaited_once()
