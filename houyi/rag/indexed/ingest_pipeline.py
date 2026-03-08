"""Ingest pipeline helpers for indexed RAG mode."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def load_chunks_for_ingest(*, paths: list[str], stats: dict[str, Any]) -> list[Any]:
    from houyi.rag.indexed.document.loaders import load_documents
    from houyi.rag.indexed.document.splitters import split_documents

    documents = await load_documents(paths)
    stats["documents"] = len(documents)
    chunks = await split_documents(documents)
    stats["chunks"] = len(chunks)
    return chunks


async def resolve_embed_contents(
    *,
    chunks: list[Any],
    contextual_retrieval: bool,
    contextualizer: Any,
    stats: dict[str, Any],
) -> list[str]:
    embed_contents = [chunk.content for chunk in chunks]
    if not contextual_retrieval or not contextualizer:
        return embed_contents
    try:
        logger.info("Applying Contextual Retrieval to %d chunks", len(chunks))
        contextualized = await contextualizer.contextualize_chunks(chunks)
        stats["contextualized_chunks"] = len(contextualized)
        logger.debug("Contextualized %d chunks", len(contextualized))
        return [chunk.contextualized_content for chunk in contextualized]
    except Exception as exc:
        logger.warning("Contextual Retrieval failed: %s, using raw content", exc)
        return embed_contents


async def update_retrieval_indexes(
    *,
    chunks: list[Any],
    embed_contents: list[str],
    progress_callback: Any,
    resources: Any,
) -> None:
    embedder = await resources.get_embedder()
    embeddings = await embedder.embed_batch(
        embed_contents,
        progress_callback=progress_callback,
    )

    vector_index = await resources.get_vector_index()
    await vector_index.add_batch(chunks, embeddings)
    await vector_index.save()

    sparse_index = await resources.get_sparse_index()
    await sparse_index.add_batch(chunks)
    await sparse_index.save()


async def maybe_build_graph(
    *,
    chunks: list[Any],
    build_graph: bool,
    graph_enabled: bool,
    stats: dict[str, Any],
    extract_graph_entities,
    resources: Any,
) -> None:
    if not build_graph or not graph_enabled:
        return
    graph_store = await resources.get_graph_store()
    entities, relations = await extract_graph_entities(chunks)
    stats["entities"] = len(entities)
    stats["relations"] = len(relations)
    await graph_store.add_entities(entities)
    await graph_store.add_relations(relations)
    await graph_store.save()


async def extract_graph_entities(
    *, chunks: list[Any], entity_extractor: Any
) -> tuple[list[Any], list[Any]]:
    if entity_extractor:
        try:
            return await entity_extractor.extract_batch(chunks)
        except Exception as exc:
            logger.warning("LLM entity extraction failed: %s, using simple", exc)

    from houyi.rag.indexed.graph.extractor import extract_entities

    return await extract_entities(chunks)
