"""Indexed ingest chunk splitting helpers."""

from __future__ import annotations

import re

from houyi.rag.types import Chunk, Document

# Maximum characters per chunk to avoid exceeding embedding model token limits
# Most embedding models have ~8K token limit, ~4 chars/token = ~32K chars
# Using 8000 chars as safe limit (~2000 tokens)
MAX_CHUNK_CHARS = 8000


async def split_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> list[Chunk]:
    """Split documents into chunks.

    Args:
        documents: Documents to split
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        strategy: Splitting strategy (recursive/sentence)

    Returns:
        List of Chunk objects
    """
    chunks: list[Chunk] = []

    for doc in documents:
        if strategy == "recursive":
            doc_chunks = _recursive_split(doc, chunk_size, chunk_overlap)
        elif strategy == "sentence":
            doc_chunks = _sentence_split(doc, chunk_size, chunk_overlap)
        else:
            doc_chunks = _recursive_split(doc, chunk_size, chunk_overlap)

        chunks.extend(doc_chunks)

    return chunks


def _recursive_split(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Recursively split document by separators."""
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, sep_idx: int = 0) -> list[str]:
        if len(text) > MAX_CHUNK_CHARS:
            return _force_split(text, chunk_overlap)

        if sep_idx >= len(separators) or len(text) <= chunk_size:
            return [text] if text else []

        sep = separators[sep_idx]
        if not sep:
            return _fallback_split(text, chunk_size, chunk_overlap)

        parts = text.split(sep)
        return _merge_split_parts(parts, sep, chunk_size, sep_idx, _split)

    text_chunks = _split(doc.content)
    return _build_chunks(doc, text_chunks, chunk_overlap)


def _force_split(text: str, chunk_overlap: int) -> list[str]:
    result = []
    for i in range(0, len(text), MAX_CHUNK_CHARS - chunk_overlap):
        piece = text[i : i + MAX_CHUNK_CHARS]
        if piece:
            result.append(piece)
    return result


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]


def _merge_split_parts(
    parts: list[str],
    sep: str,
    chunk_size: int,
    sep_idx: int,
    split_fn,
) -> list[str]:
    result: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) + len(sep) <= chunk_size:
            current = current + sep + part if current else part
            continue
        if current:
            result.append(current)
        if len(part) > chunk_size:
            result.extend(split_fn(part, sep_idx + 1))
            current = ""
            continue
        current = part
    if current:
        result.append(current)
    return result


def _build_chunks(doc: Document, text_chunks: list[str], chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    pos = 0
    for i, text in enumerate(text_chunks):
        normalized_text = text[:MAX_CHUNK_CHARS] if len(text) > MAX_CHUNK_CHARS else text
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_{i}",
            doc_id=doc.doc_id,
            content=normalized_text.strip(),
            start_idx=pos,
            end_idx=pos + len(normalized_text),
            metadata={
                "chunk_index": i,
                "source": doc.source,
            },
        )
        chunks.append(chunk)
        pos += len(normalized_text) - chunk_overlap
    return chunks


def _sentence_split(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split document by sentences."""
    # Simple sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", doc.content)

    chunks = []
    current_text = ""
    current_start = 0
    pos = 0

    for sentence in sentences:
        if len(current_text) + len(sentence) <= chunk_size:
            current_text += " " + sentence if current_text else sentence
        else:
            if current_text:
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_{len(chunks)}",
                    doc_id=doc.doc_id,
                    content=current_text.strip(),
                    start_idx=current_start,
                    end_idx=pos,
                    metadata={
                        "chunk_index": len(chunks),
                        "source": doc.source,
                    },
                )
                chunks.append(chunk)
                current_start = max(0, pos - chunk_overlap)
            current_text = sentence
        pos += len(sentence) + 1

    if current_text:
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_{len(chunks)}",
            doc_id=doc.doc_id,
            content=current_text.strip(),
            start_idx=current_start,
            end_idx=pos,
            metadata={
                "chunk_index": len(chunks),
                "source": doc.source,
            },
        )
        chunks.append(chunk)

    return chunks
