"""Document splitters for chunking."""

from __future__ import annotations

import re

from houyi.rag.types import Chunk, Document


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
        if sep_idx >= len(separators) or len(text) <= chunk_size:
            return [text] if text else []

        sep = separators[sep_idx]
        if not sep:
            # Final fallback: split by char count
            return [
                text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)
            ]

        parts = text.split(sep)
        result = []
        current = ""

        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current = current + sep + part if current else part
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size:
                    # Recursively split with next separator
                    result.extend(_split(part, sep_idx + 1))
                else:
                    current = part

        if current:
            result.append(current)

        return result

    text_chunks = _split(doc.content)

    chunks = []
    pos = 0
    for i, text in enumerate(text_chunks):
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_{i}",
            doc_id=doc.doc_id,
            content=text.strip(),
            start_idx=pos,
            end_idx=pos + len(text),
            metadata={
                "chunk_index": i,
                "source": doc.source,
            },
        )
        chunks.append(chunk)
        pos += len(text) - chunk_overlap

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
