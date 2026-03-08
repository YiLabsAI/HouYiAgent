"""Contextual Retrieval implementation.

Adds context to document chunks to improve retrieval accuracy.

Reference:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.adapters.llm.base import BaseLLMAdapter

from houyi.rag.types import Chunk, Document

logger = logging.getLogger(__name__)


@dataclass
class ContextualizedChunk:
    """A chunk with added context."""

    chunk: Chunk
    context: str  # Added contextual description
    contextualized_content: str  # Original content + context


class Contextualizer:
    """Adds context to document chunks for better retrieval.

    The Anthropic Contextual Retrieval technique prepends a brief context
    to each chunk that situates it within the overall document.
    """

    SYSTEM_PROMPT = """You are a document contextualization assistant. Given a document chunk and its source document,
generate a brief context (1-2 sentences) that situates this chunk within the broader document.

The context should help a retrieval system understand:
1. What document/section this chunk is from
2. What topic or concept this chunk relates to
3. Any relevant background needed to understand the chunk

Return only the context text, nothing else."""

    def __init__(
        self,
        adapter: BaseLLMAdapter | None = None,
        max_context_length: int = 200,
    ) -> None:
        """Initialize contextualizer.

        Args:
            adapter: LLM adapter for generating context
            max_context_length: Maximum context length in characters
        """
        self._adapter = adapter
        self.max_context_length = max_context_length

    async def contextualize_chunk(
        self,
        chunk: Chunk,
        document: Document | None = None,
        **kwargs: Any,
    ) -> ContextualizedChunk:
        """Add context to a single chunk.

        Args:
            chunk: The chunk to contextualize
            document: Optional source document for additional context
            **kwargs: Additional arguments

        Returns:
            ContextualizedChunk with added context
        """
        if self._adapter:
            context = await self._generate_context_llm(chunk, document)
        else:
            context = self._generate_context_heuristic(chunk, document)

        contextualized_content = f"{context}\n\n{chunk.content}"

        return ContextualizedChunk(
            chunk=chunk,
            context=context,
            contextualized_content=contextualized_content,
        )

    async def contextualize_chunks(
        self,
        chunks: list[Chunk],
        document: Document | None = None,
        batch_size: int = 10,
        **kwargs: Any,
    ) -> list[ContextualizedChunk]:
        """Add context to multiple chunks.

        Args:
            chunks: Chunks to contextualize
            document: Optional source document
            batch_size: Batch size for LLM calls
            **kwargs: Additional arguments

        Returns:
            List of contextualized chunks
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        results = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            for chunk in batch:
                result = await self.contextualize_chunk(chunk, document, **kwargs)
                results.append(result)
        return results

    async def _generate_context_llm(
        self,
        chunk: Chunk,
        document: Document | None = None,
    ) -> str:
        """Generate context using LLM.

        Args:
            chunk: The chunk
            document: Optional source document

        Returns:
            Generated context string
        """
        doc_info = ""
        if document:
            doc_info = f"\nSource document: {document.metadata.get('title', document.source)}"
            if document.metadata.get("summary"):
                doc_info += f"\nDocument summary: {document.metadata['summary'][:500]}"

        user_prompt = f"""Document chunk:{doc_info}

Chunk content:
{chunk.content[:1000]}

Generate a brief context (1-2 sentences) for this chunk:"""

        try:
            response = await self._adapter.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
            )

            context = response.content.strip()
            if len(context) > self.max_context_length:
                context = context[: self.max_context_length - 3] + "..."
            return context

        except Exception as e:
            logger.warning("LLM contextualization failed: %s, using heuristic", e)
            return self._generate_context_heuristic(chunk, document)

    def _generate_context_heuristic(
        self,
        chunk: Chunk,
        document: Document | None = None,
    ) -> str:
        """Generate context using heuristics.

        Args:
            chunk: The chunk
            document: Optional source document

        Returns:
            Generated context string
        """
        parts = []

        file_path = chunk.metadata.get("file_path", "")
        if file_path:
            parts.append(f"From: {file_path}")

        if chunk.metadata:
            if "section" in chunk.metadata:
                parts.append(f"Section: {chunk.metadata['section']}")
            if "page" in chunk.metadata:
                parts.append(f"Page: {chunk.metadata['page']}")

        if document and document.metadata.get("title"):
            parts.insert(0, f"Document: {document.metadata['title']}")

        if parts:
            context = ". ".join(parts) + "."
        else:
            first_line = chunk.content.split("\n")[0].strip()
            if len(first_line) > 100:
                first_line = first_line[:100] + "..."
            context = f"This chunk discusses: {first_line}"

        return context[: self.max_context_length]


async def contextualize_chunks(
    chunks: list[Chunk],
    adapter: BaseLLMAdapter | None = None,
    document: Document | None = None,
) -> list[ContextualizedChunk]:
    """Convenience function to contextualize chunks.

    Args:
        chunks: Chunks to contextualize
        adapter: Optional LLM adapter
        document: Optional source document

    Returns:
        List of contextualized chunks
    """
    contextualizer = Contextualizer(adapter=adapter)
    return await contextualizer.contextualize_chunks(chunks, document)
