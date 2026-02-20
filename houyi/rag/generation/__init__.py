"""Generation module for RAG.

Provides answer generation with citations, corrective RAG (CRAG), and streaming.
"""

from houyi.rag.generation.citation import CitationGenerator
from houyi.rag.generation.crag import CRAGValidator
from houyi.rag.generation.streaming import (
    StreamEvent,
    StreamEventType,
    StreamingAnswerGenerator,
    stream_sse,
)

__all__ = [
    "CRAGValidator",
    "CitationGenerator",
    "StreamEvent",
    "StreamEventType",
    "StreamingAnswerGenerator",
    "stream_sse",
]
