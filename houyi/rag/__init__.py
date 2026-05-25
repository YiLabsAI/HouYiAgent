"""Top-level public entrypoint for Houyi RAG.

This package exposes the stable public surface used by callers and built-in
RAG skills:

- RAG: unified facade for Agentic and Indexed modes
- search: one-shot convenience helper for quick retrieval
- RAGConfig: top-level configuration model
- RAGMode / RetrievalStrategy: shared RAG enums

The detailed implementation lives in the agentic/, indexed/, generation/,
and retrieval/ subpackages, while this package remains the stable import layer.
"""

from houyi.rag.config import RAGConfig
from houyi.rag.rag import RAG, search
from houyi.rag.types import (
    RAGMode,
    RetrievalStrategy,
)

__all__ = [
    "RAG",
    "RAGConfig",
    "RAGMode",
    "RetrievalStrategy",
    "search",
]
