"""Indexed mode for Houyi RAG.

Indexed mode provides traditional RAG with pre-built indexes:
- Vector index (hnswlib)
- Sparse index (BM25)
- Graph retrieval (PPR)
"""

from houyi.rag.indexed.mode import IndexedMode

__all__ = ["IndexedMode"]
