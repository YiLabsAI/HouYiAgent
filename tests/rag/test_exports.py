"""Export-surface tests for houyi.rag."""

from __future__ import annotations

from houyi.rag import RAG, RAGConfig, search


class TestRAGExports:
    def test_public_exports_exist(self) -> None:
        assert RAG is not None
        assert RAGConfig is not None
        assert search is not None
