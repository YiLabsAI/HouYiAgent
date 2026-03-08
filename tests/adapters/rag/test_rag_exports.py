from houyi.rag import RAG, RAGConfig, RAGMode, RetrievalStrategy, search
from houyi.rag import RAG as LegacyRAG
from houyi.rag import RAGConfig as LegacyRAGConfig
from houyi.rag import RAGMode as LegacyRAGMode
from houyi.rag import RetrievalStrategy as LegacyRetrievalStrategy
from houyi.rag import search as legacy_search


def test_rag_adapter_exports_alias_legacy_symbols() -> None:
    assert RAG is LegacyRAG
    assert RAGConfig is LegacyRAGConfig
    assert RAGMode is LegacyRAGMode
    assert RetrievalStrategy is LegacyRetrievalStrategy
    assert search is legacy_search
