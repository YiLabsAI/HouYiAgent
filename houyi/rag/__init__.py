"""Houyi RAG - Agentic RAG framework.

Supports dual-mode architecture:
- Agentic Mode: LLM-driven intelligent retrieval (no pre-built indexes)
- Indexed Mode: Traditional hybrid retrieval (Vector + Graph + BM25)

Unified API:
    from houyi.rag import RAG

    # Zero config (Agentic mode)
    rag = RAG("./docs")
    result = await rag.query("What is RAG?")

    # With indexing (Indexed mode)
    rag = RAG("./docs", mode="indexed")
    await rag.index()
    result = await rag.query("What is RAG?")

    # With LLM
    rag = RAG("./docs", mode="indexed", llm="openai:gpt-4o-mini")

For Skill integration:
    from houyi.rag.skills import kb_search_skill
    agent = Agent(skills=[kb_search_skill])
"""

from houyi.rag.config import RAGConfig
from houyi.rag.rag import RAG, search
from houyi.rag.retrieval import (
    Generator,
    HybridRetriever,
    HybridRetrieverConfig,
    Reranker,
    Retriever,
    Validator,
    create_hybrid_retriever,
)
from houyi.rag.types import (
    Chunk,
    Document,
    Entity,
    RAGMode,
    Relation,
    RetrievalResult,
    RetrievalStrategy,
    SearchResult,
    Source,
)

__all__ = [
    # Primary API
    "RAG",
    "search",
    "RAGConfig",
    "RAGMode",
    "RetrievalStrategy",
    # Internal components (for advanced usage)
    "HybridRetriever",
    "HybridRetrieverConfig",
    "create_hybrid_retriever",
    # Protocols
    "Retriever",
    "Reranker",
    "Generator",
    "Validator",
    # Types
    "Document",
    "Chunk",
    "Entity",
    "Relation",
    "SearchResult",
    "RetrievalResult",
    "Source",
]
