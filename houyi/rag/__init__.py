"""Houyi RAG - Future-oriented Agentic RAG Framework.

Dual-mode architecture:
- Agentic Mode: LLM-driven intelligent retrieval (no pre-built index required)
- Indexed Mode: Traditional hybrid retrieval (Vector + Graph + BM25)

Progressive API design:
    # Level 1: Zero-config quickstart
    from houyi.rag import search
    answer = search("What is RAG?", knowledge_dir="./docs")

    # Level 2: Configurable usage
    from houyi.rag import RAGService
    rag = RAGService(mode="agentic")
    answer = rag.query("...")

    # Level 3: Skill integration
    from houyi.rag.skills import kb_search_skill
    agent = Agent(skills=[kb_search_skill])

    # Level 4: Fully customizable
    from houyi.rag import RAGPipeline, HybridRetriever
    pipeline = RAGPipeline(retriever=HybridRetriever(...))
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
    "Chunk",
    # Types
    "Document",
    "Entity",
    "Generator",
    # Internal components (for advanced usage)
    "HybridRetriever",
    "HybridRetrieverConfig",
    "RAGConfig",
    "RAGMode",
    "Relation",
    "Reranker",
    "RetrievalResult",
    "RetrievalStrategy",
    # Protocols
    "Retriever",
    "SearchResult",
    "Source",
    "Validator",
    "create_hybrid_retriever",
    "search",
]
