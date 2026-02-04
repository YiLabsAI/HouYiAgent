"""Houyi RAG - 面向未来的 Agentic RAG 框架.

支持双模架构：
- Agentic Mode: LLM 驱动的智能检索（无需预构建索引）
- Indexed Mode: 传统混合检索（Vector + Graph + BM25）

渐进式 API 设计：
    # Level 1: 零配置入门
    from houyi.rag import search
    answer = search("什么是 RAG?", knowledge_dir="./docs")

    # Level 2: 配置化使用
    from houyi.rag import RAGService
    rag = RAGService(mode="agentic")
    answer = rag.query("...")

    # Level 3: Skill 集成
    from houyi.rag.skills import kb_search_skill
    agent = Agent(skills=[kb_search_skill])

    # Level 4: 完全自定义
    from houyi.rag import RAGPipeline, HybridRetriever
    pipeline = RAGPipeline(retriever=HybridRetriever(...))
"""

from houyi.rag.config import RAGConfig
from houyi.rag.service import RAGService, search
from houyi.rag.types import (
    Chunk,
    Document,
    Entity,
    Relation,
    RetrievalResult,
    SearchResult,
    Source,
)

__all__ = [
    # Core Service
    "RAGService",
    "search",
    "RAGConfig",
    # Types
    "Document",
    "Chunk",
    "Entity",
    "Relation",
    "SearchResult",
    "RetrievalResult",
    "Source",
]
