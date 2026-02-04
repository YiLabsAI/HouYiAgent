"""Configuration for Houyi RAG."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from houyi.rag.types import RAGMode, RetrievalStrategy


class AgenticConfig(BaseModel):
    """Configuration for Agentic mode."""

    max_rounds: int = Field(default=5, description="Maximum search iterations")
    index_file: str = Field(
        default="data_structure.md",
        description="Directory index file name",
    )
    chunk_limit: int = Field(
        default=500, description="Max lines to read per chunk"
    )


class IndexedConfig(BaseModel):
    """Configuration for Indexed mode."""

    strategies: list[RetrievalStrategy] = Field(
        default_factory=lambda: [RetrievalStrategy.BM25, RetrievalStrategy.VECTOR],
        description="Enabled retrieval strategies",
    )
    top_k: int = Field(default=10, description="Number of results per strategy")
    use_rerank: bool = Field(default=True, description="Whether to use LLM reranking")
    use_crag: bool = Field(
        default=True, description="Whether to use CRAG validation"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding."""

    provider: str = Field(
        default="openai", description="Embedding provider (openai/anthropic/local)"
    )
    model: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )
    dimension: int = Field(default=1536, description="Embedding dimension")
    batch_size: int = Field(default=100, description="Batch size for embedding")


class GraphConfig(BaseModel):
    """Configuration for graph retrieval."""

    enabled: bool = Field(default=False, description="Whether graph retrieval is enabled")
    ppr_alpha: float = Field(
        default=0.85, description="PPR teleport probability"
    )
    ppr_top_k: int = Field(default=50, description="PPR top-k results")
    community_resolution: float = Field(
        default=1.0, description="Louvain resolution parameter"
    )


class RAGConfig(BaseModel):
    """Main RAG configuration."""

    mode: RAGMode = Field(
        default=RAGMode.AUTO, description="RAG operating mode"
    )
    knowledge_dir: str = Field(
        default="knowledge/", description="Knowledge base directory"
    )
    agentic: AgenticConfig = Field(default_factory=AgenticConfig)
    indexed: IndexedConfig = Field(default_factory=IndexedConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    llm_model: str = Field(
        default="", description="LLM model for reranking/generation (empty = default)"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration"
    )

    @classmethod
    def for_agentic(cls, knowledge_dir: str = "knowledge/", **kwargs: Any) -> RAGConfig:
        """Create config optimized for Agentic mode."""
        return cls(
            mode=RAGMode.AGENTIC,
            knowledge_dir=knowledge_dir,
            **kwargs,
        )

    @classmethod
    def for_indexed(
        cls,
        knowledge_dir: str = "knowledge/",
        strategies: list[RetrievalStrategy] | None = None,
        **kwargs: Any,
    ) -> RAGConfig:
        """Create config optimized for Indexed mode."""
        indexed_config = IndexedConfig()
        if strategies:
            indexed_config.strategies = strategies
        return cls(
            mode=RAGMode.INDEXED,
            knowledge_dir=knowledge_dir,
            indexed=indexed_config,
            **kwargs,
        )
