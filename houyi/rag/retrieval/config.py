"""Configuration models for RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridRetrieverConfig:
    """Configuration for HybridRetriever."""

    fusion_method: str = "rrf"
    rrf_k: int = 60
    vector_weight: float = 0.4
    sparse_weight: float = 0.4
    graph_weight: float = 0.2
