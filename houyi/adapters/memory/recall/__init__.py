"""Memory recall pipeline.

The package provides a cascading query router, per-type retrievers,
candidate fusion, and unknown-answer guarding on top of the atomic
memory abstractions. It is independent from the legacy
MemoryRetriever in houyi/adapters/memory/retriever.py.

Public re-exports are kept minimal so callers depend on value types
and orchestration surfaces rather than the internal retriever
hierarchy.
"""

from __future__ import annotations

from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator, RecallPipelineConfig
from houyi.adapters.memory.recall.rerank import EvidenceAwareReranker, Reranker
from houyi.adapters.memory.recall.types import (
    QueryType,
    RecallCandidate,
    RecallQuery,
    RecallReason,
    RecallResult,
    RetrieverContext,
    RetrieverKind,
)

__all__ = [
    "EvidenceAwareReranker",
    "QueryType",
    "RecallCandidate",
    "RecallOrchestrator",
    "RecallPipelineConfig",
    "RecallQuery",
    "RecallReason",
    "RecallResult",
    "Reranker",
    "RetrieverContext",
    "RetrieverKind",
]
