"""Memory adapter exports."""

from houyi.adapters.memory.backends import (
    BACKEND_REGISTRY,
    MemoryBackend,
    SQLiteMemoryBackend,
    create_backend,
)
from houyi.adapters.memory.classifier import MemoryClassifier
from houyi.adapters.memory.deduplicator import MemoryDeduplicator
from houyi.adapters.memory.embedding import (
    EmbeddingProvider,
    NoOpEmbeddingProvider,
    cosine_similarity,
)
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.extractor import MemoryCandidateExtractor
from houyi.adapters.memory.forgetting import apply_forgetting
from houyi.adapters.memory.retriever import MemoryRetriever
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import (
    CandidateStatus,
    DedupMatch,
    ExtractionContext,
    ForgettingPolicy,
    MemoryCandidate,
    MemoryLifecyclePolicy,
    MemoryPolicy,
    MemoryProvenance,
    MemoryRecall,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RecallMatchMethod,
    RelevanceDetail,
    SessionContext,
    TTLPolicy,
)

__all__ = [
    "BACKEND_REGISTRY",
    "CandidateStatus",
    "DedupMatch",
    "EmbeddingProvider",
    "ExtractionContext",
    "ForgettingPolicy",
    "MemoryBackend",
    "MemoryCandidate",
    "MemoryCandidateExtractor",
    "MemoryClassifier",
    "MemoryDeduplicator",
    "MemoryEngine",
    "MemoryLifecyclePolicy",
    "MemoryPolicy",
    "MemoryProvenance",
    "MemoryRecall",
    "MemoryRecord",
    "MemoryRetriever",
    "MemoryScope",
    "MemoryStore",
    "MemoryType",
    "NoOpEmbeddingProvider",
    "RecallMatchMethod",
    "RelevanceDetail",
    "SQLiteMemoryBackend",
    "SessionContext",
    "TTLPolicy",
    "apply_forgetting",
    "cosine_similarity",
    "create_backend",
]
