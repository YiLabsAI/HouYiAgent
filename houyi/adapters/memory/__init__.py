"""Memory adapter exports."""

from houyi.adapters.memory.backends import (
    BACKEND_REGISTRY,
    MemoryBackend,
    SQLiteMemoryBackend,
    create_backend,
)
from houyi.adapters.memory.classifier import MemoryClassifier
from houyi.adapters.memory.deduplicator import MemoryDeduplicator
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.extractor import MemoryCandidateExtractor
from houyi.adapters.memory.factory import (
    build_memory_engine,
    build_memory_engine_from_env,
)
from houyi.adapters.memory.forgetting import apply_forgetting
from houyi.adapters.memory.reasoner import (
    DeterministicReasoningPolicy,
    LLMMemoryReasoningPolicy,
    MemoryReasoner,
    ReasoningPolicy,
)
from houyi.adapters.memory.retriever import MemoryRetriever
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.turn_writer import TurnDetector, TurnWriter, WriteResult
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
    "DeterministicReasoningPolicy",
    "ExtractionContext",
    "ForgettingPolicy",
    "LLMMemoryReasoningPolicy",
    "MemoryBackend",
    "MemoryCandidate",
    "MemoryCandidateExtractor",
    "MemoryClassifier",
    "MemoryDeduplicator",
    "MemoryEngine",
    "MemoryLifecyclePolicy",
    "MemoryPolicy",
    "MemoryProvenance",
    "MemoryReasoner",
    "MemoryRecall",
    "MemoryRecord",
    "MemoryRetriever",
    "MemoryScope",
    "MemoryStore",
    "MemoryType",
    "ReasoningPolicy",
    "RecallMatchMethod",
    "RelevanceDetail",
    "SQLiteMemoryBackend",
    "SessionContext",
    "TTLPolicy",
    "TurnDetector",
    "TurnWriter",
    "WriteResult",
    "apply_forgetting",
    "build_memory_engine",
    "build_memory_engine_from_env",
    "create_backend",
]
