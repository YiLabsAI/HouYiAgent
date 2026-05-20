"""HaluMem-aligned memory benchmark harness.

Public surface:

- MemoryBenchRunner — orchestrates extraction → updating → QA
 evaluation against a multi-turn dialogue dataset.
- MemoryJudge and friends — pluggable correctness oracles.
- load_halumem_medium / load_synthetic_fixture — dataset
 entry points (real HuggingFace + small built-in fixture).

Designed so the smoke test can run without network or LLM judge by
using StubMemoryJudge + the synthetic fixture, while the CLI
binds to the real HaluMem dataset and an LLMMemoryJudge.
"""

from houyi.arena.memory_bench.dataset import (
    load_halumem_medium,
    load_synthetic_fixture,
)
from houyi.arena.memory_bench.judge import (
    JudgeVerdict,
    LLMMemoryJudge,
    MemoryJudge,
    StubMemoryJudge,
)
from houyi.arena.memory_bench.metrics import (
    BenchMetrics,
    ExtractionMetrics,
    QAMetrics,
    UpdateMetrics,
)
from houyi.arena.memory_bench.runner import (
    Answerer,
    MemoryBenchReport,
    MemoryBenchRunner,
    MemoryReader,
    SessionReport,
    SubstringAnswerer,
)
from houyi.arena.memory_bench.tiered_ingestor import TieredBenchIngestor
from houyi.arena.memory_bench.timing import (
    PATH_KIND_SYNC_INLINE,
    PATH_KIND_TIERED_ASYNC,
    BenchTimings,
    SessionTimingSamples,
)
from houyi.arena.memory_bench.types import (
    BenchSession,
    DialogueTurn,
    MemoryPoint,
    QAItem,
    UpdatePair,
)

__all__ = [
    "PATH_KIND_SYNC_INLINE",
    "PATH_KIND_TIERED_ASYNC",
    "Answerer",
    "BenchMetrics",
    "BenchSession",
    "BenchTimings",
    "DialogueTurn",
    "ExtractionMetrics",
    "JudgeVerdict",
    "LLMMemoryJudge",
    "MemoryBenchReport",
    "MemoryBenchRunner",
    "MemoryJudge",
    "MemoryPoint",
    "MemoryReader",
    "QAItem",
    "QAMetrics",
    "SessionReport",
    "SessionTimingSamples",
    "StubMemoryJudge",
    "SubstringAnswerer",
    "TieredBenchIngestor",
    "UpdateMetrics",
    "UpdatePair",
    "load_halumem_medium",
    "load_synthetic_fixture",
]
