"""Bench / eval helpers for the memory adapter.

This package owns the pure-Python loaders and case schemas used by the
adversarial fixture and the LoCoMo / HaluMem runners. It deliberately
has no test framework dependencies so production code may import the
schemas (e.g. for a CLI dump) without pulling pytest in.
"""

from houyi.adapters.memory.bench.adversarial import (
    AdversarialCase,
    AdversarialExpectation,
    AdversarialKind,
    SeedFact,
    load_adversarial_fixture,
)
from houyi.adapters.memory.bench.halumem import (
    HaluMemCase,
    HaluMemTask,
    HaluMemTurn,
    cases_by_task,
    load_halumem,
)
from houyi.adapters.memory.bench.judge import (
    DeterministicJudge,
    JudgeVerdict,
    LLMMemoryJudge,
    MemoryJudge,
)
from houyi.adapters.memory.bench.locomo import (
    DEFAULT_LOCOMO_PATH,
    LoCoMoCase,
    LoCoMoSample,
    LoCoMoTurn,
    load_locomo_all,
    load_locomo_balanced,
)
from houyi.adapters.memory.bench.runner import (
    BenchOutcome,
    BenchReport,
    BenchRunner,
    CaseRunner,
)

__all__ = [
    "DEFAULT_LOCOMO_PATH",
    "AdversarialCase",
    "AdversarialExpectation",
    "AdversarialKind",
    "BenchOutcome",
    "BenchReport",
    "BenchRunner",
    "CaseRunner",
    "DeterministicJudge",
    "HaluMemCase",
    "HaluMemTask",
    "HaluMemTurn",
    "JudgeVerdict",
    "LLMMemoryJudge",
    "LoCoMoCase",
    "LoCoMoSample",
    "LoCoMoTurn",
    "MemoryJudge",
    "SeedFact",
    "cases_by_task",
    "load_adversarial_fixture",
    "load_halumem",
    "load_locomo_all",
    "load_locomo_balanced",
]
