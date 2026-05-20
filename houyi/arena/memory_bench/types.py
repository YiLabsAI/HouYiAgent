"""Data types for the HaluMem-aligned memory benchmark.

The shapes here mirror the operation-level surface defined by the
HaluMem paper (extraction → updating → QA) without coupling to a
specific dataset format. Concrete loaders translate dataset-native
records into these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """One (role, content) pair from a multi-turn session.

    The benchmark feeds these turns to the system under test in
    chronological order; each user turn becomes the input to one
    MemoryIngestor.ingest_turn call.
    """

    role: Literal["user", "assistant", "system"]
    content: str


@dataclass(frozen=True, slots=True)
class MemoryPoint:
    """A gold memory point that the SUT is expected to extract.

    Aligns with HaluMem's reference memories: a textual claim about a
    subject, optionally typed (preference / fact / event / ...). The
    salience field carries the dataset's weighted-importance score
    when present, used for Weighted Memory Recall; defaults to 1.0
    when the dataset does not annotate it.
    """

    point_id: str
    text: str
    subject: str = "user"
    memory_type: str = "fact"
    salience: float = 1.0


@dataclass(frozen=True, slots=True)
class UpdatePair:
    """A gold (old → new) memory replacement.

    Used by the updating task to score whether the SUT correctly
    superseded the prior value. old_point_id references a previously
    extracted memory; new_text is the post-update value.
    """

    update_id: str
    old_point_id: str
    new_text: str
    new_subject: str = "user"


@dataclass(frozen=True, slots=True)
class QAItem:
    """One question-answer pair with optional supporting evidence.

    answer is the reference answer; answer_type is the HaluMem
    QA category (basic_recall / multi_hop / dynamic_update /
    memory_boundary / memory_conflict / generalization). evidence
    lists the MemoryPoint.point_id values that should be sufficient
    to answer correctly — handy for upstream-error attribution.
    """

    qa_id: str
    question: str
    answer: str
    answer_type: str = "basic_recall"
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchSession:
    """One evaluation session: dialogue + gold extraction/update/QA.

    A session is the atomic unit of benchmark execution: ingest every
    user turn, then score the resulting memory state against the three
    gold sets. Scoring is deferred to MemoryBenchRunner so the
    session itself stays serializable / cacheable.
    """

    session_id: str
    dialogue: tuple[DialogueTurn, ...]
    gold_memories: tuple[MemoryPoint, ...] = ()
    gold_updates: tuple[UpdatePair, ...] = ()
    qa_items: tuple[QAItem, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
