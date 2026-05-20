"""Loader + schema for the adversarial-memory fixture ().

The fixture lives at tests/fixtures/adversarial_memory.yaml and is
loaded by:

- tests.adapters.memory.test_adversarial_fixture — schema /
 shape regression tests (run on every CI pass).
- The bench harness uses it to drive end-to-end recall + answer
 scoring against real LLMs.

Cases are intentionally small (one query + a handful of seed facts);
the goal is coverage of failure modes, not depth on any one mode.
The expected outcome is encoded structurally so the harness can score
every case without per-case glue code.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from houyi.adapters.memory.types import Certainty


class AdversarialKind(str, Enum):
    """Closed enum of failure modes the fixture exercises."""

    NEGATION_CHECK = "negation_check"
    RETRACTED_FACT = "retracted_fact"
    TEMPORAL_CONFLICT = "temporal_conflict"
    VAGUE_CERTAINTY = "vague_certainty"
    RELATIONAL_MULTI_HOP = "relational_multi_hop"
    EMPTY_MEMORY = "empty_memory"
    FALSE_PREMISE = "false_premise"
    IMPLICIT_EXPECTATION = "implicit_expectation"
    CONTRADICTION = "contradiction"
    AMBIGUOUS_SUBJECT = "ambiguous_subject"
    PARAPHRASE_RECALL = "paraphrase_recall"
    SOURCELESS_DROP = "sourceless_drop"


# Reasons the harness will accept on an abstain. Mirrors the values
# emitted by LLMAnswerer.AnswerResult.reason plus the special
# token "any_abstain" which means "any of the above is fine".
_VALID_ABSTAIN_REASONS: frozenset[str] = frozenset(
    {
        "no_candidates",
        "low_evidence",
        "explicit_absence",
        "contradicting_evidence",
        "low_top_score",
        "too_few_facts",
        "llm_idk",
        "llm_failed",
        "timeout",
        "budget_exceeded",
        "any_abstain",
    }
)


class SeedFact(BaseModel):
    """A pre-seeded AtomicFact-shaped row.

    The bench harness materializes one EntityState row per seed
    before running recall. Empty source_anchor is allowed
    deliberately — the sourceless_drop kind tests that the recall
    layer correctly refuses to surface anchorless facts.
    """

    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    certainty: Certainty = Certainty.CERTAIN
    valid_from: float | None = None
    valid_to: float | None = None
    source_anchor: str = ""


class AdversarialExpectation(BaseModel):
    """What the harness asserts after answering the query."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    """"answer" or "abstain"."""

    reason: str | None = None
    """Required when mode == 'abstain'. One of
 _VALID_ABSTAIN_REASONS or "any_abstain".
 """

    contains: list[str] = Field(default_factory=list)
    """Substrings the answer must contain (case-insensitive). Required
 when mode == 'answer' — an empty list would assert nothing.
 """

    forbid: list[str] = Field(default_factory=list)
    """Substrings the answer must NOT contain. Optional."""

    @field_validator("mode")
    @classmethod
    def _mode_known(cls, v: str) -> str:
        if v not in {"answer", "abstain"}:
            raise ValueError(f"mode must be 'answer' or 'abstain', got {v!r}")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> AdversarialExpectation:
        if self.mode == "abstain":
            if self.reason is None or self.reason not in _VALID_ABSTAIN_REASONS:
                raise ValueError(
                    f"abstain.reason must be one of {sorted(_VALID_ABSTAIN_REASONS)}, "
                    f"got {self.reason!r}"
                )
            if self.contains:
                raise ValueError("abstain cases must not set 'contains'")
        else:  # mode == 'answer'
            if not self.contains and not self.forbid:
                raise ValueError("answer cases must set at least one of 'contains' or 'forbid'")
            if self.reason is not None:
                raise ValueError("answer cases must not set 'reason'")
        return self


class AdversarialCase(BaseModel):
    """One scored fixture row."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: AdversarialKind
    query: str
    seed_facts: list[SeedFact] = Field(default_factory=list)
    expected: AdversarialExpectation


def load_adversarial_fixture(
    path: str | Path | None = None,
) -> list[AdversarialCase]:
    """Parse the YAML fixture into validated AdversarialCase rows.

    When path is None we resolve the canonical fixture relative
    to the repo root, so tests don't need to know the file layout.

    Raises:
    FileNotFoundError: if path does not exist.
    ValueError: on schema violations or duplicate ids — the bench
    harness needs ids to be a primary key for trace joining.
    """
    if path is None:
        path = (
            Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "adversarial_memory.yaml"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"adversarial fixture not found: {path}")

    raw: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "cases" not in raw:
        raise ValueError("fixture missing top-level 'cases' key")

    # Reject unexpected top-level keys
    allowed_keys = {"cases"}
    extra_keys = set(raw.keys()) - allowed_keys
    if extra_keys:
        raise ValueError(f"fixture has unexpected top-level keys: {extra_keys}")

    cases = [AdversarialCase.model_validate(item) for item in raw["cases"]]

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate case id: {case.id}")
        seen.add(case.id)

    return cases


__all__ = [
    "AdversarialCase",
    "AdversarialExpectation",
    "AdversarialKind",
    "SeedFact",
    "load_adversarial_fixture",
]
