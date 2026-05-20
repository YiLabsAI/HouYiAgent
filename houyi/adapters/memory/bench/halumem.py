"""HaluMem loader — hallucination-memory benchmark adapter ().

HaluMem ships three task families:

- memory_integrity — does the system retain the right facts?
- memory_accuracy — does it avoid fabricating?
- qa_accuracy — does the end-to-end QA hold up?

The official corpus is not vendored into this repo (license + size).
This module defines the schema + loader so the bench harness can drop
the official JSON into data/benchmarks/halumem/ and pick it up
without code changes. The loader accepts a flexible payload shape:

- a top-level list of case dicts, **or**
- a top-level dict {"cases": [...]} (mirrors the adversarial
 fixture's convention)

Each case must carry sample_id, task, question, answer
and a conversation list of {speaker, text} turns. evidence
is optional but recommended — the runner uses it to score recall hit
rate independently of the LLM answer.

Once data lands, the bench harness in wires this loader the
same way it wires LoCoMo: ingest the conversation, run recall, ask
LLMAnswerer, score with LLMMemoryJudge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class HaluMemTask(str, Enum):
    """Closed enum of HaluMem task families."""

    MEMORY_INTEGRITY = "memory_integrity"
    MEMORY_ACCURACY = "memory_accuracy"
    QA_ACCURACY = "qa_accuracy"


@dataclass(frozen=True)
class HaluMemTurn:
    """One conversation turn — minimal shape shared by all tasks."""

    speaker: str
    text: str


@dataclass(frozen=True)
class HaluMemCase:
    """One scored HaluMem case."""

    sample_id: str
    task: HaluMemTask
    conversation: tuple[HaluMemTurn, ...]
    question: str
    answer: str
    evidence: tuple[str, ...] = ()
    """Optional turn-level evidence identifiers (e.g. "turn-12").
 HaluMem's official files vary in evidence format; the loader keeps
 the raw strings so the runner can match them however it likes.
 """


def _coerce_task(raw: object) -> HaluMemTask:
    if isinstance(raw, HaluMemTask):
        return raw
    if not isinstance(raw, str):
        raise ValueError(f"task must be a string, got {type(raw).__name__}")
    try:
        return HaluMemTask(raw)
    except ValueError as exc:
        valid = [t.value for t in HaluMemTask]
        raise ValueError(f"unknown HaluMem task {raw!r}; expected one of {valid}") from exc


def _parse_turn(raw: object) -> HaluMemTurn | None:
    if not isinstance(raw, dict):
        return None
    speaker = raw.get("speaker") or raw.get("role")
    text = raw.get("text") or raw.get("content")
    if not speaker or not text:
        return None
    return HaluMemTurn(speaker=str(speaker), text=str(text))


def _parse_case(raw: dict) -> HaluMemCase:
    required = ("sample_id", "task", "question", "answer", "conversation")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"HaluMem case missing required fields: {missing}; "
            f"sample_id={raw.get('sample_id', '?')!r}"
        )

    conv_raw = raw.get("conversation") or []
    if not isinstance(conv_raw, list):
        raise ValueError("conversation must be a list of {speaker, text} dicts")
    turns = tuple(t for t in (_parse_turn(item) for item in conv_raw) if t is not None)

    evidence_raw = raw.get("evidence") or []
    if isinstance(evidence_raw, str):
        evidence_raw = [evidence_raw]
    evidence = tuple(str(e) for e in evidence_raw if e)

    return HaluMemCase(
        sample_id=str(raw["sample_id"]),
        task=_coerce_task(raw["task"]),
        conversation=turns,
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        evidence=evidence,
    )


def load_halumem(path: str | Path) -> list[HaluMemCase]:
    """Load + validate a HaluMem JSON payload.

    Raises:
    FileNotFoundError: if path does not exist.
    ValueError: on schema violations — the loader is strict because
    silently dropping malformed cases would skew the bench
    report without warning.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"HaluMem dataset not found: {src}")

    payload = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "cases" in payload:
        items = payload["cases"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("HaluMem JSON must be a top-level list or a {'cases': [...]} dict")

    if not isinstance(items, list):
        raise ValueError("HaluMem 'cases' must be a list")

    cases: list[HaluMemCase] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        case = _parse_case(item)
        if case.sample_id in seen:
            raise ValueError(f"duplicate HaluMem sample_id: {case.sample_id}")
        seen.add(case.sample_id)
        cases.append(case)
    return cases


def cases_by_task(
    cases: list[HaluMemCase],
) -> dict[HaluMemTask, list[HaluMemCase]]:
    """Group cases by task for tiered reporting."""
    out: dict[HaluMemTask, list[HaluMemCase]] = {t: [] for t in HaluMemTask}
    for case in cases:
        out[case.task].append(case)
    return out


__all__ = [
    "HaluMemCase",
    "HaluMemTask",
    "HaluMemTurn",
    "cases_by_task",
    "load_halumem",
]
