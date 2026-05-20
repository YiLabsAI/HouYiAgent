"""LoCoMo benchmark loader ().

Parses the official locomo10.json payload from
https://github.com/snap-research/locomo into structured
LoCoMoCase rows the bench harness can drive.

LoCoMo's raw shape is:

- top-level list of conversation samples (typically 10)
- each sample carries:
 - sample_id — short string id, e.g. conv-26
 - conversation — dict with speaker_a / speaker_b + a
 flat sequence of session_N (list of turns) and
 session_N_date_time (RFC-style date string) entries.
 - qa — list of QA dicts with question / answer /
 evidence (list of dia_id strings) / category.

The loader normalizes this into:

- LoCoMoTurn — single dialogue turn (speaker, dia_id,
 text, plus the parent session id + timestamp)
- LoCoMoSample — one full conversation as an ordered list of
 turns
- LoCoMoCase — one QA pair joined back to its parent
 conversation; the bench harness ingests sample.turns first, then
 asks case.question and grades the answer against case.answer.

A balanced subset is produced by load_locomo_balanced which round-
robins across samples to keep distributional balance. Callers needing
the full case sweep can use load_locomo_all instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOCOMO_PATH = Path("/Users/von/workspace/locomo/data/locomo10.json")
"""Where snap-research/locomo lands when cloned alongside this repo.

The bench scripts override this via env var; the default points at the
known clone path so unit tests in dev environments can opt into real
data without hard-coding a fixture path.
"""


@dataclass(frozen=True)
class LoCoMoTurn:
    """One dialogue turn, normalized for ingestion."""

    sample_id: str
    session_id: str
    """Session id within the conversation, e.g. session_3."""

    session_datetime: str
    """Free-form date string from the source JSON; the bench harness
 parses it lazily so a malformed date in the corpus does not break
 loading the rest of the cases.
 """

    speaker: str
    dia_id: str
    """Stable per-turn id used by LoCoMoCase.evidence to point at
 the supporting turns. Loader preserves the source string.
 """

    text: str


@dataclass(frozen=True)
class LoCoMoSample:
    """A whole LoCoMo conversation, plus its flattened turn list."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: tuple[LoCoMoTurn, ...]


@dataclass(frozen=True)
class LoCoMoCase:
    """One QA pair re-joined to its parent sample."""

    sample_id: str
    question: str
    answer: str
    evidence: tuple[str, ...]
    """Source-side dia_id list. The bench harness uses these to score
 "did recall surface the right turns?" in addition to
 "did the answer match?".
 """

    category: int
    """LoCoMo's question category. Surfaced raw so per-category
 aggregates land in the report without a join.
 """

    sample: LoCoMoSample
    """Reference back to the parent sample; lets the harness ingest
 once per sample and answer multiple cases against the same store.
 """


_SESSION_RE = re.compile(r"^session_(\d+)$")


def _parse_sample(raw: dict) -> LoCoMoSample:
    conv = raw.get("conversation") or {}
    speaker_a = str(conv.get("speaker_a", ""))
    speaker_b = str(conv.get("speaker_b", ""))

    # Pair every session_N list with its session_N_date_time
    # sibling; iterate in numeric order so turn ordering inside the
    # parsed sample reflects the source ordering.
    sessions: list[tuple[int, str, str, list]] = []
    for key, value in conv.items():
        match = _SESSION_RE.match(key)
        if match is None or not isinstance(value, list):
            continue
        idx = int(match.group(1))
        dt = str(conv.get(f"{key}_date_time", ""))
        sessions.append((idx, key, dt, value))
    sessions.sort(key=lambda t: t[0])

    sample_id = str(raw.get("sample_id", ""))
    turns: list[LoCoMoTurn] = []
    for _idx, session_key, dt, turn_list in sessions:
        for turn in turn_list:
            if not isinstance(turn, dict):
                continue
            text = turn.get("text")
            dia_id = turn.get("dia_id")
            speaker = turn.get("speaker")
            if not text or not dia_id or not speaker:
                # Skip malformed entries; the bench is graded per
                # answerable QA so a broken turn is not a fatal error.
                continue
            turns.append(
                LoCoMoTurn(
                    sample_id=sample_id,
                    session_id=session_key,
                    session_datetime=dt,
                    speaker=str(speaker),
                    dia_id=str(dia_id),
                    text=str(text),
                )
            )
    return LoCoMoSample(
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        turns=tuple(turns),
    )


def _parse_cases(sample: LoCoMoSample, raw: dict) -> list[LoCoMoCase]:
    qa_list = raw.get("qa") or []
    cases: list[LoCoMoCase] = []
    for qa in qa_list:
        if not isinstance(qa, dict):
            continue
        question = qa.get("question")
        answer = qa.get("answer")
        if not question or answer is None:
            continue
        evidence_raw = qa.get("evidence") or []
        evidence = tuple(str(e) for e in evidence_raw if isinstance(e, str))
        cases.append(
            LoCoMoCase(
                sample_id=sample.sample_id,
                question=str(question),
                answer=str(answer),
                evidence=evidence,
                category=int(qa.get("category", 0) or 0),
                sample=sample,
            )
        )
    return cases


def load_locomo_all(
    path: str | Path | None = None,
) -> list[LoCoMoCase]:
    """Return every QA case in the LoCoMo dataset.

    Args:
    path: optional override for the source JSON. Defaults to
    DEFAULT_LOCOMO_PATH.

    Raises:
    FileNotFoundError: when the source JSON is missing.
    ValueError: when the file is not a JSON list.
    """
    src = Path(path) if path else DEFAULT_LOCOMO_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"LoCoMo dataset not found at {src}; clone snap-research/locomo "
            f"or pass path= explicitly."
        )
    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LoCoMo JSON must be a top-level list of samples")

    all_cases: list[LoCoMoCase] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        sample = _parse_sample(raw)
        all_cases.extend(_parse_cases(sample, raw))
    return all_cases


def load_locomo_balanced(
    path: str | Path | None = None,
    *,
    n: int = 200,
) -> list[LoCoMoCase]:
    """Return a stratified slice of n cases across all samples.

    Cases are picked round-robin across sample_id so per-sample
    quirks don't dominate the report. When fewer than n cases
    exist in total, the entire corpus is returned (no padding).

    The slice is deterministic: source order within each sample is
    preserved, so re-running the bench against the same corpus and
    same n produces an identical case list.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    all_cases = load_locomo_all(path)
    if not all_cases:
        return []

    by_sample: dict[str, list[LoCoMoCase]] = {}
    order: list[str] = []
    for case in all_cases:
        if case.sample_id not in by_sample:
            order.append(case.sample_id)
            by_sample[case.sample_id] = []
        by_sample[case.sample_id].append(case)

    selected: list[LoCoMoCase] = []
    cursors = dict.fromkeys(order, 0)
    while len(selected) < n:
        progressed = False
        for sid in order:
            cur = cursors[sid]
            bucket = by_sample[sid]
            if cur >= len(bucket):
                continue
            selected.append(bucket[cur])
            cursors[sid] = cur + 1
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break  # exhausted all samples before reaching n
    return selected


__all__ = [
    "DEFAULT_LOCOMO_PATH",
    "LoCoMoCase",
    "LoCoMoSample",
    "LoCoMoTurn",
    "load_locomo_all",
    "load_locomo_balanced",
]
