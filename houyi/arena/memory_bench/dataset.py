"""Dataset loaders for the memory benchmark.

Two entry points:

- load_synthetic_fixture returns a small in-process suite that
 exercises all three task types. Used by the smoke test and the CLI's
 --dataset=fixture mode; no network or external deps.
- load_halumem_medium lazy-loads the HaluMem-Medium subset
 published on HuggingFace (IAAR-Shanghai/HaluMem). The HuggingFace
 datasets library is imported on demand so the module remains
 usable in environments where it is not installed.

Both loaders return list[BenchSession] so downstream consumers do
not need to know the source.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from houyi.arena.memory_bench.types import (
    BenchSession,
    DialogueTurn,
    MemoryPoint,
    QAItem,
    UpdatePair,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in synthetic fixture
# ---------------------------------------------------------------------------


_FIXTURE_SESSIONS: tuple[BenchSession, ...] = (
    BenchSession(
        session_id="fixture-extract-001",
        dialogue=(
            DialogueTurn(
                role="user",
                content=(
                    "Hi! My name is Alice and I live in Beijing. I work at a startup called Aurora."
                ),
            ),
        ),
        gold_memories=(
            MemoryPoint(point_id="m1", text="user name is Alice", subject="user"),
            MemoryPoint(point_id="m2", text="user lives in Beijing", subject="user"),
            MemoryPoint(point_id="m3", text="user works at Aurora", subject="user"),
        ),
        qa_items=(
            QAItem(
                qa_id="q1",
                question="Where does the user live?",
                answer="Beijing",
                evidence=("m2",),
            ),
        ),
    ),
    BenchSession(
        session_id="fixture-update-001",
        dialogue=(
            DialogueTurn(role="user", content="I live in Beijing."),
            DialogueTurn(role="user", content="Actually I just moved to Shanghai last week."),
        ),
        gold_memories=(MemoryPoint(point_id="m1", text="user lives in Beijing"),),
        gold_updates=(UpdatePair(update_id="u1", old_point_id="m1", new_text="Shanghai"),),
        qa_items=(
            QAItem(
                qa_id="q1",
                question="Where does the user currently live?",
                answer="Shanghai",
                answer_type="dynamic_update",
                evidence=("m1",),
            ),
        ),
    ),
    BenchSession(
        session_id="fixture-vague-001",
        dialogue=(
            DialogueTurn(role="user", content="The project status is kind of stuck, I'm not sure."),
        ),
        gold_memories=(),  # vague claims must NOT enter the main store
        qa_items=(
            QAItem(
                qa_id="q1",
                question="What is the project's current status?",
                answer="unknown",
                answer_type="memory_boundary",
            ),
        ),
    ),
)


def load_synthetic_fixture() -> list[BenchSession]:
    """Return the built-in fixture sessions.

    These cover three of the eight write-side cells (E2 extract
    accuracy, U1 update supersession, E3 vague filter) and exist so
    smoke tests can exercise the runner without network or LLM cost.
    """
    return list(_FIXTURE_SESSIONS)


# ---------------------------------------------------------------------------
# HaluMem-Medium loader (JSONL on HuggingFace)
# ---------------------------------------------------------------------------


_HF_REPO = "IAAR-Shanghai/HaluMem"
_FILENAME_MEDIUM = "HaluMem-Medium.jsonl"
_FILENAME_LONG = "HaluMem-Long.jsonl"


def load_halumem_medium(
    *,
    sample: int | None = None,
    cache_dir: str | None = None,
    long: bool = False,
) -> list[BenchSession]:
    """Load HaluMem-Medium (or -Long) sessions from HuggingFace.

    The upstream dataset ships as a single JSONL where each line is one
    persona profile containing ~65 nested sessions. We download the
    raw file via huggingface_hub, then flatten it: each nested
    session becomes one BenchSession. Memory points flagged
    is_update=True are emitted as UpdatePair entries with
    original_memories[0] as the old text.

    Parameters
    ----------
    sample:
    If given, stop after this many flattened sessions. Useful for
    keeping evaluation cost bounded during development.
    cache_dir:
    Forwarded to hf_hub_download; defaults to the user's HF
    cache (~/.cache/huggingface).
    long:
    If True, load HaluMem-Long instead of Medium.

    Raises
    ------
    RuntimeError
    If huggingface_hub is not installed; install with the
    [bench] extras (uv pip install -e ".[bench]").
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise RuntimeError(
            "huggingface_hub is required to load HaluMem; install with "
            '`uv pip install -e ".[bench]"`'
        ) from exc

        filename = _FILENAME_LONG if long else _FILENAME_MEDIUM
        local_path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        sessions: list[BenchSession] = []
        with open(local_path, encoding="utf-8") as fh:
            for persona_idx, line in enumerate(fh):
                try:
                    persona = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed persona at line %d", persona_idx)
                    continue
                for session_idx, sess in enumerate(persona.get("sessions") or []):
                    if sample is not None and len(sessions) >= sample:
                        return sessions
                    try:
                        sessions.append(
                            _session_from_halumem(
                                sess,
                                persona=persona,
                                fallback_id=f"halumem-p{persona_idx:03d}-s{session_idx:03d}",
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        logger.warning(
                            "skipping malformed session p%d-s%d", persona_idx, session_idx
                        )
                        return sessions


# ---------------------------------------------------------------------------
# Internals — HaluMem session row to BenchSession
# ---------------------------------------------------------------------------


def _session_from_halumem(
    sess: dict[str, Any],
    *,
    persona: dict[str, Any],
    fallback_id: str,
) -> BenchSession:
    """Translate one HaluMem nested session into BenchSession.

    Memory points map straight to MemoryPoint; points with
    is_update == "True" are additionally projected onto
    UpdatePair so the runner can score the update task. The
    persona's persona_info string is carried in metadata for
    later attribution.
    """
    points_raw = sess.get("memory_points") or []
    gold_memories, gold_updates = _split_memory_points(points_raw)

    dialogue = tuple(_iter_halumem_turns(sess.get("dialogue") or []))
    qa_items = tuple(_iter_halumem_qa(sess.get("questions") or []))

    metadata = {
        "persona_info": str(persona.get("persona_info") or "")[:500],
        "session_start": str(sess.get("start_time") or ""),
        "session_end": str(sess.get("end_time") or ""),
    }
    return BenchSession(
        session_id=str(sess.get("session_id") or fallback_id),
        dialogue=dialogue,
        gold_memories=gold_memories,
        gold_updates=gold_updates,
        qa_items=qa_items,
        metadata=metadata,
    )


def _split_memory_points(
    items: list[dict[str, Any]],
) -> tuple[tuple[MemoryPoint, ...], tuple[UpdatePair, ...]]:
    """Partition memory_points into gold points and gold updates.

    HaluMem encodes updates inline (is_update == "True") with the
    pre-update value in original_memories; this helper materialises
    them into the separate MemoryPoint / UpdatePair streams the
    runner expects.
    """
    points: list[MemoryPoint] = []
    updates: list[UpdatePair] = []
    for raw in items or []:
        text = str(raw.get("memory_content") or "").strip()
        if not text:
            continue
        idx = raw.get("index")
        point_id = f"m{idx}" if idx is not None else f"m{len(points)}"
        importance = raw.get("importance")
        try:
            salience = float(importance) if importance is not None else 1.0
        except (TypeError, ValueError):
            salience = 1.0
            point = MemoryPoint(
                point_id=point_id,
                text=text,
                subject="user",
                memory_type=str(raw.get("memory_type") or "fact"),
                salience=salience,
            )
            points.append(point)

            if str(raw.get("is_update") or "").lower() == "true":
                originals = raw.get("original_memories") or []
                old_text = ""
                if originals and isinstance(originals[0], dict):
                    old_text = str(originals[0].get("memory_content") or "")
                elif originals:
                    old_text = str(originals[0])
                    updates.append(
                        UpdatePair(
                            update_id=f"u{point_id}",
                            old_point_id=point_id,  # self-reference; resolver carries old_text inline
                            new_text=text,
                            new_subject="user",
                        )
                    )
                    if old_text:
                        # Stash the old value as an extra MemoryPoint so the
                        # update scorer can look it up by old_point_id.
                        points.append(
                            MemoryPoint(
                                point_id=f"{point_id}__old",
                                text=old_text,
                                subject="user",
                                memory_type=str(raw.get("memory_type") or "fact"),
                                salience=0.0,
                            )
                        )
                        # Rebind the update pair to point at the synthetic old row.
                        updates[-1] = UpdatePair(
                            update_id=updates[-1].update_id,
                            old_point_id=f"{point_id}__old",
                            new_text=text,
                            new_subject="user",
                        )
                        return tuple(points), tuple(updates)


def _iter_halumem_turns(items: list[dict[str, Any]]):
    for item in items or []:
        role = str(item.get("role") or "user").lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role not in ("user", "assistant", "system"):
            role = "user"
            yield DialogueTurn(role=role, content=content)  # type: ignore[arg-type]


def _iter_halumem_qa(items: list[dict[str, Any]]):
    for idx, item in enumerate(items or []):
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        evidence_raw = item.get("evidence") or []
        # HaluMem evidence is a list of dicts {memory_content, memory_type};
        # downstream only cares about a stable identifier, so we hash the
        # gold memory_content string.
        evidence = tuple(
            str(e.get("memory_content") or "") if isinstance(e, dict) else str(e)
            for e in evidence_raw
        )
        yield QAItem(
            qa_id=f"q{idx}",
            question=question,
            answer=answer,
            answer_type=str(item.get("question_type") or "basic_recall"),
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------


def _session_from_hf_row(row: dict[str, Any], *, fallback_id: str) -> BenchSession:
    """Deprecated translator kept for the existing unit tests.

    The real loader now calls _session_from_halumem with the
    nested HaluMem schema. This shim accepts the older flat-row shape
    used by the placeholder tests so they keep validating the
    defensive get-or-default behaviour the loader still relies on.
    """
    session_id = str(row.get("session_id") or fallback_id)
    dialogue = tuple(_iter_legacy_turns(row.get("dialogue") or row.get("conversation") or []))
    gold_memories = tuple(
        _iter_legacy_memories(row.get("memory_points") or row.get("memories") or [])
    )
    gold_updates = tuple(
        _iter_legacy_updates(row.get("memory_updates") or row.get("updates") or [])
    )
    qa_items = tuple(_iter_legacy_qa(row.get("qa_pairs") or row.get("questions") or []))
    metadata = {
        k: str(v)
        for k, v in (row.get("metadata") or {}).items()
        if isinstance(v, str | int | float)
    }
    return BenchSession(
        session_id=session_id,
        dialogue=dialogue,
        gold_memories=gold_memories,
        gold_updates=gold_updates,
        qa_items=qa_items,
        metadata=metadata,
    )


def _iter_legacy_turns(items: list[dict[str, Any]]):
    for item in items or []:
        role = str(item.get("role") or item.get("speaker") or "user")
        content = str(item.get("content") or item.get("text") or "")
        if role not in ("user", "assistant", "system"):
            role = "user"
        if content:
            yield DialogueTurn(role=role, content=content)  # type: ignore[arg-type]


def _iter_legacy_memories(items: list[dict[str, Any]]):
    for idx, item in enumerate(items or []):
        text = str(item.get("text") or item.get("content") or "").strip()
        if not text:
            continue
        yield MemoryPoint(
            point_id=str(item.get("id") or item.get("point_id") or f"m{idx}"),
            text=text,
            subject=str(item.get("subject") or "user"),
            memory_type=str(item.get("type") or item.get("memory_type") or "fact"),
            salience=float(item.get("salience") or item.get("weight") or 1.0),
        )


def _iter_legacy_updates(items: list[dict[str, Any]]):
    for idx, item in enumerate(items or []):
        new_text = str(item.get("new_text") or item.get("new") or "").strip()
        if not new_text:
            continue
        yield UpdatePair(
            update_id=str(item.get("id") or f"u{idx}"),
            old_point_id=str(item.get("old_point_id") or item.get("old_id") or ""),
            new_text=new_text,
            new_subject=str(item.get("subject") or "user"),
        )


def _iter_legacy_qa(items: list[dict[str, Any]]):
    for idx, item in enumerate(items or []):
        question = str(item.get("question") or item.get("q") or "").strip()
        answer = str(item.get("answer") or item.get("a") or "").strip()
        if not question or not answer:
            continue
        evidence = tuple(str(e) for e in (item.get("evidence") or item.get("evidence_ids") or []))
        yield QAItem(
            qa_id=str(item.get("id") or f"q{idx}"),
            question=question,
            answer=answer,
            answer_type=str(item.get("type") or item.get("answer_type") or "basic_recall"),
            evidence=evidence,
        )
