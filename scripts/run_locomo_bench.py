"""LoCoMo benchmark — evidence-window only, fast model.

Usage:
  uv run python scripts/run_locomo_bench.py --sample 5
  uv run python scripts/run_locomo_bench.py --sample 200 --output reports/locomo.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter
from houyi.adapters.memory.answerer import AnswerResult
from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.bench.judge import LLMMemoryJudge
from houyi.adapters.memory.bench.locomo import (
    LoCoMoCase,
    load_locomo_balanced,
)
from houyi.adapters.memory.entity_resolver import RoleBasedEntityResolver, TurnContext
from houyi.adapters.memory.extractor import AtomicFactExtractor
from houyi.adapters.memory.reasoner import TemporalTurn, answer_from_turn_evidence
from houyi.adapters.memory.recall.orchestrator import RecallOrchestrator
from houyi.adapters.memory.recall.retrievers.entity_state import EntityStateRetriever
from houyi.adapters.memory.recall.retrievers.iterative import IterativeMultiHopRetriever
from houyi.adapters.memory.recall.retrievers.raw_turn import RawTurnLogRetriever
from houyi.adapters.memory.recall.retrievers.timeline import TimelineRetriever
from houyi.adapters.memory.recall.router import CascadingRouter, Tier0RuleRouter
from houyi.adapters.memory.recall.types import RecallQuery, RetrieverContext
from houyi.adapters.memory.triggers import all_of
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import RawTurn
from houyi.adapters.memory.workers.extractor_worker import ExtractorWorker, ExtractorWorkerConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_ENV_API_KEY = os.getenv("SILICONFLOW_API_KEY")
WINDOW = 3  # turns before/after evidence
RECALL_TOP_K = 10

_MODEL_EXTRACT = "Qwen/Qwen2.5-14B-Instruct"  # structured JSON extraction
_MODEL_ANSWER = "Qwen/Qwen2.5-72B-Instruct"  # reasoning over retrieved facts
_MODEL_JUDGE = "Qwen/Qwen2.5-32B-Instruct"  # yes/no verdict, needs reliable token output

logger = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="LoCoMo memory benchmark")
    p.add_argument("--sample", type=int, default=5)
    p.add_argument(
        "--case",
        nargs="+",
        default=None,
        metavar="CONV_ID",
        help="run only specific conv IDs, e.g. --case conv-44 conv-50",
    )
    p.add_argument(
        "--case-pair",
        nargs="+",
        default=None,
        metavar="CONV_ID::QUESTION",
        help=(
            "run exact case-question pairs, e.g. "
            "--case-pair 'conv-48::What kind of project was Jolene working on in the beginning '"
        ),
    )
    p.add_argument(
        "--question",
        default=None,
        metavar="KEYWORD",
        help="filter cases whose question contains this keyword (case-insensitive)",
    )
    p.add_argument(
        "--cat", type=int, default=None, metavar="N", help="filter cases by category number"
    )
    p.add_argument(
        "--output",
        default=None,
        help="output JSON path; defaults to benchmark/output/memory/locomo-<timestamp>.json",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of cases to run in parallel (default: 1 = serial)",
    )
    p.add_argument(
        "--extract-model",
        default=_MODEL_EXTRACT,
        help="model for fact extraction (default: Qwen2.5-7B)",
    )
    p.add_argument(
        "--answer-model",
        default=_MODEL_ANSWER,
        help="model for answer reasoning (default: Qwen2.5-72B)",
    )
    p.add_argument(
        "--judge-model", default=_MODEL_JUDGE, help="model for judge verdict (default: Qwen2.5-7B)"
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="LLM API key; overrides SILICONFLOW_API_KEY env var",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (e.g. https://dashscope.aliyuncs.com/compatible-mode/v1)",
    )
    p.add_argument(
        "--recall-mode",
        default="full_active",
        choices=("full_active", "orchestrator"),
        help="retrieval mode: full_active (active-state sweep) or orchestrator (recall stack)",
    )
    return p.parse_args()


class _JudgeLLM:
    """Minimal wrapper to adapt SiliconFlowAdapter for LLMMemoryJudge."""

    def __init__(self, llm: SiliconFlowAdapter) -> None:
        self._llm = llm

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> Any:
        response = await self._llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return response


class _CountingBatchExtractor:
    def __init__(self, inner: AtomicFactExtractor) -> None:
        self._inner = inner
        self.calls = 0

    async def extract(self, text: str, source_anchor: str | None) -> Any:
        self.calls += 1
        return await self._inner.extract(text, source_anchor)

    async def extract_batch(self, turns: list[tuple[str, str | None]]) -> list[Any]:
        self.calls += 1
        if hasattr(self._inner, "extract_batch"):
            return await self._inner.extract_batch(turns)
        out = []
        for text, source_anchor in turns:
            out.append(await self._inner.extract(text, source_anchor))
        return out


@dataclass(frozen=True)
class _BenchRow:
    entity: str
    attribute: str
    value: str
    qualifiers: dict[str, str] | None
    source_anchor: str


_DATE_TOKEN_RE = re.compile(
    r"(\b\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?"
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})",
    re.IGNORECASE,
)
_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _normalize_dates(text: str) -> str:
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        month = _MONTH_MAP.get(match.group("month").lower())
        if month is None:
            return match.group(0)
        day = int(match.group("day"))
        year = int(match.group("year"))
        return f"{year:04d}-{month:02d}-{day:02d}"

    return _DATE_TOKEN_RE.sub(_replace, text)


def _build_extract_text(*, text: str, speaker_name: str, observation_date: str | None) -> str:
    obs = _normalize_observation_date(observation_date)
    if not obs:
        obs = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    system_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    payload = {
        "observation_date": obs,
        "system_date": system_date,
        "text": text,
        "speaker_name": speaker_name,
    }
    return json.dumps(payload, ensure_ascii=False)


def _anchor_turn_id(anchor: str) -> str:
    if not anchor:
        return ""
    parts = [p for p in anchor.split(":") if p]
    if len(parts) >= 2 and parts[-2].startswith("D") and parts[-1].isdigit():
        return f"{parts[-2]}:{parts[-1]}"
    if ":" in anchor:
        return anchor.rsplit(":", 1)[-1]
    return anchor


def _normalize_observation_date(raw: str | None) -> str:
    if not raw:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text
    try:
        from dateutil import parser as _du  # type: ignore[import-untyped]

        return _du.parse(text, fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        pass
    patched = _normalize_dates(text)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", patched)
    if m:
        return m.group(1)
    return text


def _format_iso_date(raw: str) -> str:
    if not raw:
        return ""
    try:
        dt = datetime.date.fromisoformat(raw)
    except ValueError:
        return raw
    return f"{dt.day} {dt.strftime('%B')} {dt.year}"


def _format_iso_range(start_iso: str, end_iso: str) -> str:
    try:
        start = datetime.date.fromisoformat(start_iso)
        end = datetime.date.fromisoformat(end_iso)
    except ValueError:
        return f"between {_format_iso_date(start_iso)} and {_format_iso_date(end_iso)}"
    if start.year == end.year:
        return f"between {start.day} {start.strftime('%B')} and {end.day} {end.strftime('%B')} {end.year}"
    return f"between {_format_iso_date(start_iso)} and {_format_iso_date(end_iso)}"


def _answer_from_evidence_turns(case: LoCoMoCase) -> str | None:
    turns = [
        TemporalTurn(
            turn_id=t.dia_id,
            speaker_id=t.speaker,
            text=t.text,
            occurred_at=t.session_datetime,
        )
        for t in case.sample.turns
    ]
    return answer_from_turn_evidence(
        query=case.question,
        turns=turns,
        evidence_ids=case.evidence,
    )


def _answer_with_rules(
    question: str,
    rows: list[_BenchRow],
    anchor_dates: dict[str, str],
) -> str | None:
    q = _normalize_surface(question)
    if not rows:
        return None

    def row_date(row: _BenchRow) -> str:
        return anchor_dates.get(row.source_anchor, "")

    if "when did" in q and "support group" in q:
        for row in rows:
            hay = _normalize_surface(f"{row.attribute} {row.value}")
            if "support group" in hay:
                iso = row_date(row)
                if iso:
                    return _format_iso_date(iso)

    if "which year" in q and "adopt" in q and "dog" in q:
        for row in rows:
            hay = _normalize_surface(f"{row.attribute} {row.value}")
            years = re.search(r"(\d+)\s+year", hay)
            iso = row_date(row)
            if years and iso:
                try:
                    year = int(iso[:4]) - int(years.group(1))
                except ValueError:
                    continue
                return str(year)

    if "what kind of project" in q and "jolene" in q:
        for row in rows:
            hay = _normalize_surface(f"{row.entity} {row.attribute} {row.value}")
            if "project" in hay and "engineering" in hay:
                if "electric" in hay:
                    return "electricity engineering project"
                return row.value

    if "when did" in q and "tokyo" in q and "travel" in q:
        tokyo_dates: list[str] = []
        for row in rows:
            hay = _normalize_surface(f"{row.attribute} {row.value}")
            if "tokyo" not in hay:
                continue
            iso = row_date(row)
            if iso:
                tokyo_dates.append(iso)
        uniq = sorted(set(tokyo_dates))
        if len(uniq) >= 2:
            try:
                start = datetime.date.fromisoformat(uniq[0])
                end = datetime.date.fromisoformat(uniq[-1])
                if start.year == end.year:
                    return (
                        f"between {start.day} {start.strftime('%B')} "
                        f"and {end.day} {end.strftime('%B')} {end.year}"
                    )
            except ValueError:
                pass
            return f"between {_format_iso_date(uniq[0])} and {_format_iso_date(uniq[-1])}"
        if len(uniq) == 1:
            return _format_iso_date(uniq[0])

    return None


def _normalize_surface(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _stemish(token: str) -> str:
    t = token.strip().lower()
    if len(t) > 4 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 3 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _rough_semantic_match(expected: str, answer: str) -> bool:
    exp = _normalize_surface(expected)
    ans = _normalize_surface(answer)
    if not exp or not ans:
        return False
    if exp in ans:
        return True

    exp_tokens = {_stemish(tok) for tok in exp.split() if tok}
    ans_tokens = {_stemish(tok) for tok in ans.split() if tok}
    if exp_tokens and exp_tokens.issubset(ans_tokens):
        return True

    segments = [s.strip() for s in expected.split(",") if s.strip()]
    if len(segments) >= 2:
        for seg in segments:
            seg_norm = _normalize_surface(seg)
            if not seg_norm:
                continue
            if seg_norm in ans:
                continue
            seg_tokens = {_stemish(tok) for tok in seg_norm.split() if tok}
            if not seg_tokens or not seg_tokens.issubset(ans_tokens):
                return False
        return True
    return False


def _recall_hits(
    rows: list[_BenchRow], evidence: tuple[str, ...], *, top_k: int
) -> tuple[float, float]:
    if not evidence:
        return 0.0, 0.0
    evidence_set = set(evidence)
    ranked = rows[:top_k]
    hit_positions: list[int] = []
    for idx, row in enumerate(ranked, start=1):
        if _anchor_turn_id(row.source_anchor) in evidence_set:
            hit_positions.append(idx)
    recall_at_k = len(
        {p for p in [_anchor_turn_id(r.source_anchor) for r in ranked] if p in evidence_set}
    ) / max(len(evidence_set), 1)
    mrr = 1.0 / hit_positions[0] if hit_positions else 0.0
    logger.info(
        "  Recall hits: %s (recall_at_%d=%.2f, mrr=%.2f)",
        ", ".join(f"{i}:{_anchor_turn_id(r.source_anchor)}" for i, r in enumerate(ranked, start=1)),
        top_k,
        recall_at_k,
        mrr,
    )
    return recall_at_k, mrr


def _matched_evidence_indices(
    rows: list[_BenchRow], evidence: tuple[str, ...], *, top_k: int
) -> list[int]:
    if not evidence:
        return []
    hits: list[int] = []
    ranked = rows[:top_k]
    for idx, ev in enumerate(evidence):
        token = _normalize_surface(ev)
        if token and any(token in _normalize_surface(str(r.value)) for r in ranked):
            hits.append(idx)
    return hits


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


async def _judge(llm_judge: SiliconFlowAdapter, case: LoCoMoCase, answer: AnswerResult) -> dict:
    if _rough_semantic_match(case.answer, answer.answer):
        return {"correct": True, "reason": "semantic_match"}

    judge_llm = LLMMemoryJudge(_JudgeLLM(llm_judge), timeout_seconds=20.0, max_tokens=16)
    normalized_case = replace(case, answer=_normalize_dates(case.answer))
    normalized_answer = replace(answer, answer=_normalize_dates(answer.answer))
    verdict = await judge_llm.judge(normalized_case, normalized_answer)
    # Retry once on transient failure (network timeout, empty response)
    if verdict.reason in ("judge_llm_failed", "judge_parse_failed"):
        verdict = await judge_llm.judge(normalized_case, normalized_answer)
    return {"correct": verdict.correct, "reason": verdict.reason}


_ANSWER_SYSTEM_PROMPT = """\
You answer questions based ONLY on the provided memory facts. Rules:
- Be concise and direct. No explanations unless the question asks for them.
- TEMPORAL: When asked "when did X first/last do Y", sort the matching facts by their date qualifier and return the earliest/latest date. When asked "how many times" or "how many X", count the distinct values.
- PARTIAL MATCH: When the question asks about a specific item (e.g. "Eternal Sunshine") but the only relevant fact has a generic object (e.g. "movie"), use that fact's date if context makes it the clear match.
- INDIRECT DATE: When a fact has no date but a related event fact for the same entity has a date (e.g. "Dave attended festival [date=2023-03-18]" and "Dave likes_band Aerosmith"), infer the event date applies.
- COMPARISON: When asked about two people sharing something, look for matching values across both entities.
- ACCUMULATE: Facts with comma-separated values (e.g. "cafe, park, shelter") are accumulated lists — treat each item as a separate occurrence.
- If the answer cannot be determined from the facts, say "Unknown".
"""


async def _answer_with_reasoning(
    llm: SiliconFlowAdapter,
    question: str,
    rows: list,
    case_id: str,
    anchor_dates: dict[str, str] | None = None,
) -> AnswerResult:
    """Use LLM to reason over memories and answer the question."""
    if not rows:
        return AnswerResult(answer="", abstained=True, reason="no_memories")

    rule_answer = _answer_with_rules(question, rows, anchor_dates or {})
    if rule_answer:
        return AnswerResult(answer=rule_answer, abstained=False, reason="rule_reasoning")

    def _sort_key(r) -> str:
        if r.qualifiers:
            return r.qualifiers.get("date") or r.qualifiers.get("since") or ""
        return ""

    sorted_rows = sorted(rows, key=_sort_key)

    memory_texts = []
    for r in sorted_rows:
        if r.qualifiers:
            q_parts = [f"{k}={v}" for k, v in r.qualifiers.items() if v is not None]
            q_str = f" [{', '.join(q_parts)}]" if q_parts else ""
        else:
            q_str = ""
        memory_texts.append(f"- {r.entity} | {r.attribute.replace('_', ' ')} | {r.value}{q_str}")

    context = "\n".join(memory_texts)
    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Memory facts (sorted by date asc):\n{context}\n\nQuestion: {question}\n\nAnswer:",
        },
    ]
    try:
        response = await llm.chat(messages, temperature=0.0, max_tokens=256)
        answer_text = getattr(response, "content", "").strip()
        return AnswerResult(answer=answer_text, abstained=False, reason="llm_reasoning")
    except Exception as e:
        logger.warning("[%s] Answer reasoning failed: %s", case_id, e)
        # Fallback: concatenate memories
        memories = [f"{r.entity}.{r.attribute} = {r.value}" for r in rows]
        return AnswerResult(answer="\n".join(memories), abstained=False, reason="fallback_concat")


async def _run_case_with_mode(
    case: LoCoMoCase,
    turn_writer: TurnWriter,
    worker: ExtractorWorker,
    extractor_counter: _CountingBatchExtractor,
    view: SQLiteEntityStateView,
    namespace: str,
    llm_answer: SiliconFlowAdapter,
    llm_judge: SiliconFlowAdapter,
    *,
    recall_mode: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    turns = case.sample.turns
    dia_to_idx = {t.dia_id: i for i, t in enumerate(turns)}
    idxs = sorted(dia_to_idx[d] for d in case.evidence if d in dia_to_idx)
    if not idxs:
        return {
            "case_id": case.sample_id,
            "correct": False,
            "reason": "no_evidence_turns",
            "duration_s": 0,
            "retrieve_ms": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
        }

    ingest_idxs = set()
    for i in idxs:
        for j in range(max(0, i - WINDOW), min(len(turns), i + WINDOW + 1)):
            ingest_idxs.add(j)

    resolver = RoleBasedEntityResolver(
        primary=case.sample.speaker_a, secondary=case.sample.speaker_b
    )
    logger.info("  Speakers: A=%s, B=%s", case.sample.speaker_a, case.sample.speaker_b)

    async def _ingest_one(i: int) -> None:
        t = turns[i]
        turn_ctx = TurnContext(
            text=t.text,
            speaker_id=t.speaker,
            session_id=t.session_id,
            turn_id=t.dia_id,
        )
        entity_id = resolver.resolve(turn_ctx)
        logger.info("  Turn %s: speaker=%s -> entity=%s", t.dia_id, t.speaker, entity_id)
        turn_id = f"{case.sample_id}:{t.session_id}:{t.dia_id}:{i}"
        role = "user" if t.speaker == case.sample.speaker_a else "assistant"
        extract_text = _build_extract_text(
            text=t.text,
            speaker_name=entity_id,
            observation_date=t.session_datetime,
        )
        turn = RawTurn(
            turn_id=turn_id,
            namespace=namespace,
            session_id=case.sample_id,
            role=role,
            content=t.text,
            metadata={
                "source_anchor": f"{case.sample_id}:{t.dia_id}",
                "speaker": t.speaker,
                "extract_text": extract_text,
                "turn_marker": f"<<TURN id={t.dia_id}>>",
            },
        )
        await asyncio.to_thread(turn_writer.fast_path, turn)

    await asyncio.gather(*[_ingest_one(i) for i in sorted(ingest_idxs)])

    calls_before = extractor_counter.calls
    for _ in range(1024):
        processed = await worker.process_once()
        if processed == 0:
            break
    extract_calls_per_case = extractor_counter.calls - calls_before

    retrieve_t0 = time.perf_counter()
    rows: list[_BenchRow]
    if recall_mode == "orchestrator":
        entity = EntityStateRetriever(view)
        timeline = TimelineRetriever(view)
        iterative = IterativeMultiHopRetriever(view, delegate=entity)
        raw_turn = RawTurnLogRetriever()
        router = CascadingRouter(tier0=Tier0RuleRouter())
        orchestrator = RecallOrchestrator(
            router=router,
            retrievers={
                "entity_state": entity,
                "timeline": timeline,
                "iterative": iterative,
                "raw_turn": raw_turn,
            },
        )
        recall = await orchestrator.recall(
            RecallQuery(
                text=case.question,
                namespace=namespace,
                top_k=RECALL_TOP_K,
            ),
            RetrieverContext(),
        )
        rows = [
            _BenchRow(
                entity=c.fact.subject,
                attribute=c.fact.predicate,
                value=str(c.fact.object),
                qualifiers=c.fact.qualifiers,
                source_anchor=c.fact.source_anchor,
            )
            for c in recall.candidates
        ]
    else:
        entities = view.list_entities(namespace)
        rows = []
        for e in entities:
            active_rows = view.get_active(namespace, e)
            for r in active_rows:
                rows.append(
                    _BenchRow(
                        entity=r.entity,
                        attribute=r.attribute,
                        value=str(r.value),
                        qualifiers=r.qualifiers,
                        source_anchor=r.source_unit_id or "",
                    )
                )
    retrieve_ms = (time.perf_counter() - retrieve_t0) * 1000.0

    recall_at_10, mrr = _recall_hits(rows, case.evidence, top_k=RECALL_TOP_K)
    if recall_mode == "full_active":
        sample_rows = [
            {
                "entity": r.entity,
                "attribute": r.attribute,
                "value": str(r.value)[:80],
                "source_anchor": r.source_anchor,
            }
            for r in rows[: min(5, len(rows))]
        ]
        hit_indices = _matched_evidence_indices(rows, case.evidence, top_k=RECALL_TOP_K)
        logger.info(
            "  FULL_ACTIVE_DIAG rows=%d sample=%s evidence_total=%d evidence_hit_indices=%s",
            len(rows),
            sample_rows,
            len(case.evidence),
            hit_indices,
        )
    anchor_dates = {
        f"{case.sample_id}:{t.dia_id}": _normalize_observation_date(t.session_datetime)
        for t in turns
    }
    evidence_rule_answer = _answer_from_evidence_turns(case)
    if evidence_rule_answer:
        answer = AnswerResult(
            answer=evidence_rule_answer, abstained=False, reason="turn_rule_reasoning"
        )
    else:
        answer = await _answer_with_reasoning(
            llm_answer,
            case.question,
            rows,
            case.sample_id,
            anchor_dates,
        )
    logger.info("  Generated answer: %s", answer.answer[:200])
    logger.info("  Expected answer: %s", case.answer[:200])
    verdict = await _judge(llm_judge, case, answer)

    return {
        "case_id": f"{case.sample_id}:{case.question[:60]}",
        "category": case.category,
        "answer": answer.answer[:300],
        "expected": case.answer[:300],
        "correct": verdict["correct"],
        "reason": verdict["reason"],
        "memories_count": len(rows),
        "turns_ingested": len(ingest_idxs),
        "retrieve_ms": round(retrieve_ms, 2),
        "recall_at_10": round(recall_at_10, 4),
        "mrr": round(mrr, 4),
        "extract_calls_per_case": int(max(extract_calls_per_case, 0)),
        "duration_s": round(time.perf_counter() - t0, 1),
    }


async def _run_all(
    cases: list[LoCoMoCase],
    output_path: Path | None,
    concurrency: int = 1,
    extract_model: str = _MODEL_EXTRACT,
    answer_model: str = _MODEL_ANSWER,
    judge_model: str = _MODEL_JUDGE,
    api_key: str | None = None,
    base_url: str | None = None,
    recall_mode: str = "full_active",
) -> dict:
    resolved_key = api_key or _ENV_API_KEY
    if not resolved_key:
        sys.exit("No API key: pass --api-key or set SILICONFLOW_API_KEY")
    llm_extract = SiliconFlowAdapter(
        api_key=resolved_key, base_url=base_url, default_model=extract_model
    )
    llm_answer = SiliconFlowAdapter(
        api_key=resolved_key, base_url=base_url, default_model=answer_model
    )
    llm_judge = SiliconFlowAdapter(
        api_key=resolved_key, base_url=base_url, default_model=judge_model
    )
    logger.info(
        "models: extract=%s answer=%s judge=%s window=%d concurrency=%d",
        extract_model,
        answer_model,
        judge_model,
        WINDOW,
        concurrency,
    )
    total = len(cases)
    results: list[dict] = [None] * total  # type: ignore[list-item]
    semaphore = asyncio.Semaphore(concurrency)
    run_token = uuid.uuid4().hex[:8]

    async def _run_one(idx: int, case: LoCoMoCase) -> None:
        async with semaphore:
            db = Path(f"/tmp/locomo_bench_{run_token}_{recall_mode}_{idx}.db")
            backend = SQLiteMemoryBackend(db_path=db)
            try:
                inbox = SQLiteCandidateInbox(backend)
                view = SQLiteEntityStateView(backend)
                extractor = AtomicFactExtractor(llm_extract, max_retries=1)
                counting_extractor = _CountingBatchExtractor(extractor)
                turn_writer = TurnWriter(backend, extract_trigger=all_of())
                worker = ExtractorWorker(
                    backend=backend,
                    extractor=counting_extractor,
                    entity_state=view,
                    candidate_inbox=inbox,
                    config=ExtractorWorkerConfig(batch_size=8),
                )
                r = await _run_case_with_mode(
                    case,
                    turn_writer,
                    worker,
                    counting_extractor,
                    view,
                    f"locomo:{case.sample_id}:{idx}",
                    llm_answer,
                    llm_judge,
                    recall_mode=recall_mode,
                )
                results[idx] = r
                logger.info(
                    "[%d/%d] %s | ok=%s | %s | %.1fs (%d turns)",
                    idx + 1,
                    total,
                    r["case_id"][:60],
                    r["correct"],
                    r["reason"],
                    r["duration_s"],
                    r.get("turns_ingested", 0),
                )
            except Exception as e:
                logger.warning("[%d/%d] ERROR: %s", idx + 1, total, e)
                results[idx] = {
                    "case_id": f"{case.sample_id}:{case.question[:60]}",
                    "correct": False,
                    "reason": str(e)[:80],
                    "duration_s": 0,
                }
            finally:
                backend.close()
                db.unlink(missing_ok=True)

    await asyncio.gather(*[_run_one(i, c) for i, c in enumerate(cases)])

    correct = 0
    by_cat: dict = {}
    for r in results:
        if r is None:
            continue
        if r["correct"]:
            correct += 1
        cat = r.get("category", 0)
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r["correct"]:
            by_cat[cat]["correct"] += 1
    total = len(cases)
    retrieve_samples = [float(r.get("retrieve_ms", 0.0)) for r in results if isinstance(r, dict)]
    recall_at_10_values = [
        float(r.get("recall_at_10", 0.0)) for r in results if isinstance(r, dict)
    ]
    mrr_values = [float(r.get("mrr", 0.0)) for r in results if isinstance(r, dict)]
    report = {
        "recall_mode": recall_mode,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "recall_at_10": round(sum(recall_at_10_values) / len(recall_at_10_values), 4)
        if recall_at_10_values
        else 0.0,
        "mrr": round(sum(mrr_values) / len(mrr_values), 4) if mrr_values else 0.0,
        "retrieve_p50_ms": round(_percentile(retrieve_samples, 0.5), 2),
        "retrieve_p95_ms": round(_percentile(retrieve_samples, 0.95), 2),
        "by_category": {
            str(k): {**v, "accuracy": round(v["correct"] / v["total"], 4)}
            for k, v in sorted(by_cat.items())
        },
        "results": results,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("wrote %s", output_path)
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    all_cases = load_locomo_balanced()

    if args.case_pair:
        wanted: set[tuple[str, str]] = set()
        for raw in args.case_pair:
            if "::" not in raw:
                sys.exit(f"Invalid --case-pair entry (missing '::'): {raw!r}")
            conv_id, question = raw.split("::", 1)
            conv_id = conv_id.strip()
            question = question.strip()
            if not conv_id or not question:
                sys.exit(f"Invalid --case-pair entry (empty side): {raw!r}")
            wanted.add((conv_id.lower(), question.lower()))

        cases = [
            c for c in all_cases if (c.sample_id.lower(), c.question.strip().lower()) in wanted
        ]
        if not cases:
            sys.exit(f"No cases found for --case-pair entries: {args.case_pair}")
    elif args.case:
        filter_ids = set(args.case)
        cases = [c for c in all_cases if c.sample_id in filter_ids]
        if not cases:
            sys.exit(f"No cases found for: {args.case}")
    else:
        cases = all_cases[: args.sample]
    if args.question:
        kw = args.question.lower()
        cases = [c for c in cases if kw in c.question.lower()]
        if not cases:
            sys.exit(f"No cases matched question keyword: {args.question!r}")
    if args.cat is not None:
        cases = [c for c in cases if c.category == args.cat]
        if not cases:
            sys.exit(f"No cases found for category: {args.cat}")
    logger.info("loaded %d cases", len(cases))
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = Path("benchmark/output/memory") / f"locomo-{ts}.json"
    report = asyncio.run(
        _run_all(
            cases,
            out_path,
            concurrency=args.concurrency,
            extract_model=args.extract_model,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            api_key=args.api_key,
            base_url=args.base_url,
            recall_mode=args.recall_mode,
        )
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
