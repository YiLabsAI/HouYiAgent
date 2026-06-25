"""LoCoMo benchmark — evidence-window only, fast model.

Usage:
  uv run python scripts/run_locomo_bench.py --sample 5
  uv run python scripts/run_locomo_bench.py --sample 200 --output reports/locomo.json
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import copy
import datetime
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from houyi.adapters.embedding import (
    DashScopeEmbeddingProvider,
    EmbeddingProvider,
    SiliconFlowEmbeddingProvider,
    make_embedding_provider,
)
from houyi.adapters.llm.factory import LLMAdapterFactory
from houyi.adapters.memory.answerer import AnswerResult
from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.bench.judge import LLMMemoryJudge
from houyi.adapters.memory.bench.locomo import (
    LoCoMoCase,
    load_locomo_balanced,
)
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.entity_resolver import RoleBasedEntityResolver, TurnContext
from houyi.adapters.memory.extractor import (
    _ATOMIC_FACT_BATCH_SYSTEM_PROMPT,
    AtomicFactExtractor,
)
from houyi.adapters.memory.recall.factory import _build_default_recall_orchestrator
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.triggers import all_of
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import RawTurn
from houyi.adapters.memory.workers.embedding_backfill import (
    EmbeddingBackfillConfig,
    EmbeddingBackfillWorker,
)
from houyi.adapters.memory.workers.extractor_worker import ExtractorWorker, ExtractorWorkerConfig
from houyi.infrastructure.config.env_config import EnvConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_ENV_API_KEY = os.getenv("SILICONFLOW_API_KEY")
WINDOW = 3  # turns before/after evidence
RECALL_TOP_K = 10
# Extraction throughput knobs. Larger batches cut LLM round-trips (the
# dominant latency). Drain workers parallelize the LLM extraction across
# batches; a shared asyncio write-lock (passed to ExtractorWorker)
# serializes only the SQLite write phases, so higher drain concurrency no
# longer triggers "database is locked". Big conversations (15+ extract
# batches) benefit most from running several batches in flight.
EXTRACT_BATCH_SIZE = 16
DRAIN_WORKERS = 4
_ABLATE_RETRIEVERS: set[str] = set()
"""Retriever registry keys to strip from the route table for ablation runs.

Populated from --ablate-retrievers in main(); empty by default so normal
runs use the unmodified default route table.
"""

_FUSION_STRATEGY = "rrf"
"""Cross-source fusion strategy: "weighted" (per-kind min-max) or "rrf".

Populated from --fusion-strategy in main(); defaults to "rrf" (rank-based
ReciprocalRankFuser, robust to incomparable retriever score scales).
"""

# SiliconFlow default model set (provider=siliconflow).
_SF_MODEL_EXTRACT = "Qwen/Qwen2.5-14B-Instruct"  # structured JSON extraction
_SF_MODEL_ANSWER = "Qwen/Qwen2.5-72B-Instruct"  # reasoning over retrieved facts
_SF_MODEL_JUDGE = "Qwen/Qwen2.5-32B-Instruct"  # yes/no verdict, needs reliable token output
_SF_MODEL_EVOLVE = "Qwen/Qwen2.5-72B-Instruct"  # reflection over sampled memories

# Bailian/DashScope default model set (provider=dashscope). Kept independent of
# the SiliconFlow set so picking a provider never silently inherits the other's
# model identifiers.
_DS_MODEL_EXTRACT = "glm-5.1"  # structured JSON extraction
_DS_MODEL_ANSWER = "qwen3.7-max"  # reasoning over retrieved facts
_DS_MODEL_JUDGE = "qwen3.7-max"  # yes/no verdict
_DS_MODEL_EVOLVE = "glm-5.2"  # reflection over sampled memories during evolution

# Per-provider default model sets, resolved after CLI parsing when the
# corresponding --extract/answer/judge/evolve-model flag is left unset.
_PROVIDER_MODEL_DEFAULTS: dict[str, dict[str, str]] = {
    "siliconflow": {
        "extract": _SF_MODEL_EXTRACT,
        "answer": _SF_MODEL_ANSWER,
        "judge": _SF_MODEL_JUDGE,
        "evolve": _SF_MODEL_EVOLVE,
    },
    "dashscope": {
        "extract": _DS_MODEL_EXTRACT,
        "answer": _DS_MODEL_ANSWER,
        "judge": _DS_MODEL_JUDGE,
        "evolve": _DS_MODEL_EVOLVE,
    },
}

# Backwards-compatible aliases used as fallbacks in helper signatures.
_MODEL_EXTRACT = _SF_MODEL_EXTRACT
_MODEL_ANSWER = _SF_MODEL_ANSWER
_MODEL_JUDGE = _SF_MODEL_JUDGE

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
        action="append",
        default=None,
        metavar="CONV_ID::QUESTION",
        help=(
            "run exact case-question pairs, e.g. "
            "--case-pair 'conv-48::What kind of project was Jolene working on in the beginning '"
        ),
    )
    p.add_argument(
        "--ablate-retrievers",
        nargs="+",
        default=None,
        metavar="RETRIEVER",
        help=(
            "strip these retriever registry keys from the route table for "
            "ablation, e.g. --ablate-retrievers graph"
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
        "--llm-provider",
        default="siliconflow",
        choices=tuple(_PROVIDER_MODEL_DEFAULTS),
        help=(
            "LLM backend for extract/answer/judge: siliconflow (default) or "
            "dashscope (Bailian). Selecting a provider also picks that "
            "provider's default model set unless --extract/answer/judge-model "
            "are given. dashscope resolves DASHSCOPE_API_KEY/BASE_URL from env."
        ),
    )
    p.add_argument(
        "--extract-model",
        default=None,
        help="model for fact extraction (default: provider-specific)",
    )
    p.add_argument(
        "--answer-model",
        default=None,
        help="model for answer reasoning (default: provider-specific)",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="model for judge verdict (default: provider-specific)",
    )
    p.add_argument(
        "--evolve-model",
        default=None,
        help=(
            "model for evolution reflection in --evolve-mode "
            "(default: provider-specific; dashscope -> glm-5.2)"
        ),
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
        "--embedding-provider",
        default="siliconflow",
        choices=("siliconflow", "dashscope", "local", "noop"),
        help="embedding provider: siliconflow (default), dashscope (bailian), local, or noop",
    )
    p.add_argument(
        "--embedding-model",
        default=None,
        help="embedding model name; defaults to EMBEDDING_MODEL env var or provider default",
    )
    p.add_argument(
        "--embedding-api-key",
        default=None,
        help="API key for embedding provider; defaults to provider-specific env var (e.g., SILICONFLOW_API_KEY or DASHSCOPE_API_KEY)",
    )
    p.add_argument(
        "--use-window",
        action="store_true",
        default=False,
        help="Enable evidence-window turn restriction to only ingest turns around the target evidence, accelerating cold extraction.",
    )
    p.add_argument(
        "--evolve-mode",
        action="store_true",
        default=False,
        help=(
            "Run a self-evolution pass per case: answer (baseline), evolve memories "
            "via the causal recall-replay evaluator, re-answer, and emit a before/after "
            "accuracy + recall@10 report. Memories are derived only from stored records."
        ),
    )
    p.add_argument(
        "--no-consolidate",
        action="store_true",
        default=False,
        help=(
            "Skip the deterministic entity-state supersede pass that normally runs "
            "before reflection in --evolve-mode. Ablation switch to isolate the "
            "consolidation pass's recall contribution from reflection's."
        ),
    )
    p.add_argument(
        "--no-reflect",
        action="store_true",
        default=False,
        help=(
            "Skip the failure-anchored reflector that normally runs after "
            "consolidation in --evolve-mode. Ablation switch to isolate "
            "reflection's contribution from consolidation's."
        ),
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Size of the evidence window (context range before and after each evidence turn) when use-window is enabled. (default: 10)",
    )
    p.add_argument(
        "--fusion-strategy",
        default="rrf",
        choices=("weighted", "rrf"),
        help=(
            "cross-source fusion strategy: rrf (rank-based ReciprocalRankFuser, "
            "default) or weighted (per-kind min-max)"
        ),
    )
    return p.parse_args()


class _JudgeLLM:
    """Minimal wrapper to adapt an LLM adapter for LLMMemoryJudge."""

    def __init__(self, llm: Any) -> None:
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


# Per-turn extraction cache shared across all cases in a single bench run.
# With --use-window, the DB cache is fragmented by evidence_hash, so the
# same conversation is re-extracted once per question. The bench log shows
# conv-44/43/41 each fully extracted twice. Keying extraction results by
# (anchor, text) lets the second question reuse the first's facts and skip
# the LLM call entirely. Results are deep-copied on hit because downstream
# promotion writes them into each case's own backend.
#
# PERSISTENT: loaded at startup and saved on exit (atexit), so clearing the
# session-DB cache to re-project (e.g. after a namespace/config change) does
# NOT re-call the LLM — it replays the cached ExtractionResult and only
# re-projects into the DB. This turns a ~50min re-extract into ~10min reproject.
_TURN_EXTRACT_CACHE_FILE = "/tmp/locomo_turn_extract_cache.pkl"


def _load_turn_cache() -> dict[str, Any]:
    try:
        with open(_TURN_EXTRACT_CACHE_FILE, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            print(f"[TurnCache] Loaded {len(loaded)} cached extraction results")
            return loaded
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[TurnCache] load failed ({exc}); starting empty")
    return {}


def _save_turn_cache() -> None:
    try:
        with open(_TURN_EXTRACT_CACHE_FILE, "wb") as f:
            pickle.dump(_TURN_EXTRACT_CACHE, f)
        print(f"[TurnCache] Saved {len(_TURN_EXTRACT_CACHE)} extraction results")
    except Exception as exc:
        print(f"[TurnCache] save failed ({exc})")


_TURN_EXTRACT_CACHE: dict[str, Any] = _load_turn_cache()
atexit.register(_save_turn_cache)


def _turn_cache_key(text: str, source_anchor: str | None) -> str:
    return hashlib.md5(f"{source_anchor or ''}\x00{text}".encode()).hexdigest()


class _CountingBatchExtractor:
    def __init__(self, inner: AtomicFactExtractor) -> None:
        self._inner = inner
        self.calls = 0

    async def extract(self, text: str, source_anchor: str | None) -> Any:
        key = _turn_cache_key(text, source_anchor)
        cached = _TURN_EXTRACT_CACHE.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
        self.calls += 1
        result = await self._inner.extract(text, source_anchor)
        _TURN_EXTRACT_CACHE[key] = result
        return result

    async def extract_batch(
        self, turns: list[tuple[str, str | None]], namespace: str = "default"
    ) -> list[Any]:
        results: list[Any] = [None] * len(turns)
        uncached: list[tuple[int, tuple[str, str | None]]] = []
        for i, (text, anchor) in enumerate(turns):
            hit = _TURN_EXTRACT_CACHE.get(_turn_cache_key(text, anchor))
            if hit is not None:
                results[i] = copy.deepcopy(hit)
            else:
                uncached.append((i, (text, anchor)))

        if uncached:
            self.calls += 1
            sub = [turn for _, turn in uncached]
            if hasattr(self._inner, "extract_batch"):
                out = await self._inner.extract_batch(sub, namespace=namespace)
            else:
                out = [await self._inner.extract(text, anchor) for text, anchor in sub]
            for (i, (text, anchor)), res in zip(uncached, out, strict=True):
                _TURN_EXTRACT_CACHE[_turn_cache_key(text, anchor)] = res
                results[i] = res
        return results


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


def _recall_metrics(
    rows: list[_BenchRow], evidence: tuple[str, ...], *, top_k: int
) -> tuple[float, float, float]:
    """Compute recall@k, nDCG@k, precision@k (6 golden metrics).

    Relevance is binary: a candidate is relevant if its source_anchor matches
    a gold evidence anchor. nDCG rewards relevant items at higher ranks
    (position-discounted); Precision measures noise (non-relevant slots in
    top-k). Both use first-occurrence only (duplicate anchors do not double-
    count).
    """
    import math

    if not evidence:
        return 0.0, 0.0, 0.0
    evidence_set = set(evidence)
    ranked = rows[:top_k]
    seen: set[str] = set()
    first_hits: list[int] = []
    for idx, row in enumerate(ranked, start=1):
        anchor = _anchor_turn_id(row.source_anchor)
        if anchor in evidence_set and anchor not in seen:
            seen.add(anchor)
            first_hits.append(idx)
    recall_at_k = len(seen) / max(len(evidence_set), 1)
    dcg = sum(1.0 / math.log2(pos + 1) for pos in first_hits)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(evidence_set), top_k) + 1))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
    precision_at_k = len(seen) / max(len(ranked), 1)
    logger.info(
        "  Recall hits: %s (R@%d=%.2f, nDCG@%d=%.2f, P@%d=%.2f)",
        ", ".join(f"{i}:{_anchor_turn_id(r.source_anchor)}" for i, r in enumerate(ranked, start=1)),
        top_k,
        recall_at_k,
        top_k,
        ndcg_at_k,
        top_k,
        precision_at_k,
    )
    return recall_at_k, ndcg_at_k, precision_at_k


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


async def _judge(llm_judge: Any, case: LoCoMoCase, answer: AnswerResult) -> dict:
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


def _recalls_to_rows(engine: MemoryEngine, recalls: list[Any]) -> list[_BenchRow]:
    """Map engine recalls to the bench's evidence-anchored rows."""
    rows: list[_BenchRow] = []
    record_index = engine._build_record_index()
    for r in recalls:
        record = record_index.get(r.memory_id)
        if not record:
            continue
        parts = record.key.split(".", 1)
        subj = parts[0] if len(parts) > 1 else ""
        pred = parts[1] if len(parts) > 1 else record.key
        comp_anchors = r.qualifiers.get("compound_source_anchors")
        if comp_anchors and isinstance(comp_anchors, list):
            for anchor in comp_anchors:
                rows.append(
                    _BenchRow(
                        entity=subj,
                        attribute=pred,
                        value=record.content,
                        qualifiers=None,
                        source_anchor=anchor,
                    )
                )
        else:
            rows.append(
                _BenchRow(
                    entity=subj,
                    attribute=pred,
                    value=record.content,
                    qualifiers=None,
                    source_anchor=record.provenance.source_ids[0]
                    if (record.provenance and record.provenance.source_ids)
                    else "",
                )
            )
    return rows


async def _answer_and_score(
    engine: MemoryEngine,
    case: LoCoMoCase,
    namespace: str,
    llm_judge: Any,
    *,
    obs_date: str,
    sys_date: str,
) -> dict[str, Any]:
    """Run one answer pass and score recall + verdict for a case."""
    from houyi.adapters.memory.types import SessionContext

    answer = await engine.answer(
        case.question,
        session_context=SessionContext(
            session_id=namespace, current_observation_date=obs_date, current_system_date=sys_date
        ),
        top_k=RECALL_TOP_K,
    )
    rows = _recalls_to_rows(engine, answer.extras.get("recalls", []))
    recall_at_10, ndcg_at_10, precision_at_10 = _recall_metrics(
        rows, case.evidence, top_k=RECALL_TOP_K
    )
    verdict = await _judge(llm_judge, case, answer)
    return {
        "answer": answer,
        "rows": rows,
        "recall_at_10": recall_at_10,
        "ndcg_at_10": ndcg_at_10,
        "precision_at_10": precision_at_10,
        "verdict": verdict,
        "retrieve_ms": float(answer.extras.get("recall_ms", 0.0)),
    }


async def _evolve_and_rescore(
    engine: MemoryEngine,
    backfill: EmbeddingBackfillWorker,
    case: LoCoMoCase,
    namespace: str,
    llm_judge: Any,
    llm_reflect: Any | None = None,
    *,
    baseline: dict[str, Any],
    obs_date: str,
    sys_date: str,
    consolidate: bool = True,
    reflect: bool = True,
) -> dict[str, Any]:
    """Run one evolution pass (consolidation + reflection) and re-score.

    Reflection is failure-anchored: it re-extracts query-answering facts from
    the SOURCE turns for the failing question, grounds each against its source,
    and keeps only facts that are actually retrievable. Evolution never sees
    the gold answer, so any accuracy/recall gain is honest self-evolution, not
    fitting. Newly promoted records are embedding-backfilled before the second
    answer pass so the vector retriever can surface them.
    """
    report = engine.evolve(
        consolidate=consolidate,
        reflect=reflect,
        failing_queries=[case.question] if reflect else None,
        namespace=namespace,
        llm=llm_reflect,
    )

    # Backfill embeddings for any promoted records before the second pass.
    while await backfill.process_once():
        pass

    after = await _answer_and_score(
        engine, case, namespace, llm_judge, obs_date=obs_date, sys_date=sys_date
    )
    logger.info(
        "  [evolve] kept=%d before_correct=%s after_correct=%s",
        len(report.created_records),
        baseline["verdict"]["correct"],
        after["verdict"]["correct"],
    )
    consolidation = report.consolidation
    reflection = report.reflection
    return {
        "records_created": len(report.created_records),
        "k1_rows_closed": consolidation.rows_closed if consolidation else 0,
        "k1_triples_resolved": consolidation.triples_resolved if consolidation else 0,
        "reflection_extracted": reflection.facts_extracted if reflection else 0,
        "reflection_kept": reflection.facts_kept if reflection else 0,
        "reflection_retracted": reflection.facts_retracted if reflection else 0,
        "k1_skipped_accumulate": consolidation.skipped_accumulate if consolidation else 0,
        "before_correct": bool(baseline["verdict"]["correct"]),
        "after_correct": bool(after["verdict"]["correct"]),
        "before_recall_at_10": round(baseline["recall_at_10"], 4),
        "after_recall_at_10": round(after["recall_at_10"], 4),
        "before_ndcg_at_10": round(baseline["ndcg_at_10"], 4),
        "before_precision_at_10": round(baseline["precision_at_10"], 4),
        "after_ndcg_at_10": round(after["ndcg_at_10"], 4),
        "after_precision_at_10": round(after["precision_at_10"], 4),
        "after_answer": after["answer"].answer,
        "after_reason": after["verdict"]["reason"],
    }


async def _run_case_with_mode(
    case: LoCoMoCase,
    turn_writer: TurnWriter,
    worker: ExtractorWorker,
    extractor_counter: _CountingBatchExtractor,
    view: SQLiteEntityStateView,
    backend: SQLiteMemoryBackend,
    namespace: str,
    llm_answer: Any,
    llm_judge: Any,
    *,
    embedding_provider: str = "siliconflow",
    embedding_model: str | None = None,
    embedding_api_key: str | None = None,
    base_url: str | None = None,
    skip_ingestion: bool = False,
    use_window: bool = False,
    window_size: int = 10,
    evolve_mode: bool = False,
    llm_reflect: Any | None = None,
    consolidate: bool = True,
    reflect: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    extract_calls_per_case = 0
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
            "ndcg_at_10": 0.0,
            "precision_at_10": 0.0,
        }

    # Determine which turns to ingest (windowed or full)
    if use_window:
        ingest_idxs = set()
        for i in idxs:
            for j in range(max(0, i - window_size), min(len(turns), i + window_size + 1)):
                ingest_idxs.add(j)
    else:
        ingest_idxs = set(range(len(turns)))

    if not skip_ingestion:
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

        # Fully parallelize the queue draining!
        # Instead of doing 1024 sequential loops, we spin up concurrent asyncio tasks
        # that process batches in parallel. This saturates the LLM API and the disk cache.
        async def _drain_worker() -> int:
            total_processed = 0
            while True:
                processed = await worker.process_once()
                if processed == 0:
                    break
                total_processed += processed
            return total_processed

        # Run parallel batch process tasks concurrently
        drained_counts = await asyncio.gather(*[_drain_worker() for _ in range(DRAIN_WORKERS)])
        processed = sum(drained_counts)

        extract_calls_per_case = extractor_counter.calls - calls_before
        logger.info(
            "  extractor processed %d turns, %d LLM calls (parallelized)",
            processed,
            extract_calls_per_case,
        )
    else:
        logger.info("  [Cache Hit] Skipping ingestion & extraction for %s", case.sample_id)
        processed = len(turns)

    # Debug: inspect entity-state and raw-turn contents
    _ents = view.list_entities(namespace)
    logger.info("  entity-state entities: %s (count=%d)", _ents[:10], len(_ents))
    _pending_emb = len(backend.list_pending_embeddings())
    logger.info("  pending embeddings: %d", _pending_emb)

    retrieve_t0 = time.perf_counter()

    # Embedding provider for the vector retriever. Errors propagate so
    # bench failure modes stay loud.
    _env = EnvConfig.get()
    _emb_key: str = embedding_api_key or _env.siliconflow_api_key or ""
    if embedding_provider == "siliconflow":
        embedding_prov: EmbeddingProvider = DiskCacheWrapper(
            SiliconFlowEmbeddingProvider(
                api_key=_emb_key,
                model=embedding_model,
            )
        )
    elif embedding_provider == "dashscope":
        _ds_key: str = embedding_api_key or _env.dashscope_api_key or ""
        embedding_prov = DiskCacheWrapper(
            DashScopeEmbeddingProvider(
                api_key=_ds_key,
                model=embedding_model,
            )
        )
    else:
        embedding_prov = DiskCacheWrapper(
            make_embedding_provider(
                provider=embedding_provider,
                model=embedding_model,
                api_key=embedding_api_key,
            )
        )

    # Backfill pending embeddings before retrieval so vector search has data.
    backfill = EmbeddingBackfillWorker(
        backend=backend,
        provider=embedding_prov,
        config=EmbeddingBackfillConfig(batch_size=16),
    )
    backfill_total = 0
    while True:
        filled = await backfill.process_once()
        if filled == 0:
            break
        backfill_total += filled
    if backfill_total:
        logger.info("backfilled %d embeddings for vector search", backfill_total)

    # The default recall stack handles entity_state, timeline, iterative,
    # raw_turn, and (when an embedding provider is supplied) vector
    # retrieval. The hand-written wh-word entity hint filter is no
    # longer needed because EntityStateRetriever now drops question
    # words on every code path.
    # Answer path. Delegates to the SDK MemoryEngine.answer end-to-end.
    # The SDK runs the wired RecallOrchestrator + reasoning policies
    # and returns the AnswerResult directly.
    engine = MemoryEngine(
        MemoryStore(backend=backend),
        llm_adapter=llm_answer,
        recall_orchestrator=_build_default_recall_orchestrator(
            backend=backend,
            entity_state=view,
            embedding_provider=embedding_prov,
            config=_ablated_recall_config(),
            llm_adapter=llm_answer,
        ),
        embedding_provider=embedding_prov,
        entity_state=view,
    )
    last_turn = case.sample.turns[-1] if case.sample.turns else None
    obs_date = _normalize_observation_date(last_turn.session_datetime) if last_turn else None
    if not obs_date:
        obs_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    sys_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

    baseline = await _answer_and_score(
        engine, case, namespace, llm_judge, obs_date=obs_date, sys_date=sys_date
    )
    # Pure retrieval latency self-reported by the engine. The previous
    # wall-clock span here also covered embedding-provider construction,
    # backfill, and the answer-LLM generation, inflating P50/P95 by an
    # order of magnitude.
    answer_pipeline_ms = (time.perf_counter() - retrieve_t0) * 1000.0
    retrieve_ms = baseline["retrieve_ms"] or answer_pipeline_ms

    logger.info("  Generated answer: %s", baseline["answer"].answer[:200])
    logger.info("  Expected answer: %s", case.answer[:200])

    result: dict[str, Any] = {
        "case_id": f"{case.sample_id}:{case.question[:60]}",
        "category": case.category,
        "answer": baseline["answer"].answer,
        "expected": case.answer,
        "correct": baseline["verdict"]["correct"],
        "reason": baseline["verdict"]["reason"],
        "memories_count": len(baseline["rows"]),
        "turns_ingested": len(ingest_idxs),
        "retrieve_ms": round(retrieve_ms, 2),
        "answer_pipeline_ms": round(answer_pipeline_ms, 2),
        "recall_at_10": round(baseline["recall_at_10"], 4),
        "ndcg_at_10": round(baseline["ndcg_at_10"], 4),
        "precision_at_10": round(baseline["precision_at_10"], 4),
        "extract_calls_per_case": int(max(extract_calls_per_case, 0)),
        "duration_s": round(time.perf_counter() - t0, 1),
    }

    if evolve_mode:
        result["evolve"] = await _evolve_and_rescore(
            engine,
            backfill,
            case,
            namespace,
            llm_judge,
            baseline=baseline,
            obs_date=obs_date,
            sys_date=sys_date,
            llm_reflect=llm_reflect,
            consolidate=consolidate,
            reflect=reflect,
        )

    return result


class DiskCacheWrapper:
    def __init__(self, inner: Any, cache_file: str = "/tmp/locomo_llm_cache.json") -> None:
        self._inner = inner
        self._cache_file = Path(cache_file)
        self._cache: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._load_cache()

    def _get_key(
        self, messages: list[Any], temperature: float, max_tokens: int | None, **kwargs: Any
    ) -> str:
        # Normalize pydantic LLMMessage objects to dicts so the cache key is
        # JSON-serializable. The adapter contract accepts LLMMessage | dict,
        # but json.dumps (used for the hash) cannot serialize pydantic models,
        # which would raise and surface as a silent reflection fallback.
        norm_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                norm_messages.append(msg)
            else:
                role = getattr(msg, "role", None)
                role_val = getattr(role, "value", role)
                norm_messages.append({"role": role_val, "content": getattr(msg, "content", None)})
        # Create stable hash key for messages and parameters
        serialized = json.dumps(
            {
                "messages": norm_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"},
            },
            sort_keys=True,
        )
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text())
                logger.info(
                    "Loaded %d cached LLM/Embedding responses from %s",
                    len(self._cache),
                    self._cache_file,
                )
            except Exception:
                logger.warning("Failed to load LLM cache, starting fresh", exc_info=True)

    def _save_cache(self) -> None:
        try:
            self._cache_file.write_text(json.dumps(self._cache, indent=2))
        except Exception:
            logger.warning("Failed to save LLM cache to disk", exc_info=True)

    async def chat(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        key = self._get_key(messages, temperature, max_tokens, tools=tools, **kwargs)
        # 1. Thread-safe lock-free read check (Python dict lookups are atomic)
        if key in self._cache and "content" in self._cache[key]:
            cached_data = self._cache[key]

            @dataclass
            class FakeResponse:
                content: str

            return FakeResponse(content=cached_data["content"])

        # 2. Call real LLM without holding any locks
        response = await self._inner.chat(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        content = getattr(response, "content", None)
        if isinstance(content, str):
            # 3. Only lock to update the dict and write to disk asynchronously to prevent blocking threads
            async with self._lock:
                self._cache[key] = {"content": content}
            # Save cache to disk in a separate background thread so we never block the event loop with IO
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_cache)
        return response

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Hash each text independently or cache the whole batch
        # For simplicity and speed, cache the batch
        key = hashlib.md5(json.dumps(texts).encode("utf-8")).hexdigest()
        if key in self._cache and "embeddings" in self._cache[key]:
            return self._cache[key]["embeddings"]

        embeddings = await self._inner.embed(texts)
        async with self._lock:
            self._cache[key] = {"embeddings": embeddings}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_cache)
        return embeddings

    def dimension(self) -> int:
        return self._inner.dimension()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _ablated_recall_config():
    """Return a RecallPipelineConfig with _ABLATE_RETRIEVERS stripped or weights lowered.

    Returns None when no ablation is requested so the factory uses its
    own default config unchanged. Used to measure each retriever's
    marginal contribution without permanently editing the route table.
    """
    from houyi.adapters.memory.recall.orchestrator import (
        _DEFAULT_FUSION_WEIGHTS,
        _DEFAULT_ROUTE_TABLE,
        RecallPipelineConfig,
    )
    from houyi.adapters.memory.recall.types import QueryType, RetrieverKind

    if not _ABLATE_RETRIEVERS:
        # No route ablation: only override config when a non-default fusion
        # strategy is requested, else let the factory use its own defaults.
        if _FUSION_STRATEGY == "weighted":
            return None
        return RecallPipelineConfig(fusion_strategy=_FUSION_STRATEGY)

    # Handle weight-lowering ablation (e.g. graph_low)
    custom_weights = None
    strip_keys = set(_ABLATE_RETRIEVERS)
    if "graph_low" in strip_keys:
        strip_keys.remove("graph_low")
        # Build modified fusion weights dict
        custom_weights = {qt: dict(weights) for qt, weights in _DEFAULT_FUSION_WEIGHTS.items()}
        if QueryType.FACTUAL_LOOKUP in custom_weights:
            custom_weights[QueryType.FACTUAL_LOOKUP][RetrieverKind.GRAPH] = 2.0

    pruned_route = {
        qt: tuple(name for name in names if name not in strip_keys)
        for qt, names in _DEFAULT_ROUTE_TABLE.items()
    }

    if custom_weights:
        return RecallPipelineConfig(
            route_table=pruned_route,
            fusion_weights=custom_weights,
            fusion_strategy=_FUSION_STRATEGY,
        )
    return RecallPipelineConfig(route_table=pruned_route, fusion_strategy=_FUSION_STRATEGY)


async def _run_all(
    cases: list[LoCoMoCase],
    output_path: Path | None,
    concurrency: int = 1,
    extract_model: str = _MODEL_EXTRACT,
    answer_model: str = _MODEL_ANSWER,
    judge_model: str = _MODEL_JUDGE,
    evolve_model: str = _SF_MODEL_EVOLVE,
    api_key: str | None = None,
    base_url: str | None = None,
    llm_provider: str = "siliconflow",
    embedding_provider: str = "siliconflow",
    embedding_model: str | None = None,
    embedding_api_key: str | None = None,
    use_window: bool = False,
    window_size: int = 10,
    evolve_mode: bool = False,
    consolidate: bool = True,
    reflect: bool = True,
) -> dict:
    # Credential resolution is provider-scoped. SiliconFlow still requires an
    # explicit key (or SILICONFLOW_API_KEY) up front; dashscope defers to the
    # factory, which fails fast on a missing DASHSCOPE_API_KEY rather than
    # borrowing another provider's key.
    if llm_provider == "siliconflow":
        resolved_key = api_key or _ENV_API_KEY
        if not resolved_key:
            sys.exit("No API key: pass --api-key or set SILICONFLOW_API_KEY")
    else:
        resolved_key = api_key

    def _make_llm(model: str) -> DiskCacheWrapper:
        return DiskCacheWrapper(
            LLMAdapterFactory.create(
                llm_provider,
                model=model,
                api_key=resolved_key,
                base_url=base_url,
            )
        )

    llm_extract = _make_llm(extract_model)
    llm_answer = _make_llm(answer_model)
    llm_judge = _make_llm(judge_model)
    # Reflection LLM is only used in --evolve-mode; build it lazily so a normal
    # run does not construct an adapter (and resolve credentials) it never uses.
    llm_reflect = _make_llm(evolve_model) if evolve_mode else None
    logger.info(
        "provider=%s models: extract=%s answer=%s judge=%s evolve=%s window=%d concurrency=%d use_window=%s window_size=%d",
        llm_provider,
        extract_model,
        answer_model,
        judge_model,
        evolve_model if evolve_mode else "-",
        WINDOW,
        concurrency,
        use_window,
        window_size,
    )
    total = len(cases)
    results: list[dict] = [None] * total  # type: ignore[list-item]
    semaphore = asyncio.Semaphore(concurrency)
    run_token = uuid.uuid4().hex[:8]

    # Generate config hash based on extraction AND embedding parameters
    # to invalidate cache when either config changes (e.g. embedding dim).
    # The extraction prompt is hashed in too: a prompt change alters what
    # facts get extracted, so the pre-extracted session DB must rebuild.
    # Without this, a prompt fix would be silently masked by stale cached
    # extractions and the bench would validate against the old behavior.
    extract_config_payload = json.dumps(
        {
            "extract_model": extract_model,
            "batch_size": EXTRACT_BATCH_SIZE,
            "use_window": use_window,
            "window_size": window_size,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model or "",
            "extract_prompt_hash": hashlib.md5(
                _ATOMIC_FACT_BATCH_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
        sort_keys=True,
    )
    config_hash = hashlib.md5(extract_config_payload.encode("utf-8")).hexdigest()[:24]
    db_cache_dir = Path("/tmp/locomo_session_db_cache")
    db_cache_dir.mkdir(parents=True, exist_ok=True)

    async def _run_one(idx: int, case: LoCoMoCase) -> None:
        async with semaphore:
            db = Path(f"/tmp/locomo_bench_{run_token}_orchestrator_{idx}.db")

            # Check if we have a pre-ingested/extracted database cache for this sample
            if use_window:
                evidence_str = ",".join(sorted(case.evidence))
                evidence_hash = hashlib.md5(evidence_str.encode("utf-8")).hexdigest()[:8]
                cached_db = db_cache_dir / f"{case.sample_id}_{evidence_hash}_{config_hash}.db"
            else:
                cached_db = db_cache_dir / f"{case.sample_id}_{config_hash}.db"

            is_cached = False
            if cached_db.exists():
                import sqlite3

                try:
                    conn = sqlite3.connect(cached_db)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM memories")
                    count = cursor.fetchone()[0]
                    conn.close()
                    if count > 0:
                        is_cached = True
                    else:
                        logger.warning(
                            "Cache DB %s exists but has 0 memories, ignoring and rebuilding...",
                            cached_db.name,
                        )
                        cached_db.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(
                        "Failed to verify cache DB %s: %s, rebuilding...", cached_db.name, e
                    )
                    cached_db.unlink(missing_ok=True)

            if is_cached:
                logger.info(
                    "  [Cache Hit] Reusing pre-extracted session database for %s", case.sample_id
                )
                shutil.copy(cached_db, db)

            backend = SQLiteMemoryBackend(db_path=db)
            try:
                inbox = SQLiteCandidateInbox(backend)
                view = SQLiteEntityStateView(backend)
                extractor = AtomicFactExtractor(llm_extract, max_retries=1, batch_max_tokens=8192)
                counting_extractor = _CountingBatchExtractor(extractor)
                turn_writer = TurnWriter(backend, extract_trigger=all_of())
                # The default MemoryRecordPromoter is wired by ExtractorWorker
                # itself, so no explicit promoter argument is needed.
                worker = ExtractorWorker(
                    backend=backend,
                    extractor=counting_extractor,
                    entity_state=view,
                    candidate_inbox=inbox,
                    event_view=backend,
                    config=ExtractorWorkerConfig(batch_size=EXTRACT_BATCH_SIZE),
                    write_lock=asyncio.Lock(),
                )
                r = await _run_case_with_mode(
                    case,
                    turn_writer,
                    worker,
                    counting_extractor,
                    view,
                    backend,
                    f"locomo:{case.sample_id}",
                    llm_answer,
                    llm_judge,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_api_key=embedding_api_key,
                    base_url=base_url,
                    skip_ingestion=is_cached,  # Pass skip_ingestion flag to avoid redundant work
                    use_window=use_window,
                    window_size=window_size,
                    evolve_mode=evolve_mode,
                    llm_reflect=llm_reflect,
                    consolidate=consolidate,
                    reflect=reflect,
                )

                # Close the backend connection first to checkpoint and flush WAL records.
                backend.close()

                # If we successfully ingested and completed extraction from scratch, cache it!
                if not is_cached and r.get("reason") != "no_evidence_turns":
                    try:
                        shutil.copy(db, cached_db)
                        logger.info(
                            "  [Cache Save] Saved fully-extracted database for %s", case.sample_id
                        )
                    except Exception as ce:
                        logger.warning("Failed to save DB cache: %s", ce)

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
                # db.unlink(missing_ok=True)

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
    ndcg_values = [float(r.get("ndcg_at_10", 0.0)) for r in results if isinstance(r, dict)]
    precision_values = [
        float(r.get("precision_at_10", 0.0)) for r in results if isinstance(r, dict)
    ]
    report = {
        "recall_mode": "orchestrator",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "recall_at_10": round(sum(recall_at_10_values) / len(recall_at_10_values), 4)
        if recall_at_10_values
        else 0.0,
        "ndcg_at_10": round(sum(ndcg_values) / len(ndcg_values), 4) if ndcg_values else 0.0,
        "precision_at_10": round(sum(precision_values) / len(precision_values), 4)
        if precision_values
        else 0.0,
        "retrieve_p50_ms": round(_percentile(retrieve_samples, 0.5), 2),
        "retrieve_p95_ms": round(_percentile(retrieve_samples, 0.95), 2),
        "retrieve_p99_ms": round(_percentile(retrieve_samples, 0.99), 2),
        "by_category": {
            str(k): {**v, "accuracy": round(v["correct"] / v["total"], 4)}
            for k, v in sorted(by_cat.items())
        },
        "results": results,
    }
    if evolve_mode:
        report["evolve"] = _summarize_evolve(results, output_path)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("wrote %s", output_path)
    return report


def _summarize_evolve(results: list[dict], output_path: Path | None) -> dict:
    """Aggregate per-case evolve deltas and write a before/after artifact.

    Reports the honest end metric: QA accuracy and mean recall@10 before vs
    after one evolution pass over the same questions. Promoted memories are
    derived from stored records only (never the gold answer), so a positive
    delta reflects genuine self-evolution benefit.
    """
    evos = [r["evolve"] for r in results if isinstance(r, dict) and "evolve" in r]
    n = len(evos)
    if n == 0:
        return {"cases": 0}

    before_correct = sum(1 for e in evos if e["before_correct"])
    after_correct = sum(1 for e in evos if e["after_correct"])
    gained = [e for e in evos if e["after_correct"] and not e["before_correct"]]
    regressed = [e for e in evos if e["before_correct"] and not e["after_correct"]]
    before_acc = before_correct / n
    after_acc = after_correct / n
    before_recall = sum(e["before_recall_at_10"] for e in evos) / n
    after_recall = sum(e["after_recall_at_10"] for e in evos) / n
    records_created = sum(e["records_created"] for e in evos)

    summary = {
        "cases": n,
        "records_created": records_created,
        "before_accuracy": round(before_acc, 4),
        "after_accuracy": round(after_acc, 4),
        "accuracy_delta": round(after_acc - before_acc, 4),
        "before_recall_at_10": round(before_recall, 4),
        "after_recall_at_10": round(after_recall, 4),
        "recall_delta": round(after_recall - before_recall, 4),
        "gained": len(gained),
        "regressed": len(regressed),
    }

    # Persist a BeforeAfterReport artifact alongside the JSON output so the
    # evolution pass leaves an auditable, comparable record on disk.
    try:
        from houyi.application.evolution.before_after import (
            BeforeAfterReport,
            make_run_id,
            write_report,
        )

        out_dir = (
            output_path.parent if output_path else Path("benchmark/output/memory")
        ) / "evolve"
        report = BeforeAfterReport(
            run_id=make_run_id(),
            optimizer="memory_dreamer",
            artifact_type="locomo_qa",
            baseline_content=f"{n} questions answered before evolution",
            optimized_content=f"{records_created} memories promoted via self-evolution",
            baseline_score=round(before_acc, 4),
            optimized_score=round(after_acc, 4),
            delta=round(after_acc - before_acc, 4),
            sample_size=n,
            signal_count=records_created,
            verdict="promote" if after_acc > before_acc else "hold",
            reason="accuracy_gain" if after_acc > before_acc else "no_gain",
            metrics={
                "before_recall_at_10": round(before_recall, 4),
                "after_recall_at_10": round(after_recall, 4),
                "gained": float(len(gained)),
                "regressed": float(len(regressed)),
            },
        )
        path = write_report(report, out_dir)
        summary["report_path"] = str(path)
        logger.info("wrote evolve before/after report %s", path)
    except Exception:
        logger.warning("failed to write evolve before/after report", exc_info=True)

    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Cap tokenizer/math-lib threads before the lazy torch + sentence_transformers
    # import. With --embedding-provider local and several cases encoding
    # concurrently via asyncio.to_thread, tokenizers' Rust parallelism and torch
    # thread pools oversubscribe, surfacing on macOS as "leaked semaphore"
    # warnings and intermittent SIGSEGV (exit 139). setdefault keeps any operator
    # override intact.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = _parse_args()

    if args.ablate_retrievers:
        _ABLATE_RETRIEVERS.update(args.ablate_retrievers)
        logger.info(
            "ABLATION: stripping retrievers from route table: %s", sorted(_ABLATE_RETRIEVERS)
        )

    global _FUSION_STRATEGY
    _FUSION_STRATEGY = args.fusion_strategy
    if _FUSION_STRATEGY != "weighted":
        logger.info("FUSION: using %s strategy", _FUSION_STRATEGY)

    if args.case_pair or args.case:
        from houyi.adapters.memory.bench.locomo import load_locomo_all

        all_cases = load_locomo_all()
    else:
        all_cases = load_locomo_balanced()

    if args.case_pair:
        wanted: set[tuple[str, str]] = set()

        def _norm(s: str) -> str:
            # Normalize backtick (U+0060) and curly quotes to plain
            # apostrophe (U+0027) so shell quoting mismatches do not
            # prevent matching dataset questions that use backticks.
            return s.lower().replace("`", "'").replace("‘", "'").replace("’", "'")

        raw_pairs = [item for group in args.case_pair for item in group]
        for raw in raw_pairs:
            if "::" not in raw:
                sys.exit(f"Invalid --case-pair entry (missing '::'): {raw!r}")
            conv_id, question = raw.split("::", 1)
            conv_id = conv_id.strip()
            question = question.strip()
            if not conv_id or not question:
                sys.exit(f"Invalid --case-pair entry (empty side): {raw!r}")
            wanted.add((_norm(conv_id), _norm(question)))

        cases = [c for c in all_cases if (_norm(c.sample_id), _norm(c.question.strip())) in wanted]
        if not cases:
            sys.exit(f"No cases found for --case-pair entries: {raw_pairs}")
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

    # Resolve per-role model defaults from the chosen provider only when the
    # caller did not pin a model explicitly. Provider sets are independent so
    # one provider never inherits the other's identifiers.
    provider_defaults = _PROVIDER_MODEL_DEFAULTS[args.llm_provider]
    extract_model = args.extract_model or provider_defaults["extract"]
    answer_model = args.answer_model or provider_defaults["answer"]
    judge_model = args.judge_model or provider_defaults["judge"]
    evolve_model = args.evolve_model or provider_defaults["evolve"]

    report = asyncio.run(
        _run_all(
            cases,
            out_path,
            concurrency=args.concurrency,
            extract_model=extract_model,
            answer_model=answer_model,
            judge_model=judge_model,
            evolve_model=evolve_model,
            api_key=args.api_key,
            base_url=args.base_url,
            llm_provider=args.llm_provider,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_api_key=args.embedding_api_key,
            use_window=args.use_window,
            window_size=args.window_size,
            evolve_mode=args.evolve_mode,
            consolidate=not args.no_consolidate,
            reflect=not args.no_reflect,
        )
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
