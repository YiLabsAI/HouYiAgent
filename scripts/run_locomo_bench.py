"""LoCoMo benchmark — evidence-window only, fast model.

Usage:
  uv run python scripts/run_locomo_bench.py --sample 5
  uv run python scripts/run_locomo_bench.py --sample 200 --output reports/locomo.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import datetime
import hashlib
import json
import logging
import os
import platform
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
    _atomic_fact_batch_response_format,
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
        "--clear-case-cache",
        nargs="+",
        default=None,
        metavar="SAMPLE_ID",
        help=(
            "Clear the named cases' full cache footprint before the run: each case's "
            "DB cache file(s) and every LLM/embedding cache entry namespaced under "
            "that sample id. Forces a fresh re-extraction for those cases only, "
            "without nuking the rest of the cache. Accepts multiple ids so one "
            "command can clear every failing case and re-run them together. "
            "e.g. --clear-case-cache conv-42 conv-47 conv-49"
        ),
    )
    p.add_argument(
        "--debug-trace",
        action="store_true",
        default=False,
        help=(
            "Capture per-stage recall snapshots (dbg_raw/dbg_fused/dbg_reranked/"
            "dbg_final/dbg_mmr_dropped) into each case's result JSON for offline "
            "root-causing of where a gold fact drops. Off by default -- tracing "
            "adds per-query overhead and bloats the output, so enable only for "
            "diagnostic runs."
        ),
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


# Per-case cache namespace for the LLM DiskCache. The LLM cache keys entries
# by md5 of the call payload alone, which carries no case id; a stale cached
# extraction for case X could otherwise be served to a re-run of X after a
# code/prompt change that left the turn text identical (same messages =>
# same md5 => cache hit => the bad extraction frozen in, defeating any
# attempt to re-extract that one case). Namespacing every key with the
# running case's sample_id makes each case's cache entries independently
# deletable (--clear-case-cache) without a whole-run bypass.
# asyncio task-scoped: each _run_one task sets its own value, so concurrency
# is safe; LLM calls stay on the event loop (no to_thread), so the value is
# visible at every key computation.
_CURRENT_CASE_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_CURRENT_CASE_ID", default="shared"
)


def _case_prefix() -> str:
    return f"{_CURRENT_CASE_ID.get()}:"


def _is_namespaced_cache_key(key: str) -> bool:
    # Namespaced keys are "{case_id}:{md5}"; legacy keys (pre-namespace) are
    # bare 32-hex md5 with no colon. Drop legacy entries on load so the cache
    # never accumulates dead weight from before the refactor and a stale
    # pre-refactor response can never be served under a new key.
    return ":" in key


class _CountingBatchExtractor:
    """Thin pass-through counter around AtomicFactExtractor.

    Counts extraction invocations per case (extract_calls_per_case): 0 means
    the DB cache hit and ingestion was skipped, >0 means extraction ran. The
    cross-run/cross-question extraction dedup this class used to do via a
    persistent TurnCache is now handled by the per-case namespaced LLM
    DiskCache, which is prompt-aware (a prompt change changes the messages
    md5 and invalidates the entry) and re-parses on every hit, so it carries
    none of the TurnCache staleness risk (TurnCache was blind to both prompt
    changes and parsing-logic changes because it cached the parsed result).
    """

    def __init__(self, inner: AtomicFactExtractor) -> None:
        self._inner = inner
        self.calls = 0
        # Per-anchor extraction yield: {anchor: {"facts": n, "events": m}}.
        # Populated only when extraction actually runs (DB cache miss); stays
        # empty on a DB-cache hit (skip_ingestion), which is itself the signal
        # that no fresh extraction happened for this case.
        self.per_anchor: dict[str, dict[str, int]] = {}

    def _record(self, source_anchor: str | None, res: Any) -> None:
        anchor = (source_anchor or "").strip() or "<no-anchor>"
        slot = self.per_anchor.setdefault(anchor, {"facts": 0, "events": 0})
        slot["facts"] += len(getattr(res, "facts", None) or [])
        slot["events"] += len(getattr(res, "events", None) or [])

    async def extract(self, text: str, source_anchor: str | None) -> Any:
        self.calls += 1
        res = await self._inner.extract(text, source_anchor)
        self._record(source_anchor, res)
        return res

    async def extract_batch(
        self, turns: list[tuple[str, str | None]], namespace: str = "default"
    ) -> list[Any]:
        self.calls += 1
        results = await self._inner.extract_batch(turns, namespace=namespace)
        for (_text, anchor), res in zip(turns, results, strict=False):
            self._record(anchor, res)
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


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _light_stem(token: str) -> str:
    t = token.strip().lower()
    if len(t) > 4 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 3 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _token_subset_match(expected: str, answer: str) -> bool:
    exp = _normalize_text(expected)
    ans = _normalize_text(answer)
    if not exp or not ans:
        return False
    if exp in ans:
        return True

    exp_tokens = {_light_stem(tok) for tok in exp.split() if tok}
    ans_tokens = {_light_stem(tok) for tok in ans.split() if tok}
    if exp_tokens and exp_tokens.issubset(ans_tokens):
        return True

    segments = [s.strip() for s in expected.split(",") if s.strip()]
    if len(segments) >= 2:
        for seg in segments:
            seg_norm = _normalize_text(seg)
            if not seg_norm:
                continue
            if seg_norm in ans:
                continue
            seg_tokens = {_light_stem(tok) for tok in seg_norm.split() if tok}
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


def _provenance() -> dict[str, Any]:
    """Record which interpreter + deps produced this run, so any later
    analysis can verify whether the cross-encoder was actually loaded.

    A result JSON with no provenance cannot self-certify that its rerank
    numbers came from the cross-encoder rather than a silent heuristic
    fallback -- which is exactly the gap that let a bare-python3 run pass
    as a cross-encoder run. Provenance turns a convention (use uv run) into
    a verifiable fact in the artifact.
    """
    info: dict[str, Any] = {
        "python": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import sentence_transformers as st

        info["sentence_transformers"] = st.__version__
    except Exception as exc:  # pragma: no cover - diagnostic
        info["sentence_transformers"] = f"not installed: {type(exc).__name__}"
    # Probe whether the cross-encoder model actually loads in this env.
    try:
        from houyi.adapters.memory.recall.rerank_cross_encoder import (
            CrossEncoderReranker,
        )

        info["cross_encoder_loads"] = CrossEncoderReranker()._load_model() is not None
    except Exception as exc:  # pragma: no cover - diagnostic
        info["cross_encoder_loads"] = f"error: {type(exc).__name__}: {exc}"
    return info


async def _judge(llm_judge: Any, case: LoCoMoCase, answer: AnswerResult) -> dict:
    if _token_subset_match(case.answer, answer.answer):
        return {"correct": True, "reason": "semantic_match"}

    judge_llm = LLMMemoryJudge(_JudgeLLM(llm_judge), timeout_seconds=20.0, max_tokens=16)
    normalized_case = replace(case, answer=_normalize_dates(case.answer))
    normalized_answer = replace(answer, answer=_normalize_dates(answer.answer))
    verdict = await judge_llm.judge(normalized_case, normalized_answer)
    # Retry once on transient failure (network timeout, empty response)
    if verdict.reason in ("judge_llm_failed", "judge_parse_failed"):
        verdict = await judge_llm.judge(normalized_case, normalized_answer)
    # Accuracy counts only real answers: a token-subset match (above) or an
    # LLM-judged MATCH. Abstention is NOT correct -- if the gold is genuinely
    # absent from the conversation the case is a benchmark-data problem to
    # exclude, not a correct abstention; if the gold is present the system
    # failed to recall or reason it. Either way abstention is wrong. The
    # reason is kept so the abstention count stays observable.
    return {"correct": verdict.reason == "llm_match", "reason": verdict.reason}


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


_GOLD_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "and",
        "or",
        "is",
        "was",
        "were",
        "are",
        "am",
        "be",
        "been",
        "being",
        "did",
        "does",
        "do",
        "with",
        "for",
        "from",
        "by",
        "as",
        "it",
        "this",
        "that",
        "he",
        "she",
        "her",
        "his",
        "their",
        "they",
        "you",
        "your",
        "our",
        "my",
        "i",
        "few",
        "years",
        "year",
        "before",
        "after",
        "first",
        "new",
        "what",
        "when",
        "where",
        "who",
        "how",
        "kind",
        "type",
        "all",
        "both",
        "has",
        "have",
        "had",
        "more",
        "most",
        "some",
        "any",
        "such",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "many",
        "much",
        "often",
        "long",
        "sort",
        "between",
        "among",
        "during",
        "than",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "also",
        "just",
        "only",
        "very",
        "too",
        "each",
        "every",
        "these",
        "those",
        "other",
        "another",
        "same",
        "about",
        "because",
        "while",
        "since",
        "until",
        "without",
        "within",
        "across",
        "through",
        "along",
        "around",
        "behind",
        "above",
        "below",
        "away",
        "back",
        "today",
        "tomorrow",
        "yesterday",
    ]
)


# _MONTH_MAP is defined above (shared with the answer date parser). Date
# normalization keeps YYYY-MM granularity minimum -- a bare year appears in
# many candidates' text and over-matches, so it is excluded.


def _date_tokens(text: str) -> set[str]:
    """Normalized date tokens at YYYY-MM granularity or finer.

    Bare YYYY is deliberately excluded: a bare year appears in many
    unrelated candidates and would over-match, the same common-token
    over-fit failure mode as short keyword matching. YYYY-MM keeps the date
    signal discriminative.
    """
    tokens: set[str] = set()
    for y, m, d in re.findall(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text):
        tokens.add(f"{y}-{int(m):02d}-{int(d):02d}")
        tokens.add(f"{y}-{int(m):02d}")
    for y, m in re.findall(r"\b(20\d{2})-(\d{1,2})\b", text):
        tokens.add(f"{y}-{int(m):02d}")
    low = text.lower()
    # Month-name + NEAREST year (NOT cartesian). The cartesian product
    # (every year x every month) fabricates dates: "March 2022 and again in
    # 2023" must yield only 2022-03, not a made-up 2023-03. Each month
    # occurrence pairs with the year closest to it in the text.
    month_positions: list[tuple[int, int]] = []
    for name, num in _MONTH_MAP.items():
        for m in re.finditer(r"\b" + name + r"\b", low):
            month_positions.append((m.start(), num))
    year_positions = [(m.start(), m.group()) for m in re.finditer(r"\b20\d{2}\b", text)]
    for mpos, mnum in month_positions:
        if year_positions:
            nearest = min(year_positions, key=lambda yp: abs(yp[0] - mpos))
            tokens.add(f"{nearest[1]}-{mnum:02d}")
    return tokens


def _extract_qa_tokens(question: str, answer: str) -> tuple[set[str], set[str], str | None]:
    """Split question + answer into (content_tokens, proper_nouns, subject_entity).

    content_tokens: lowercase non-stopword tokens len>2 (common-noun signal
    -- the descriptive nouns in the question/answer that are not proper
    names). proper_nouns: lowercase tokens that were capitalized in the
    source (named entities), stopword-filtered so sentence-initial
    "When"/"Where" do not count.

    subject_entity: the first proper noun in the QUESTION. It is the
    conversation principal the question is about and is EXCLUDED from the
    matching proper-noun set: subject entities appear in nearly every fact's
    text, so matching them floods gold with same-subject noise.
    """
    content: set[str] = set()
    proper: set[str] = set()
    subject_entity: str | None = None
    for text, is_question in ((question, True), (answer, False)):
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text):
            if len(tok) <= 2:
                continue
            low = tok.lower()
            if low in _GOLD_STOPWORDS:
                continue
            if tok[0].isupper():
                proper.add(low)
                if is_question and subject_entity is None:
                    subject_entity = low
            else:
                content.add(low)
    matching_proper = proper - ({subject_entity} if subject_entity else set())
    return content, matching_proper, subject_entity


def _content_match(c: dict, content_tokens: set[str], proper_nouns: set[str]) -> bool:
    """True when a candidate is gold by content.

    (a) a proper-noun token appears in the candidate OBJECT (object position
    only, so subject entities are excluded). This is the single-token gold
    path: a proper-noun answer member survives without a 2-token co-occurrence.
    (b) >=2 content tokens co-occur in the full candidate blob (multi-word
    guard against common-word false positives).

    No lowercase topic-noun heuristic: without a POS tagger, a half-built
    noun extractor is a new over-match surface. Lowercase topic nouns rely
    on the >=2 co-occurrence rule or fall through to evidence-turn
    (turn-inferred) -- honest, not faked.

    Matching is word-boundary (token-set), never substring: a short token
    must not match a longer word that merely contains it (e.g. a 3-letter
    function word matching an unrelated longer word). Substring matching is
    a common-token over-fit failure mode; exact-token matching prevents it.
    """
    obj_tokens = set(str(c.get("o", "")).lower().split())
    if proper_nouns and (proper_nouns & obj_tokens):
        return True
    if content_tokens:
        blob_tokens = set(_cand_blob(c).split())
        if len(content_tokens & blob_tokens) >= 2:
            return True
    return False


def _member_key(c: dict, content_tokens: set[str], proper_nouns: set[str]) -> tuple:
    """Identity of the gold member a candidate matches, for dedup.

    coverage counts DISTINCT gold members, not candidate rows: the same
    member is often recalled as several rows (duplicate anchors from the
    same turn, or the same entity surfaced by multiple retrievers in
    different surface forms). Counting rows inflates the apparent miss.

    Key = the matched proper-nouns in the object, or if matched only by
    content tokens, the matched content-token set. Imperfect: two
    paraphrased members that share no proper-noun or content token will not
    dedup -- documented limitation.
    """
    obj_tokens = set(str(c.get("o", "")).lower().split())
    matched_proper = proper_nouns & obj_tokens
    if matched_proper:
        return ("p", frozenset(matched_proper))
    blob_tokens = set(_cand_blob(c).split())
    return ("c", frozenset(content_tokens & blob_tokens))


def _evidence_turn_match(c: dict, evidence_turns: tuple[str, ...]) -> bool:
    """True when the candidate source_anchor matches a gold evidence turn.

    Supplementary signal only -- compound member anchors are not chased
    (a compound's member turns are not in the snapshot). Used when content
    match is empty -- always reported as turn-inferred, never as certain
    gold.
    """
    anchor = str(c.get("a", ""))
    if not anchor or anchor.startswith("fact:"):
        return False
    return any(anchor.endswith(f":{ev}") for ev in evidence_turns)


def _answer_value_in_pool(pool: list[dict], answer_date_tokens: set[str]) -> bool | None:
    """Parallel signal: is the answer's date VALUE in the recall pool?

    Returns None when the answer carries no date (check skipped), True when a
    YYYY-MM token from the answer appears in any pool candidate, False
    otherwise. Kept as a separate field -- never folded into fate.

    Named value_in_pool (not value_extracted): the scan sees the post-recall
    pool, so False conflates two distinct failures -- the date fact may be in
    the store but unretrieved (retrieve-miss), or never extracted at all
    (extract-miss). The route disambiguates these via whether the evidence
    turn was recalled (evidence_recalled): recalled + value-absent =
    extract-within-turn; not-recalled = retrieve-miss.
    """
    if not answer_date_tokens:
        return None
    pool_dates: set[str] = set()
    for c in pool:
        pool_dates |= _date_tokens(_cand_blob(c))
    return bool(answer_date_tokens & pool_dates)


def _route_of(
    *,
    classification: str,
    correct: bool,
    value_in_pool: bool | None,
    evidence_recalled: bool | None,
    is_enum: bool,
    coverage: tuple[int, int],
    tail_leak: bool,
    rerank_miss: bool,
) -> str:
    """Route key is (fate x correct x value x coverage).

    correct=True preempts to exempt: a correctly-answered case is not a fix
    target, even if the date value was absent from the pool (the answerer
    may have inferred it -- a latent risk, surfaced via the value_in_pool
    field, but not a routing failure).

    value_in_pool=False preempts enum: a date whose value is not in the pool
    is an extract/retrieve miss regardless of enumeration.

    Enumeration questions route by gold coverage, not worst-of-one fate: a
    single missed member must not mask that another member reached final
    only via tail-leak (rank beyond the cutoff, near-zero score). The
    composite route (coverage + rerank-miss + tail-leak) surfaces all
    diseases instead of one.
    """
    if correct:
        return "exempt-correct"
    # value-absent preempts enum: a date question whose value is not in the
    # pool is an extract/retrieve miss regardless of whether it is also
    # enumeration -- do not let the enum branch mask extract-within-turn
    # (point 2b: is_enum used to fire first and lose the value signal).
    if value_in_pool is False:
        if evidence_recalled is True:
            return "extract-within-turn"
        if evidence_recalled is False:
            return "retrieve-miss"
        return "value-not-in-pool"
    if is_enum:
        if coverage[0] >= coverage[1]:
            return "answerer (all members in final, answered wrong)"
        parts = [f"enum-coverage {coverage[0]}/{coverage[1]}"]
        if rerank_miss:
            parts.append("rerank-miss")
        if tail_leak:
            parts.append("tail-leak")
        return " +".join(parts)
    if classification == "NOT-IN-POOL":
        return "extraction-retrieve"
    if classification == "IN-FINAL":
        return "answerer"
    return "rerank-boost"


def _gold_keywords(answer: str) -> list[str]:
    """Tokenize a gold answer into matchable keywords (lowercase, alpha, non-stopword)."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", answer.lower())
    return [t for t in tokens if t not in _GOLD_STOPWORDS and len(t) > 2]


def _cand_blob(c: dict) -> str:
    return " ".join(str(c.get(k, "")) for k in ("s", "p", "o", "a")).lower()


def _cand_key(c: dict) -> tuple:
    """Stable content identity for matching a candidate across snapshots.

    dbg_reranked_raw / dbg_reranked / dbg_final are independent _dbg_snapshot
    calls over the same underlying candidates, so id() never matches across
    them. The (subject, predicate, object, anchor) tuple does. object is
    truncated identically (40 chars) in every snapshot, so the key is stable.
    """
    return (c.get("s"), c.get("p"), c.get("o"), c.get("a"))


def _gold_fact_fate(trace: dict, case: LoCoMoCase, *, correct: bool) -> dict:
    """Fact-level gold fate from the recall trace (content-primary).

    Gold matching priority:

    1. content (primary): proper-noun in OBJECT OR >=2 content tokens
       co-occurring in the blob. Covers vector-recall gold that
       evidence-turn cannot see -- a vector-recall fact carries a hash
       anchor with no turn id (the store strips turn metadata), so content
       is its only matchable signal.
    2. evidence-turn (supplementary, turn-inferred): only when content is
       empty. Covers lowercase topic nouns without a POS tagger: a
       lowercase noun does not content-match, so the case falls through to
       the evidence turn (turn-inferred, honestly flagged, not faked via a
       noun extractor).

    worst-fate runs ONLY over content-matched candidates: turn siblings are
    not let near it, so a noise sibling from a gold turn cannot drag the
    classification.

    answer-value-in-pool is a PARALLEL field, never folded into fate: when
    the answer's date value is absent from the pool, the case routes to
    extract-within-turn (evidence turn recalled) or retrieve-miss (not),
    not masked as "recalled".

    Enumeration questions route by gold coverage (distinct members), not
    worst-of-one fate: a single missed member must not mask that another
    member reached final only via tail-leak. The composite route (coverage
    + rerank-miss + tail-leak) surfaces all diseases.

    Known limitation: _dbg_snapshot truncates object to 40 chars, so a
    token past char 40 in a long object is missed; may under-match on
    long-object facts.

    Ranks on RAW pre-boost rerank score (dbg_reranked_raw); window cutoff
    is len(final), not a hardcoded 10.
    """
    final = trace.get("dbg_final", [])
    reranked = trace.get("dbg_reranked_raw") or trace.get("dbg_reranked", [])
    content, proper, subject = _extract_qa_tokens(case.question, case.answer)
    answer_dates = _date_tokens(case.answer)
    evidence = case.evidence

    out: dict[str, Any] = {
        "subject_entity": subject,
        "content_tokens": sorted(content)[:12],
        "proper_nouns": sorted(proper)[:12],
        "evidence_turns": list(evidence),
        "answer_value_tokens": sorted(answer_dates)[:8],
        "facts": [],
    }

    if not reranked:
        out["classification"] = "NOT-IN-POOL"
        out["note"] = "empty reranked pool (no recall)"
        out["value_in_pool"] = None
        out["route"] = _route_of(
            classification="NOT-IN-POOL",
            correct=correct,
            value_in_pool=None,
            evidence_recalled=False,
            is_enum=False,
            coverage=(0, 0),
            tail_leak=False,
            rerank_miss=False,
        )
        return out

    final_keys = {_cand_key(c) for c in final}
    final_n = len(final)
    rs_sorted = sorted([c.get("rs") or 0.0 for c in reranked], reverse=True)
    cutoff_rank = final_n if final_n > 0 else len(reranked)
    cutoff = (
        rs_sorted[cutoff_rank - 1]
        if 0 < cutoff_rank <= len(rs_sorted)
        else (rs_sorted[-1] if rs_sorted else 0.0)
    )

    def _fate_of(c: dict, rank: int) -> str:
        in_final = _cand_key(c) in final_keys
        if in_final:
            return "IN-FINAL"
        return "MMR-OUT" if rank <= cutoff_rank else "RERANK-OUT"

    # Primary: content-matched gold. worst-fate runs over THIS set only (H).
    matched: list[dict] = []
    gold_in_final = 0
    member_in_final: dict[tuple, bool] = {}
    for i, c in enumerate(reranked):
        if _content_match(c, content, proper):
            rank = i + 1
            cls = _fate_of(c, rank)
            in_final = cls == "IN-FINAL"
            if in_final:
                gold_in_final += 1
            matched.append(
                {
                    "object": c.get("o"),
                    "rank": rank,
                    "rs": round(c.get("rs") or 0.0, 4),
                    "in_final": in_final,
                    "classification": cls,
                    "match": "content",
                }
            )
            mk = _member_key(c, content, proper)
            member_in_final[mk] = member_in_final.get(mk, False) or in_final

    turn_inferred = False
    if not matched and evidence:
        # Fallback: evidence-turn (turn-inferred). Never certain gold --
        # the matched candidate may be a turn sibling, not the answer fact
        # (a different fact from the same evidence turn). Reported with
        # caveat, not let into worst-fate of content.
        for i, c in enumerate(reranked):
            if _evidence_turn_match(c, evidence):
                rank = i + 1
                cls = _fate_of(c, rank)
                in_final = cls == "IN-FINAL"
                if in_final:
                    gold_in_final += 1
                matched.append(
                    {
                        "object": c.get("o"),
                        "rank": rank,
                        "rs": round(c.get("rs") or 0.0, 4),
                        "in_final": in_final,
                        "classification": cls,
                        "match": "turn-inferred",
                    }
                )
                mk = _member_key(c, content, proper)
                member_in_final[mk] = member_in_final.get(mk, False) or in_final
        turn_inferred = bool(matched)

    value_in_pool = _answer_value_in_pool(reranked, answer_dates)
    # Did the evidence turn get recalled at all? Disambiguates value-absent:
    # recalled-turn + value-absent = extract-within-turn; absent-turn =
    # retrieve-miss. None when there is no evidence annotation.
    evidence_recalled: bool | None = None
    if evidence:
        evidence_recalled = any(_evidence_turn_match(c, evidence) for c in reranked)

    # Enumeration: the PIPELINE's own signal (trace.enumeration_coverage),
    # NOT the matched-row count. Row count misjudges cases where one member
    # is recalled as several rows (duplicate anchors or multiple retriever
    # forms) -- the pipeline flag is the sound signal of whether the window
    # was widened for enumeration.
    is_enum = bool(trace.get("enumeration_coverage"))
    # coverage counts DISTINCT gold members (deduped by member_key), not
    # candidate rows -- the same member recalled as N rows is still one
    # member.
    distinct_total = len(member_in_final)
    distinct_in_final = sum(1 for v in member_in_final.values() if v)
    tail_leak = any(m["in_final"] and m["rank"] > cutoff_rank for m in matched)
    rerank_miss = any(m["classification"] in ("MMR-OUT", "RERANK-OUT") for m in matched)

    out["window_relevance"] = {
        "window_size": final_n,
        "gold_in_window": gold_in_final,
        "noise_in_window": max(final_n - gold_in_final, 0),
    }
    out["value_in_pool"] = value_in_pool
    out["evidence_recalled"] = evidence_recalled
    out["cutoff_rank"] = cutoff_rank
    out["cutoff_rs"] = round(cutoff, 4)
    out["tail_leak"] = tail_leak
    out["coverage"] = [distinct_in_final, distinct_total]

    if not matched:
        out["classification"] = "NOT-IN-POOL"
        out["note"] = "gold not in reranked pool by content or evidence-turn"
        out["route"] = _route_of(
            classification="NOT-IN-POOL",
            correct=correct,
            value_in_pool=value_in_pool,
            evidence_recalled=evidence_recalled,
            is_enum=False,
            coverage=(0, 0),
            tail_leak=False,
            rerank_miss=False,
        )
        return out

    priority = {"NOT-IN-POOL": 0, "RERANK-OUT": 1, "MMR-OUT": 2, "IN-FINAL": 3}
    worst = min(matched, key=lambda m: priority[m["classification"]])
    out["classification"] = worst["classification"]
    out["turn_inferred"] = turn_inferred
    out["facts"] = matched
    out["route"] = _route_of(
        classification=worst["classification"],
        correct=correct,
        value_in_pool=value_in_pool,
        evidence_recalled=evidence_recalled,
        is_enum=is_enum,
        coverage=(distinct_in_final, distinct_total),
        tail_leak=tail_leak,
        rerank_miss=rerank_miss,
    )
    return out


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
        "trace": answer.extras.get("trace", {}),
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
    shared_cache: DiskCache,
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
    debug_trace: bool = False,
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
            ),
            shared_cache,
        )
    elif embedding_provider == "dashscope":
        _ds_key: str = embedding_api_key or _env.dashscope_api_key or ""
        embedding_prov = DiskCacheWrapper(
            DashScopeEmbeddingProvider(
                api_key=_ds_key,
                model=embedding_model,
            ),
            shared_cache,
        )
    else:
        embedding_prov = DiskCacheWrapper(
            make_embedding_provider(
                provider=embedding_provider,
                model=embedding_model,
                api_key=embedding_api_key,
            ),
            shared_cache,
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
        debug_trace=debug_trace,
    )
    last_turn = case.sample.turns[-1] if case.sample.turns else None
    obs_date = _normalize_observation_date(last_turn.session_datetime) if last_turn else None
    if not obs_date:
        obs_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    sys_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

    baseline = await _answer_and_score(
        engine,
        case,
        namespace,
        llm_judge,
        obs_date=obs_date,
        sys_date=sys_date,
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
    # Per-anchor extraction yield so a diagnostic run can show, per source
    # turn, how many facts/events the extractor produced (e.g. did the gold
    # evidence turn D1:7 yield 0 facts). Empty when ingestion was skipped
    # (DB cache hit) -- which is itself the "no fresh extraction" signal.
    result["extraction_per_anchor"] = dict(extractor_counter.per_anchor)
    if debug_trace:
        # Per-stage recall snapshots for offline root-causing. Only present
        # when --debug-trace is on so normal-run output stays lean.
        result["trace"] = baseline.get("trace", {})
        # Fact-granularity gold fate: distinguish "gold answer fact out by low
        # rerank score" / "dropped by MMR" / "never in pool" / "in final".
        # recall@10 is turn-level (any fact from the gold turn counts as a hit)
        # so it hides a gold ANSWER fact that was rerank-scored low or dropped
        # by MMR while a sibling fact from the same turn covers the turn. This
        # metric is the fact-level truth that recall@10 masks.
        result["gold_fact_fate"] = _gold_fact_fate(
            baseline.get("trace", {}), case, correct=result["correct"]
        )

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


class DiskCache:
    """Shared, file-backed LLM/embedding response cache.

    One instance is shared across every per-model DiskCacheWrapper
    (extract/answer/judge/reflect) so they all read and write the SAME
    in-memory dict. Previously each wrapper owned a private dict but wrote
    the same file, so the last wrapper to save clobbered every other's
    entries (only the last writer's responses survived a run) -- the
    extraction LLM cache was effectively non-functional and every DB-miss
    re-extracted fresh. Sharing one dict makes all four wrappers cooperate
    on a single cache.
    """

    def __init__(self, cache_file: str = "/tmp/locomo_llm_cache.json") -> None:
        self._cache_file = Path(cache_file)
        self._cache: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._load_cache()

    def clear_case(self, sample_id: str) -> int:
        """Delete every cached entry belonging to one case. Returns count removed."""
        prefix = f"{sample_id}:"
        victims = [k for k in self._cache if k.startswith(prefix)]
        for k in victims:
            del self._cache[k]
        if victims:
            self._save_cache()
        return len(victims)

    def get(self, key: str) -> Any | None:
        # CPython dict lookup is atomic; reading without the lock keeps the
        # hit path fast while a concurrent put writes a different key.
        return self._cache.get(key)

    async def put(self, key: str, entry: dict[str, Any]) -> None:
        async with self._lock:
            self._cache[key] = entry
        # Persist in a background thread so dict+IO never blocks the event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_cache)

    def _load_cache(self) -> None:
        if self._cache_file.exists():
            try:
                loaded = json.loads(self._cache_file.read_text())
                if isinstance(loaded, dict):
                    kept = {k: v for k, v in loaded.items() if _is_namespaced_cache_key(k)}
                    dropped = len(loaded) - len(kept)
                    if dropped:
                        logger.info(
                            "Dropped %d legacy (pre-namespace) LLM cache entries; "
                            "%d namespaced entries loaded from %s",
                            dropped,
                            len(kept),
                            self._cache_file,
                        )
                    else:
                        logger.info(
                            "Loaded %d cached LLM/Embedding responses from %s",
                            len(kept),
                            self._cache_file,
                        )
                    self._cache = kept
            except Exception:
                logger.warning("Failed to load LLM cache, starting fresh", exc_info=True)

    def _save_cache(self) -> None:
        try:
            self._cache_file.write_text(json.dumps(self._cache, indent=2))
        except Exception:
            logger.warning("Failed to save LLM cache to disk", exc_info=True)


def _llm_cache_key(
    messages: list[Any], temperature: float, max_tokens: int | None, **kwargs: Any
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
    # Stable hash of messages+params, namespaced by the running case id so
    # each case's entries are independently deletable (--clear-case-cache).
    serialized = json.dumps(
        {
            "messages": norm_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"},
        },
        sort_keys=True,
    )
    return f"{_case_prefix()}{hashlib.md5(serialized.encode('utf-8')).hexdigest()}"


class DiskCacheWrapper:
    """Per-model LLM adapter that caches responses in a shared DiskCache.

    Holds its own inner adapter (one model) and hit/miss counters (per-model
    stats for the run-end log), but delegates all cache state to the shared
    DiskCache so concurrent wrappers no longer clobber each other's entries.
    """

    def __init__(self, inner: Any, cache: DiskCache) -> None:
        self._inner = inner
        self._cache = cache
        self._hits = 0
        self._misses = 0

    def cache_stats(self) -> tuple[int, int]:
        """Return (hits, misses) since this wrapper was created."""
        return self._hits, self._misses

    def clear_case(self, sample_id: str) -> int:
        return self._cache.clear_case(sample_id)

    async def chat(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        key = _llm_cache_key(messages, temperature, max_tokens, tools=tools, **kwargs)
        entry = self._cache.get(key)
        if entry is not None and "content" in entry:
            self._hits += 1

            @dataclass
            class FakeResponse:
                content: str

            return FakeResponse(content=entry["content"])

        self._misses += 1
        # Timeout backstop: the openai SDK per-request timeout does not always
        # fire for streaming responses, so this keeps a full run from hanging
        # on one stuck call (case scored wrong, bench moves on). The clock
        # starts when the coroutine is awaited, so under high --concurrency the
        # asyncio scheduling + connection-pool queue wait counts against it. At
        # 60s this silently killed every batch extraction call (large, slow,
        # queued behind ~concurrency*DRAIN_WORKERS peers) -> empty response ->
        # whole-batch fallback to context-free single-turn extraction, which
        # loses cross-turn coreference. 180s leaves room for the queued call to
        # actually issue; the real HTTP body completes well under the SDK's own
        # 60s once it starts.
        try:
            response = await asyncio.wait_for(
                self._inner.chat(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
                timeout=180.0,
            )
        except TimeoutError:
            logger.warning("LLM call timed out after 180s, treating as empty response")

            class _Empty:
                content = ""

            return _Empty()
        content = getattr(response, "content", None)
        if isinstance(content, str):
            await self._cache.put(key, {"content": content})
        return response

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Cache the whole batch, namespaced by case id for per-case deletion.
        key = f"{_case_prefix()}{hashlib.md5(json.dumps(texts).encode('utf-8')).hexdigest()}"
        entry = self._cache.get(key)
        if entry is not None and "embeddings" in entry:
            return entry["embeddings"]

        embeddings = await self._inner.embed(texts)
        await self._cache.put(key, {"embeddings": embeddings})
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
    debug_trace: bool = False,
    clear_case_cache: list[str] | None = None,
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

    # One shared cache across all per-model wrappers so they cooperate on a
    # single in-memory dict + file instead of clobbering each other on save.
    shared_llm_cache = DiskCache()

    def _make_llm(model: str) -> DiskCacheWrapper:
        return DiskCacheWrapper(
            LLMAdapterFactory.create(
                llm_provider,
                model=model,
                api_key=resolved_key,
                base_url=base_url,
            ),
            shared_llm_cache,
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
    # The extraction prompt AND the batch response_format schema are hashed:
    # either alters what facts get extracted, so the pre-extracted session DB
    # must rebuild. Without this, a prompt or schema fix would be silently
    # masked by stale cached extractions. The schema is hashed at n_turns=1
    # since its structure (not the dynamic minItems) governs extraction shape.
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
            "extract_schema_hash": hashlib.md5(
                json.dumps(_atomic_fact_batch_response_format(1), sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        sort_keys=True,
    )
    config_hash = hashlib.md5(extract_config_payload.encode("utf-8")).hexdigest()[:24]
    db_cache_dir = Path("/tmp/locomo_session_db_cache")
    db_cache_dir.mkdir(parents=True, exist_ok=True)

    if clear_case_cache:
        for sid in clear_case_cache:
            # All wrappers share one DiskCache, so clearing the shared cache
            # once covers extract/answer/judge/reflect in a single pass.
            cleared_entries = shared_llm_cache.clear_case(sid)
            db_files = sorted(db_cache_dir.glob(f"{sid}_*.db"))
            for dbf in db_files:
                dbf.unlink(missing_ok=True)
            logger.info(
                "[clear-case-cache] %s: cleared %d LLM/embedding cache entries, deleted %d DB file(s)",
                sid,
                cleared_entries,
                len(db_files),
            )

    async def _run_one(idx: int, case: LoCoMoCase) -> None:
        async with semaphore:
            # Namespace all LLM/embedding cache keys for this case so they are
            # independently deletable via --clear-case-cache. Task-scoped: each
            # _run_one task runs on its own context copy, so concurrency is safe.
            _CURRENT_CASE_ID.set(case.sample_id)
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
                    shared_llm_cache,
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
                    debug_trace=debug_trace,
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

    # Cache hit/miss observability: LLM cache is per-run shared, so only a
    # run aggregate is meaningful. high hits = LLM responses reused from disk
    # (no real re-extract/answer); high misses = fresh LLM calls. Per-case
    # freeze status is in the inline DB-cache hit/rebuild + TurnCache lines.
    for _name, _wrapper in (
        ("extract", llm_extract),
        ("answer", llm_answer),
        ("judge", llm_judge),
        ("reflect", llm_reflect),
    ):
        if _wrapper is None:
            continue
        _h, _m = _wrapper.cache_stats()
        _rate = (_h / (_h + _m) * 100) if (_h + _m) else 0.0
        logger.info("[cache] llm %s hits=%d misses=%d hit_rate=%.0f%%", _name, _h, _m, _rate)

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
        "provenance": _provenance(),
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
    # Fail-loud on silent rerank degradation. The default reranker chain
    # leads with CrossEncoderReranker; if it failed to load (missing
    # sentence_transformers, bare python3 env) the whole run silently ranks
    # on the heuristic and every quality number is a heuristic number, not a
    # cross-encoder number. That must never pass unmarked.
    prov = report.get("provenance", {})
    if prov.get("cross_encoder_loads") is not True:
        logger.warning(
            "RERANK DEGRADED: cross_encoder_loads=%s -- this run did NOT use "
            "the cross-encoder; quality numbers are heuristic-rerank numbers. "
            "Re-run with `uv run python` so sentence_transformers resolves.",
            prov.get("cross_encoder_loads"),
        )
    degraded_tiers = []
    for r in results:
        if not isinstance(r, dict):
            continue
        tier = r.get("trace", {}).get("rerank", {}).get("tier")
        if tier and tier != "CrossEncoderReranker":
            degraded_tiers.append((r.get("case_id"), tier))
    if degraded_tiers:
        logger.warning(
            "RERANK TIER FALLBACK on %d/%d cases: %s",
            len(degraded_tiers),
            total,
            degraded_tiers[:5],
        )
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
            debug_trace=args.debug_trace,
            clear_case_cache=args.clear_case_cache,
        )
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
