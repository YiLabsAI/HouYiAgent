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

from houyi.adapters.embedding import (
    DashScopeEmbeddingProvider,
    EmbeddingProvider,
    SiliconFlowEmbeddingProvider,
    make_embedding_provider,
)
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
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.entity_resolver import RoleBasedEntityResolver, TurnContext
from houyi.adapters.memory.extractor import AtomicFactExtractor
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


async def _run_case_with_mode(
    case: LoCoMoCase,
    turn_writer: TurnWriter,
    worker: ExtractorWorker,
    extractor_counter: _CountingBatchExtractor,
    view: SQLiteEntityStateView,
    backend: SQLiteMemoryBackend,
    namespace: str,
    llm_answer: SiliconFlowAdapter,
    llm_judge: SiliconFlowAdapter,
    *,
    embedding_provider: str = "siliconflow",
    embedding_model: str | None = None,
    embedding_api_key: str | None = None,
    base_url: str | None = None,
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
    logger.info("  extractor processed %d turns, %d LLM calls", processed, extract_calls_per_case)

    # Debug: inspect entity-state and raw-turn contents
    _ents = view.list_entities(namespace)
    logger.info("  entity-state entities: %s (count=%d)", _ents[:10], len(_ents))
    _pending_emb = len(backend.list_pending_embeddings())
    logger.info("  pending embeddings: %d", _pending_emb)

    retrieve_t0 = time.perf_counter()
    rows: list[_BenchRow]

    # Embedding provider for the vector retriever. Errors propagate so
    # bench failure modes stay loud.
    _env = EnvConfig.get()
    _emb_key: str = embedding_api_key or _env.siliconflow_api_key or ""
    if embedding_provider == "siliconflow":
        embedding_prov: EmbeddingProvider = SiliconFlowEmbeddingProvider(
            api_key=_emb_key,
            model=embedding_model,
        )
    elif embedding_provider == "dashscope":
        _ds_key: str = embedding_api_key or _env.dashscope_api_key or ""
        embedding_prov = DashScopeEmbeddingProvider(
            api_key=_ds_key,
            model=embedding_model,
        )
    else:
        embedding_prov = make_embedding_provider(
            provider=embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
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
        ),
        embedding_provider=embedding_prov,
    )
    from houyi.adapters.memory.types import SessionContext

    answer = await engine.answer(
        case.question,
        session_context=SessionContext(session_id=namespace),
        top_k=RECALL_TOP_K,
    )
    recalls = answer.extras.get("recalls", [])
    rows = []
    for r in recalls:
        record = engine._find_record(r.memory_id)
        if record:
            parts = record.key.split(".", 1)
            subj = parts[0] if len(parts) > 1 else ""
            pred = parts[1] if len(parts) > 1 else record.key
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
    retrieve_ms = (time.perf_counter() - retrieve_t0) * 1000.0

    recall_at_10, mrr = _recall_hits(rows, case.evidence, top_k=RECALL_TOP_K)
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
    embedding_provider: str = "siliconflow",
    embedding_model: str | None = None,
    embedding_api_key: str | None = None,
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
            db = Path(f"/tmp/locomo_bench_{run_token}_orchestrator_{idx}.db")
            backend = SQLiteMemoryBackend(db_path=db)
            try:
                inbox = SQLiteCandidateInbox(backend)
                view = SQLiteEntityStateView(backend)
                extractor = AtomicFactExtractor(llm_extract, max_retries=1)
                counting_extractor = _CountingBatchExtractor(extractor)
                turn_writer = TurnWriter(backend, extract_trigger=all_of())
                # The default MemoryRecordPromoter is wired by ExtractorWorker
                # itself, so no explicit promoter argument is needed.
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
                    backend,
                    f"locomo:{case.sample_id}:{idx}",
                    llm_answer,
                    llm_judge,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_api_key=embedding_api_key,
                    base_url=base_url,
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
    mrr_values = [float(r.get("mrr", 0.0)) for r in results if isinstance(r, dict)]
    report = {
        "recall_mode": "orchestrator",
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
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_api_key=args.embedding_api_key,
        )
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
