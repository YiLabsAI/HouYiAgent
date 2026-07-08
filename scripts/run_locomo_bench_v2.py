"""LoCoMo benchmark v2 — streaming conversation mode.

v1 (run_locomo_bench.py) ingests per QUESTION: each case rebuilds a
DB, optionally windowed to only the turns around that question's gold
evidence (--use-window), which leaks oracle knowledge of where the
answer lives and never exercises the SDK the way a real caller does.

v2 mirrors production SDK usage instead:

    per conversation (once):
        for turn in conversation.turns (chronological):
            engine.write_turn(turn)          # real-time extract + store
        per session boundary:
            engine.evolve(consolidate=True, reflect=False)   # cheap, zero-LLM
        engine.flush()                        # drain extract + embedding queues
    per question (many, against the same shared conversation store):
        engine.answer(question)
    once, at the end of the conversation (only for questions that failed):
        engine.evolve(consolidate=False, reflect=True, failing_queries=[...])
        re-answer the failed questions

No evidence window: the full conversation is ingested, same as a real
caller who does not know in advance what will be asked. Turns are
written in time order (not concurrently) because supersede/consolidate
depends on write order. consolidate is interleaved per session because
it is deterministic and free; reflect runs once at the end because it
is LLM-driven and failure-anchored (it needs to see which questions
actually failed before it has anything to reflect on).

Usage:
  uv run python scripts/run_locomo_bench_v2.py --case conv-50
  uv run python scripts/run_locomo_bench_v2.py --sample 30 --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Script's own directory is prepended to sys.path by the interpreter when
# invoked as python scripts/run_locomo_bench_v2.py, so this sibling import
# resolves without extra path surgery. v2 deliberately reuses v1's cache,
# scoring, and provenance plumbing instead of duplicating it -- only the
# ingestion SHAPE differs between the two scripts, not the scoring math.
from run_locomo_bench import (
    _CURRENT_CASE_ID,
    _ENV_API_KEY,
    _PROVIDER_MODEL_DEFAULTS,
    EXTRACT_BATCH_SIZE,
    RECALL_TOP_K,
    DiskCache,
    DiskCacheWrapper,
    _ablated_recall_config,
    _build_extract_text,
    _CountingBatchExtractor,
    _judge,
    _normalize_observation_date,
    _percentile,
    _provenance,
    _recall_metrics,
    _recalls_to_rows,
)

from houyi.adapters.embedding import (
    DashScopeEmbeddingProvider,
    EmbeddingProvider,
    SiliconFlowEmbeddingProvider,
    make_embedding_provider,
)
from houyi.adapters.llm.factory import LLMAdapterFactory
from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.bench.locomo import (
    LoCoMoCase,
    LoCoMoSample,
    load_locomo_all,
    load_locomo_balanced,
)
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.entity_resolver import RoleBasedEntityResolver, TurnContext
from houyi.adapters.memory.extractor import AtomicFactExtractor
from houyi.adapters.memory.recall.factory import _build_default_recall_orchestrator
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.triggers import all_of
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import RawTurn, SessionContext
from houyi.adapters.memory.workers.embedding_backfill import (
    EmbeddingBackfillConfig,
    EmbeddingBackfillWorker,
)
from houyi.adapters.memory.workers.extractor_worker import ExtractorWorker, ExtractorWorkerConfig
from houyi.infrastructure.config.env_config import EnvConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo memory benchmark v2 (streaming conversation)")
    p.add_argument("--sample", type=int, default=30)
    p.add_argument(
        "--case",
        nargs="+",
        default=None,
        metavar="CONV_ID",
        help="run only specific conversation IDs, e.g. --case conv-44 conv-50",
    )
    p.add_argument(
        "--output",
        default=None,
        help="output JSON path; defaults to benchmark/output/memory/locomo-v2-<timestamp>.json",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of CONVERSATIONS to ingest/answer in parallel (default: 1 = serial)",
    )
    p.add_argument(
        "--llm-provider",
        default="siliconflow",
        choices=tuple(_PROVIDER_MODEL_DEFAULTS),
    )
    p.add_argument("--extract-model", default=None)
    p.add_argument("--answer-model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--evolve-model", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument(
        "--embedding-provider",
        default="siliconflow",
        choices=("siliconflow", "dashscope", "local", "noop"),
    )
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument(
        "--reflect",
        action="store_true",
        default=False,
        help="Run the end-of-conversation failure-anchored reflect pass (default: off).",
    )
    p.add_argument(
        "--consolidate",
        action="store_true",
        default=False,
        help="Run the per-session interleaved consolidate pass (default: off).",
    )
    p.add_argument(
        "--debug-trace",
        action="store_true",
        default=False,
        help="Capture per-stage recall snapshots into each result (see v1 for details).",
    )
    p.add_argument(
        "--extract-concurrency",
        type=int,
        default=3,
        help=(
            "number of concurrent ExtractorWorker.run_forever loops draining "
            "the L1 queue per conversation (default: 3). N>1 trades a bounded "
            "claim/write-order guarantee for higher drain throughput under the "
            "bulk-ingest backlog the bench creates."
        ),
    )
    p.add_argument(
        "--extract-batch-size",
        type=int,
        default=None,
        help="Override the L1 extract batch size (default: EXTRACT_BATCH_SIZE=16). Smaller batches reduce batch attention loss on some models.",
    )
    p.add_argument(
        "--answer-concurrency",
        type=int,
        default=5,
        help="number of questions to answer+judge in parallel per conversation (default: 5). Incompatible with --debug-trace.",
    )
    p.add_argument(
        "--fresh-extract",
        action="store_true",
        default=False,
        help=(
            "Archive the persistent conv DB and re-extract from scratch. The "
            "conv DB is keyed by sample_id so a prior run extraction is reused "
            "by default; pass this when extractor or ingest code changed (a "
            "stale done extraction would otherwise be reused silently)."
        ),
    )
    return p.parse_args()


def _group_by_conversation(cases: list[LoCoMoCase]) -> list[tuple[LoCoMoSample, list[LoCoMoCase]]]:
    """Group cases by their parent conversation, preserving first-seen order.

    Mirrors LoCoMoCase's own docstring intent ("lets the harness ingest once
    per sample and answer multiple cases against the same store") which v1
    never actually exploits -- every case rebuilds its own DB there.
    """
    order: list[str] = []
    groups: dict[str, list[LoCoMoCase]] = {}
    samples: dict[str, LoCoMoSample] = {}
    for case in cases:
        if case.sample_id not in groups:
            order.append(case.sample_id)
            groups[case.sample_id] = []
            samples[case.sample_id] = case.sample
        groups[case.sample_id].append(case)
    return [(samples[sid], groups[sid]) for sid in order]


def _archive_db(db: Path, sample_id: str) -> None:
    """Copy a conv DB to the archive dir before it is rebuilt or deleted.

    Honors the archive-before-clear discipline: keep the old extraction so a
    diff/repro/root-cause is still possible after the bench rebuilds the DB.
    No-op when the DB does not exist.
    """
    if not db.exists():
        return
    archive_dir = Path("/tmp/locomo_v2_archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    dest = archive_dir / f"{sample_id}_{stamp}.db"
    shutil.copy2(db, dest)
    logger.info("archived conv DB %s -> %s", db, dest)


def _make_embedding_provider(
    *,
    embedding_provider: str,
    embedding_model: str | None,
    embedding_api_key: str | None,
    shared_cache: DiskCache,
) -> EmbeddingProvider:
    env = EnvConfig.get()
    if embedding_provider == "siliconflow":
        key = embedding_api_key or env.siliconflow_api_key or ""
        return DiskCacheWrapper(
            SiliconFlowEmbeddingProvider(api_key=key, model=embedding_model), shared_cache
        )
    if embedding_provider == "dashscope":
        key = embedding_api_key or env.dashscope_api_key or ""
        return DiskCacheWrapper(
            DashScopeEmbeddingProvider(api_key=key, model=embedding_model), shared_cache
        )
    return DiskCacheWrapper(
        make_embedding_provider(
            provider=embedding_provider, model=embedding_model, api_key=embedding_api_key
        ),
        shared_cache,
    )


async def _run_conversation(
    sample: LoCoMoSample,
    cases: list[LoCoMoCase],
    *,
    llm_extract: Any,
    llm_answer: Any,
    llm_judge: Any,
    llm_reflect: Any,
    shared_cache: DiskCache,
    embedding_provider: str,
    embedding_model: str | None,
    embedding_api_key: str | None,
    consolidate_enabled: bool,
    reflect_enabled: bool,
    debug_trace: bool,
    extract_concurrency: int = 1,
    extract_batch_size: int | None = None,
    answer_concurrency: int = 5,
    fresh_extract: bool = False,
) -> dict[str, Any]:
    """Ingest one conversation once, then answer every question against it."""
    _CURRENT_CASE_ID.set(sample.sample_id)
    t_ingest0 = time.perf_counter()

    db = Path(f"/tmp/locomo_v2_{sample.sample_id}.db")
    namespace = f"locomo:{sample.sample_id}"
    expected_turns = len(sample.turns)

    # The conv DB is keyed by sample_id so a prior run's extraction is
    # reusable: the engine already skips re-extraction for turns whose
    # extract_queue row is state=done (enqueue_extract is turn_id-idempotent
    # and claim_extract_jobs only claims pending rows). Re-running write_turn
    # for already-persisted turn_ids would hit UNIQUE(turn_id), so a fully
    # ingested DB skips the write loop; a partial DB is archived and rebuilt
    # (resuming mid-ingest would collide on turn_id). --fresh-extract forces
    # the archive+rebuild path for when extractor or ingest code changed.
    reuse_ingest = False
    if fresh_extract:
        _archive_db(db, sample.sample_id)
        db.unlink(missing_ok=True)
    elif db.exists():
        probe = SQLiteMemoryBackend(db_path=db)
        try:
            actual = probe.count_raw_turns(namespace, sample.sample_id)
        finally:
            probe.close()
        if actual == expected_turns:
            reuse_ingest = True
            logger.info(
                "reuse %s: %d turns already ingested, skipping write+extract",
                sample.sample_id,
                expected_turns,
            )
        elif actual > 0:
            logger.info(
                "partial ingest for %s (%d/%d turns): archive + rebuild",
                sample.sample_id,
                actual,
                expected_turns,
            )
            _archive_db(db, sample.sample_id)
            db.unlink(missing_ok=True)

    backend = SQLiteMemoryBackend(db_path=db)
    result: dict[str, Any] = {"sample_id": sample.sample_id, "cases": []}
    try:
        inbox = SQLiteCandidateInbox(backend)
        view = SQLiteEntityStateView(backend)
        extractor = AtomicFactExtractor(llm_extract, max_retries=1, batch_max_tokens=8192)
        counting_extractor = _CountingBatchExtractor(extractor)
        turn_writer = TurnWriter(backend, extract_trigger=all_of())
        extractor_worker = ExtractorWorker(
            backend=backend,
            extractor=counting_extractor,
            entity_state=view,
            candidate_inbox=inbox,
            event_view=backend,
            config=ExtractorWorkerConfig(batch_size=extract_batch_size or EXTRACT_BATCH_SIZE),
            write_lock=asyncio.Lock(),
        )
        embedding_prov = _make_embedding_provider(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            shared_cache=shared_cache,
        )
        backfill_worker = EmbeddingBackfillWorker(
            backend=backend,
            provider=embedding_prov,
            config=EmbeddingBackfillConfig(batch_size=16),
        )
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
            turn_writer=turn_writer,
            extractor_worker=extractor_worker,
            backfill_worker=backfill_worker,
            debug_trace=debug_trace,
        )
        await engine.start(extractor_worker_concurrency=extract_concurrency)

        resolver = RoleBasedEntityResolver(primary=sample.speaker_a, secondary=sample.speaker_b)

        # --- Ingest phase: real-time, in chronological order -------------
        # Skipped entirely when reuse_ingest: the conv DB already holds every
        # turn (count_raw_turns == expected) and the engine skips re-extraction
        # for done turns, so write_turn would only collide on UNIQUE(turn_id).
        # The flush below still drains any extraction a prior run left pending.
        if not reuse_ingest:
            current_session: str | None = None
            for i, t in enumerate(sample.turns):
                if (
                    consolidate_enabled
                    and current_session is not None
                    and t.session_id != current_session
                ):
                    # Session boundary: cheap deterministic consolidate so
                    # supersede resolves conflicts as the conversation
                    # accumulates, not only once at the very end.
                    await asyncio.to_thread(
                        engine.evolve, consolidate=True, reflect=False, namespace=namespace
                    )
                current_session = t.session_id
                turn_ctx = TurnContext(
                    text=t.text, speaker_id=t.speaker, session_id=t.session_id, turn_id=t.dia_id
                )
                entity_id = resolver.resolve(turn_ctx)
                role = "user" if t.speaker == sample.speaker_a else "assistant"
                extract_text = _build_extract_text(
                    text=t.text, speaker_name=entity_id, observation_date=t.session_datetime
                )
                turn = RawTurn(
                    turn_id=f"{sample.sample_id}:{t.session_id}:{t.dia_id}:{i}",
                    namespace=namespace,
                    session_id=sample.sample_id,
                    role=role,
                    content=t.text,
                    metadata={
                        "source_anchor": f"{sample.sample_id}:{t.dia_id}",
                        "speaker": t.speaker,
                        "extract_text": extract_text,
                        "turn_marker": f"<<TURN id={t.dia_id}>>",
                    },
                )
                await engine.write_turn(turn)

        # Drain the extraction + embedding-backfill queues before answering:
        # a real caller's write_turn calls are also async-extracted in the
        # background, but no answer should be scored against a half-drained
        # queue. On reuse_ingest this is near-instant when extraction already
        # finished; it still drains any work a prior run left pending.
        await engine.flush(timeout=1800.0)
        if consolidate_enabled:
            await asyncio.to_thread(
                engine.evolve, consolidate=True, reflect=False, namespace=namespace
            )

        ingest_ms = (time.perf_counter() - t_ingest0) * 1000.0
        turns_ingested = len(sample.turns)

        last_turn = sample.turns[-1] if sample.turns else None
        obs_date = _normalize_observation_date(last_turn.session_datetime) if last_turn else None
        if not obs_date:
            obs_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        sys_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

        async def _answer_one(case: LoCoMoCase) -> dict[str, Any]:
            t0 = time.perf_counter()
            answer = await engine.answer(
                case.question,
                session_context=SessionContext(
                    session_id=namespace,
                    current_observation_date=obs_date,
                    current_system_date=sys_date,
                ),
                top_k=RECALL_TOP_K,
            )
            rows = _recalls_to_rows(engine, answer.extras.get("recalls", []))
            recall_at_10, ndcg_at_10, precision_at_10 = _recall_metrics(
                rows, case.evidence, top_k=RECALL_TOP_K
            )
            verdict = await _judge(llm_judge, case, answer)
            return {
                "case_id": f"{case.sample_id}:{case.question}",
                "category": case.category,
                "question": case.question,
                "answer": answer.answer,
                "expected": case.answer,
                "correct": bool(verdict["correct"]),
                "reason": verdict["reason"],
                "recall_at_10": recall_at_10,
                "ndcg_at_10": ndcg_at_10,
                "precision_at_10": precision_at_10,
                "retrieve_ms": float(answer.extras.get("recall_ms", 0.0))
                or (time.perf_counter() - t0) * 1000.0,
                "turns_ingested": turns_ingested,
                "ingest_ms": round(ingest_ms, 1),
            }

        # --- Answer phase: many questions, one shared conversation store -
        # Concurrent answer+judge with a bounded semaphore: engine.answer
        # shares the conv DB (read-safe) but engine._last_recall_trace is
        # engine-level mutable state, so --debug-trace is rejected when
        # answer-concurrency > 1 (see main). Semaphore starts small because
        # answer+judge hit the same dashscope key as extract and can trip
        # transient throttling.
        sem = asyncio.Semaphore(answer_concurrency)

        async def _answer_with_sem(case: LoCoMoCase) -> dict[str, Any]:
            async with sem:
                try:
                    return await _answer_one(case)
                except Exception as e:
                    logger.exception("answer failed for %s", case.question[:60])
                    return {
                        "case_id": f"{case.sample_id}:{case.question}",
                        "category": case.category,
                        "question": case.question,
                        "answer": "",
                        "expected": case.answer,
                        "correct": False,
                        "reason": f"error: {e}"[:120],
                        "recall_at_10": 0.0,
                        "ndcg_at_10": 0.0,
                        "precision_at_10": 0.0,
                        "retrieve_ms": 0.0,
                        "turns_ingested": turns_ingested,
                        "ingest_ms": round(ingest_ms, 1),
                    }

        case_results = list(await asyncio.gather(*[_answer_with_sem(c) for c in cases]))

        # --- Evolve phase: failure-anchored reflect, once, on real fails -
        failing = [(c, r) for c, r in zip(cases, case_results, strict=True) if not r["correct"]]
        evolve_summary: dict[str, Any] | None = None
        if reflect_enabled and failing:
            failing_queries = [c.question for c, _ in failing]
            report = await asyncio.to_thread(
                engine.evolve,
                consolidate=False,
                reflect=True,
                failing_queries=failing_queries,
                namespace=namespace,
                llm=llm_reflect,
            )
            await engine.flush(timeout=120.0)
            reflection = report.reflection
            gained = 0
            for case, old_result in failing:
                new_result = await _answer_one(case)
                idx_in_list = case_results.index(old_result)
                new_result["before_correct"] = old_result["correct"]
                new_result["before_answer"] = old_result["answer"]
                case_results[idx_in_list] = new_result
                if new_result["correct"] and not old_result["correct"]:
                    gained += 1
            evolve_summary = {
                "failing_before": len(failing),
                "facts_extracted": reflection.facts_extracted if reflection else 0,
                "facts_kept": reflection.facts_kept if reflection else 0,
                "facts_retracted": reflection.facts_retracted if reflection else 0,
                "gained": gained,
            }

        result["cases"] = case_results
        result["evolve"] = evolve_summary
        result["turns_ingested"] = turns_ingested
        result["ingest_ms"] = round(ingest_ms, 1)
        await engine.stop()
    finally:
        with __import__("contextlib").suppress(Exception):
            backend.close()
    return result


async def _run_all(
    groups: list[tuple[LoCoMoSample, list[LoCoMoCase]]],
    *,
    output_path: Path | None,
    concurrency: int,
    extract_model: str,
    answer_model: str,
    judge_model: str,
    evolve_model: str,
    api_key: str | None,
    base_url: str | None,
    llm_provider: str,
    embedding_provider: str,
    embedding_model: str | None,
    embedding_api_key: str | None,
    consolidate_enabled: bool,
    reflect_enabled: bool,
    debug_trace: bool,
    extract_concurrency: int = 1,
    extract_batch_size: int | None = None,
    answer_concurrency: int = 5,
    fresh_extract: bool = False,
) -> dict[str, Any]:
    if llm_provider == "siliconflow":
        resolved_key = api_key or _ENV_API_KEY
        if not resolved_key:
            sys.exit("No API key: pass --api-key or set SILICONFLOW_API_KEY")
    else:
        resolved_key = api_key

    shared_llm_cache = DiskCache()

    def _make_llm(model: str) -> DiskCacheWrapper:
        return DiskCacheWrapper(
            LLMAdapterFactory.create(
                llm_provider, model=model, api_key=resolved_key, base_url=base_url
            ),
            shared_llm_cache,
        )

    llm_extract = _make_llm(extract_model)
    llm_answer = _make_llm(answer_model)
    llm_judge = _make_llm(judge_model)
    llm_reflect = _make_llm(evolve_model) if reflect_enabled else None

    logger.info(
        "v2 streaming bench: provider=%s models: extract=%s answer=%s judge=%s evolve=%s "
        "conversations=%d concurrency=%d consolidate=%s reflect=%s",
        llm_provider,
        extract_model,
        answer_model,
        judge_model,
        evolve_model if reflect_enabled else "-",
        len(groups),
        concurrency,
        consolidate_enabled,
        reflect_enabled,
    )

    semaphore = asyncio.Semaphore(concurrency)
    conv_results: list[dict[str, Any] | None] = [None] * len(groups)

    async def _run_one(i: int, sample: LoCoMoSample, cases: list[LoCoMoCase]) -> None:
        async with semaphore:
            try:
                r = await _run_conversation(
                    sample,
                    cases,
                    llm_extract=llm_extract,
                    llm_answer=llm_answer,
                    llm_judge=llm_judge,
                    llm_reflect=llm_reflect,
                    shared_cache=shared_llm_cache,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                    embedding_api_key=embedding_api_key,
                    consolidate_enabled=consolidate_enabled,
                    reflect_enabled=reflect_enabled,
                    debug_trace=debug_trace,
                    extract_concurrency=extract_concurrency,
                    extract_batch_size=extract_batch_size,
                    answer_concurrency=answer_concurrency,
                    fresh_extract=fresh_extract,
                )
                conv_results[i] = r
                n_correct = sum(1 for c in r["cases"] if c["correct"])
                logger.info(
                    "[%d/%d] %s | %d/%d correct | ingest=%.1fs",
                    i + 1,
                    len(groups),
                    sample.sample_id,
                    n_correct,
                    len(r["cases"]),
                    r["ingest_ms"] / 1000.0,
                )
            except Exception as e:
                logger.exception("[%d/%d] ERROR on %s", i + 1, len(groups), sample.sample_id)
                conv_results[i] = {
                    "sample_id": sample.sample_id,
                    "cases": [
                        {
                            "case_id": f"{c.sample_id}:{c.question}",
                            "category": c.category,
                            "correct": False,
                            "reason": f"error: {e}"[:120],
                            "recall_at_10": 0.0,
                            "ndcg_at_10": 0.0,
                            "precision_at_10": 0.0,
                            "retrieve_ms": 0.0,
                        }
                        for c in cases
                    ],
                    "evolve": None,
                }

    await asyncio.gather(*[_run_one(i, s, c) for i, (s, c) in enumerate(groups)])

    all_case_results: list[dict[str, Any]] = []
    for r in conv_results:
        if r:
            all_case_results.extend(r["cases"])

    correct = sum(1 for c in all_case_results if c["correct"])
    total = len(all_case_results)
    by_cat: dict[Any, dict[str, int]] = {}
    for c in all_case_results:
        cat = c.get("category", 0)
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if c["correct"]:
            by_cat[cat]["correct"] += 1

    retrieve_samples = [float(c.get("retrieve_ms", 0.0)) for c in all_case_results]
    recall_values = [float(c.get("recall_at_10", 0.0)) for c in all_case_results]
    ndcg_values = [float(c.get("ndcg_at_10", 0.0)) for c in all_case_results]
    precision_values = [float(c.get("precision_at_10", 0.0)) for c in all_case_results]

    evolve_summaries = [r["evolve"] for r in conv_results if r and r.get("evolve")]

    report = {
        "recall_mode": "orchestrator-streaming-v2",
        "provenance": _provenance(),
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "recall_at_10": round(sum(recall_values) / len(recall_values), 4) if recall_values else 0.0,
        "ndcg_at_10": round(sum(ndcg_values) / len(ndcg_values), 4) if ndcg_values else 0.0,
        "precision_at_10": round(sum(precision_values) / len(precision_values), 4)
        if precision_values
        else 0.0,
        "retrieve_p50_ms": round(_percentile(retrieve_samples, 0.5), 2),
        "retrieve_p95_ms": round(_percentile(retrieve_samples, 0.95), 2),
        "by_category": {
            str(k): {**v, "accuracy": round(v["correct"] / v["total"], 4)}
            for k, v in sorted(by_cat.items())
        },
        "evolve_summary": {
            "conversations_with_reflect": len(evolve_summaries),
            "gained": sum(e["gained"] for e in evolve_summaries),
        }
        if evolve_summaries
        else None,
        "conversations": conv_results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("wrote %s", output_path)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = _parse_args()

    if args.debug_trace and args.answer_concurrency > 1:
        sys.exit(
            "--debug-trace is incompatible with --answer-concurrency>1: engine._last_recall_trace is engine-level shared mutable state and would corrupt across concurrent answers."
        )

    defaults = _PROVIDER_MODEL_DEFAULTS[args.llm_provider]
    extract_model = args.extract_model or defaults["extract"]
    answer_model = args.answer_model or defaults["answer"]
    judge_model = args.judge_model or defaults["judge"]
    evolve_model = args.evolve_model or defaults["evolve"]

    if args.case:
        wanted = set(args.case)
        all_cases = [c for c in load_locomo_all() if c.sample_id in wanted]
    else:
        all_cases = load_locomo_balanced(n=args.sample)
    if not all_cases:
        sys.exit(f"No cases matched --case {args.case}" if args.case else "No cases")

    groups = _group_by_conversation(all_cases)

    output_path = (
        Path(args.output)
        if args.output
        else Path("benchmark/output/memory")
        / f"locomo-v2-{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d-%H%M%S')}.json"
    )

    report = asyncio.run(
        _run_all(
            groups,
            output_path=output_path,
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
            consolidate_enabled=args.consolidate,
            reflect_enabled=args.reflect,
            debug_trace=args.debug_trace,
            extract_concurrency=args.extract_concurrency,
            extract_batch_size=args.extract_batch_size,
            answer_concurrency=args.answer_concurrency,
            fresh_extract=args.fresh_extract,
        )
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("conversations",)},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
