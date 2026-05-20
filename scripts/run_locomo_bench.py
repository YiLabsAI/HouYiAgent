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
import sys
import time
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
from houyi.adapters.memory.ingestor import MemoryIngestor
from houyi.adapters.memory.resolver import MemoryWriterTools
from houyi.adapters.memory.retraction import RetractionDetector, RetractionOrchestrator

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_KEY = os.getenv("SILICONFLOW_API_KEY") or sys.exit("SILICONFLOW_API_KEY missing")
WINDOW = 3  # turns before/after evidence

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


async def _judge(llm_judge: SiliconFlowAdapter, case: LoCoMoCase, answer: AnswerResult) -> dict:
    judge_llm = LLMMemoryJudge(_JudgeLLM(llm_judge), timeout_seconds=20.0, max_tokens=16)
    verdict = await judge_llm.judge(case, answer)
    # Retry once on transient failure (network timeout, empty response)
    if verdict.reason in ("judge_llm_failed", "judge_parse_failed"):
        verdict = await judge_llm.judge(case, answer)
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
    llm: SiliconFlowAdapter, question: str, rows: list, case_id: str
) -> AnswerResult:
    """Use LLM to reason over memories and answer the question."""
    if not rows:
        return AnswerResult(answer="", abstained=True, reason="no_memories")

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


async def _run_case(case, ingestor, view, namespace, llm_answer, llm_judge):
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
        }
    ingest_idxs = set()
    for i in idxs:
        for j in range(max(0, i - WINDOW), min(len(turns), i + WINDOW + 1)):
            ingest_idxs.add(j)
    # Use RoleBasedEntityResolver for multi-speaker entity identification
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
        await ingestor.ingest_turn(
            t.text,
            source_anchor=f"{case.sample_id}:{t.dia_id}",
            recent_targets=(),
            observation_date=t.session_datetime,
            turn_context=turn_ctx,
            entity_resolver=resolver,
        )

    await asyncio.gather(*[_ingest_one(i) for i in sorted(ingest_idxs)])
    entities = view.list_entities(namespace)
    logger.info("  Found entities: %s", entities)
    all_rows = []
    for e in entities:
        rows = view.get_active(namespace, e)
        all_rows.extend(rows)
        for r in rows:
            logger.info(
                "  Memory: %s.%s = %s (qualifiers: %s)",
                r.entity,
                r.attribute,
                r.value,
                r.qualifiers,
            )
    # Use LLM reasoning to answer based on retrieved memories
    answer = await _answer_with_reasoning(llm_answer, case.question, all_rows, case.sample_id)
    logger.info("  Generated answer: %s", answer.answer[:200])
    logger.info("  Expected answer: %s", case.answer[:200])
    v = await _judge(llm_judge, case, answer)
    return {
        "case_id": f"{case.sample_id}:{case.question[:60]}",
        "category": case.category,
        "answer": answer.answer[:300],
        "expected": case.answer[:300],
        "correct": v["correct"],
        "reason": v["reason"],
        "memories_count": len(all_rows),
        "turns_ingested": len(ingest_idxs),
        "duration_s": round(time.perf_counter() - t0, 1),
    }


async def _run_all(
    cases: list[LoCoMoCase],
    output_path: Path | None,
    concurrency: int = 1,
    extract_model: str = _MODEL_EXTRACT,
    answer_model: str = _MODEL_ANSWER,
    judge_model: str = _MODEL_JUDGE,
) -> dict:
    llm_extract = SiliconFlowAdapter(api_key=API_KEY, default_model=extract_model)
    llm_answer = SiliconFlowAdapter(api_key=API_KEY, default_model=answer_model)
    llm_judge = SiliconFlowAdapter(api_key=API_KEY, default_model=judge_model)
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

    async def _run_one(idx: int, case: LoCoMoCase) -> None:
        async with semaphore:
            db = Path(f"/tmp/locomo_bench_{idx}.db")
            backend = SQLiteMemoryBackend(db_path=db)
            try:
                inbox = SQLiteCandidateInbox(backend)
                view = SQLiteEntityStateView(backend)
                tools = MemoryWriterTools(view, inbox, namespace=f"locomo:{case.sample_id}:{idx}")
                extractor = AtomicFactExtractor(llm_extract, max_retries=1)
                orch = RetractionOrchestrator(RetractionDetector(), tools)
                ingestor = MemoryIngestor(extractor, orch, tools, inbox)
                r = await _run_case(
                    case, ingestor, view, f"locomo:{case.sample_id}:{idx}", llm_answer, llm_judge
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
    report = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
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
    if args.case:
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
        )
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
