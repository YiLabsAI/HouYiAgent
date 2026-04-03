"""Benchmark runner for DeepResearch-Bench evaluation.

Loads queries from a JSONL file (DeepResearch-Bench ``query.jsonl`` format),
runs each through ``ResearchSession``, and writes results in the required
``{id, prompt, article}`` JSONL format for benchmark scoring.

Usage::

    runner = BenchmarkRunner(llm_adapter=llm, web_search=ws)
    await runner.run(
        query_path="data/prompt_data/query.jsonl",
        output_path="data/test_data/raw_data/houyi.jsonl",
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.session import ResearchSession
from houyi.application.research.types import ResearchSettings
from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuery:
    """A single query from DeepResearch-Bench."""

    id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result for a single benchmark query, ready for JSONL serialisation."""

    id: str
    prompt: str
    article: str
    quality_score: float = 0.0
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Aggregate stats across all benchmark queries."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    avg_duration: float = 0.0
    avg_quality: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)


class BenchmarkRunner:
    """Orchestrates batch Deep Research runs over benchmark queries.

    Parameters
    ----------
    llm_adapter:
        The LLM backend to use for all research stages.
    web_search:
        Web search provider for information retrieval.
    settings:
        Default ``ResearchSettings`` applied to every query.
    concurrency:
        Max parallel research sessions (default 3 to respect API limits).
    session_timeout:
        Per-query timeout in seconds.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        web_search: WebSearchService,
        settings: ResearchSettings | None = None,
        concurrency: int = 3,
        session_timeout: float = 600,
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._ws = web_search
        self._settings = settings or ResearchSettings(depth="deep")
        self._concurrency = concurrency
        self._timeout = session_timeout
        self._llm_kwargs = llm_kwargs

    async def run(
        self,
        query_path: str | Path,
        output_path: str | Path,
        resume: bool = True,
    ) -> BenchmarkSummary:
        """Run the full benchmark suite.

        Args:
            query_path: Path to ``query.jsonl`` (DeepResearch-Bench format).
            output_path: Path to write results JSONL.
            resume: If True, skip queries already present in *output_path*.
        """
        queries = _load_queries(Path(query_path))
        done_ids = _load_done_ids(Path(output_path)) if resume else set()
        pending = [q for q in queries if q.id not in done_ids]

        logger.info(
            "Benchmark: %d total, %d done, %d pending",
            len(queries),
            len(done_ids),
            len(pending),
        )

        sem = asyncio.Semaphore(self._concurrency)
        results: list[BenchmarkResult] = []
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        async def _run_one(query: BenchmarkQuery) -> BenchmarkResult:
            async with sem:
                return await self._execute_query(query)

        tasks = [asyncio.create_task(_run_one(q)) for q in pending]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            _append_result(out_path, result)
            status = "OK" if not result.error else f"FAIL: {result.error[:60]}"
            logger.info(
                "[%s] %s (%.1fs, quality=%.1f)",
                result.id,
                status,
                result.duration_seconds,
                result.quality_score,
            )

        return _build_summary(results)

    async def _execute_query(self, query: BenchmarkQuery) -> BenchmarkResult:
        """Run a single research query end-to-end.

        The session itself enforces a dynamic timeout whose budget varies by
        orchestration mode (DIRECT 180s/q, DELEGATE 300s/q, AUTONOMOUS parallel).
        The benchmark runner adds a 60s buffer on top for plan generation + overhead.
        """
        start = time.monotonic()
        try:
            session = ResearchSession(
                llm_adapter=self._llm,
                web_search=self._ws,
                settings=self._settings,
                **self._llm_kwargs,
            )
            await session.start(query.prompt)
            await session.confirm_plan()
            outer_timeout = session._session_timeout() + 60
            await asyncio.wait_for(session.execute(), timeout=outer_timeout)

            try:
                report = await session.get_report()
                article = _report_to_article(report)
            except RuntimeError:
                article = ""
                logger.warning(
                    "Query %s: execute() completed but report not ready (internal timeout)",
                    query.id,
                )

            duration = time.monotonic() - start
            score = session.quality_score
            overall = score.overall if hasattr(score, "overall") else float(score or 0)
            return BenchmarkResult(
                id=query.id,
                prompt=query.prompt,
                article=article,
                quality_score=overall,
                duration_seconds=round(duration, 2),
                error="internal timeout (report incomplete)" if not article else None,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            logger.error("Query %s failed: %s", query.id, exc, exc_info=True)
            return BenchmarkResult(
                id=query.id,
                prompt=query.prompt,
                article="",
                duration_seconds=round(duration, 2),
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Report → article conversion
# ---------------------------------------------------------------------------


def _report_to_article(report: Any) -> str:
    """Convert a ResearchReport into a flat Markdown article with citations.

    DeepResearch-Bench expects a single Markdown string with inline URL
    citations for FACT evaluation.
    """
    parts: list[str] = []
    parts.append(f"# {report.title}\n")
    if report.summary:
        parts.append(f"{report.summary}\n")

    ref_lookup: dict[str, str] = {}
    for ref in report.references:
        ref_lookup[ref.reference_id] = ref.url

    for section in report.sections:
        parts.append(f"## {section.title}\n")
        content = section.content
        for cit in section.citations:
            url = ref_lookup.get(cit.reference_id, "")
            if url:
                content = content.replace(
                    f"[{cit.reference_id}]",
                    f"[{cit.reference_id}]({url})",
                )
        parts.append(content)
        parts.append("")

    if report.references:
        parts.append("## References\n")
        for ref in report.references:
            parts.append(f"- [{ref.title}]({ref.url})")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_queries(path: Path) -> list[BenchmarkQuery]:
    """Load benchmark queries from a JSONL file."""
    queries: list[BenchmarkQuery] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            queries.append(
                BenchmarkQuery(
                    id=str(data.get("id", "")),
                    prompt=str(data.get("prompt", data.get("query", ""))),
                    metadata={k: v for k, v in data.items() if k not in ("id", "prompt", "query")},
                )
            )
    logger.info("Loaded %d benchmark queries from %s", len(queries), path)
    return queries


def _load_done_ids(path: Path) -> set[str]:
    """Load IDs of already-completed queries from the output file."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                done.add(str(data.get("id", "")))
            except json.JSONDecodeError:
                continue
    return done


def _append_result(path: Path, result: BenchmarkResult) -> None:
    """Append a single result to the output JSONL file."""
    entry = {
        "id": result.id,
        "prompt": result.prompt,
        "article": result.article,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_summary(results: list[BenchmarkResult]) -> BenchmarkSummary:
    succeeded = [r for r in results if not r.error]
    return BenchmarkSummary(
        total=len(results),
        succeeded=len(succeeded),
        failed=len(results) - len(succeeded),
        avg_duration=sum(r.duration_seconds for r in results) / max(len(results), 1),
        avg_quality=sum(r.quality_score for r in succeeded) / max(len(succeeded), 1),
        results=results,
    )
