"""Benchmark runner for DeepResearch-Bench evaluation.

Loads queries from a JSONL file (DeepResearch-Bench query.jsonl format),
runs each through ResearchRuntime, and writes results in the required
{id, prompt, article} JSONL format for benchmark scoring.

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
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.runtime import ResearchRuntime
from houyi.application.research.runtime.errors import ResearchReportNotReadyError
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
    search_elapsed_ms: float = 0.0
    phase_timings_ms: dict[str, float] = field(default_factory=dict)
    section_input_metrics: list[dict[str, Any]] = field(default_factory=list)
    per_question_elapsed_ms: list[dict[str, Any]] = field(default_factory=list)
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
        Default ResearchSettings applied to every query.
    concurrency:
        Max parallel research runs (default 3 to respect API limits).
    run_timeout:
        Per-query timeout in seconds.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        web_search: WebSearchService,
        settings: ResearchSettings | None = None,
        concurrency: int = 3,
        run_timeout: float = 600,
        **llm_kwargs: Any,
    ) -> None:
        self._llm = llm_adapter
        self._ws = web_search
        self._settings = settings or ResearchSettings(depth="deep")
        self._concurrency = concurrency
        self._timeout = run_timeout
        self._llm_kwargs = llm_kwargs

    async def run(
        self,
        query_path: str | Path,
        output_path: str | Path,
        resume: bool = True,
    ) -> BenchmarkSummary:
        """Run the full benchmark suite.

        Args:
            query_path: Path to query.jsonl (DeepResearch-Bench format).
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
        metrics_path = _metrics_output_path(out_path)

        async def _run_one(query: BenchmarkQuery) -> BenchmarkResult:
            async with sem:
                return await self._execute_query(query)

        tasks = [asyncio.create_task(_run_one(q)) for q in pending]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            _append_result(out_path, result)
            _append_metrics(metrics_path, result)
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

        The runtime itself enforces a dynamic timeout whose budget varies by
        orchestration mode (DIRECT 180s/q, DELEGATE 300s/q, AUTONOMOUS parallel).
        The benchmark runner adds a 60s buffer on top for plan generation + overhead.
        """
        start = time.monotonic()
        runtime: ResearchRuntime | None = None
        try:
            runtime = ResearchRuntime(
                llm_adapter=self._llm,
                web_search=self._ws,
                settings=self._settings,
                **self._llm_kwargs,
            )
            await runtime.start(query.prompt)
            await runtime.confirm_plan()
            outer_timeout = runtime._runtime_timeout() + 60
            await asyncio.wait_for(runtime.execute(), timeout=outer_timeout)

            try:
                report = await runtime.get_report()
                article = _report_to_article(report)
            except ResearchReportNotReadyError:
                article = ""
                logger.warning(
                    "Query %s: execute() completed but report not ready (internal timeout)",
                    query.id,
                )

            duration = time.monotonic() - start
            score = runtime.quality_score
            overall = score.overall if hasattr(score, "overall") else float(score or 0)
            # Merge search-phase decomposition timings into phase_timings_ms
            phase_timings = dict(runtime.phase_timings_ms)
            phase_timings["aggregate_ms"] = runtime.aggregate_ms
            phase_timings["intermediate_ms"] = runtime.intermediate_ms
            section_input_metrics = []
            if article:
                report = await runtime.get_report()
                section_input_metrics = list(report.metadata.section_input_metrics or [])
            return BenchmarkResult(
                id=query.id,
                prompt=query.prompt,
                article=article,
                quality_score=overall,
                duration_seconds=round(duration, 2),
                search_elapsed_ms=runtime.search_elapsed_ms,
                phase_timings_ms=phase_timings,
                section_input_metrics=section_input_metrics,
                per_question_elapsed_ms=runtime.per_question_elapsed_ms,
                error="internal timeout (report incomplete)" if not article else None,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            logger.error("Query %s failed: %s", query.id, exc, exc_info=True)
            phase_timings = {}
            # Keep partial timings for failed runs so benchmark attribution survives exceptions.
            if runtime is not None:
                phase_timings = dict(runtime.phase_timings_ms)
                phase_timings["aggregate_ms"] = runtime.aggregate_ms
                phase_timings["intermediate_ms"] = runtime.intermediate_ms
            return BenchmarkResult(
                id=query.id,
                prompt=query.prompt,
                article="",
                duration_seconds=round(duration, 2),
                search_elapsed_ms=runtime.search_elapsed_ms if runtime is not None else 0.0,
                phase_timings_ms=phase_timings,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Report → article conversion
# ---------------------------------------------------------------------------


# Matches inline markdown links whose anchor text is non-empty and whose
# URL is http(s). Used by _renumber_citations to convert verbose
# [title](url) citations into short [N] numbered form.
#
# The anchor permits **one level of nested brackets** so that LLM-produced
# citations like [[PDF] Report title](https://...) or
# [Paper [v2]](https://...) are matched and normalised. Without this
# allowance the earlier regex stopped at the first ], leaving the
# verbose anchor leaking into the body (observed on ZH case1: 73 such
# raw links, inflating body chars by ~34%; on EN5 qid=52: 24 raw links).
# Kept deliberately conservative — no recursive nesting — to avoid
# over-matching on prose that happens to contain bracketed phrases.
_INLINE_LINK_RE = re.compile(r"\[((?:[^\[\]\n]|\[[^\[\]\n]*\])+)\]\((https?://[^\s)]+)\)")
_REFERENCES_HEADING_RE = re.compile(r"^## References\s*$", re.MULTILINE)


def _renumber_citations(article: str) -> str:
    """Convert [title](url) inline citations into [N] numbered form.

    Scans the article body (content before any ## References heading),
    assigns sequential numbers by URL first-appearance order (repeated URLs
    reuse the same number), and replaces each inline link with [N].
    Rebuilds the References section as - [N] [title](url) entries.

    The body is otherwise preserved verbatim. If the article contains no
    inline links, it is returned unchanged.
    """
    ref_match = _REFERENCES_HEADING_RE.search(article)
    if ref_match is not None:
        body = article[: ref_match.start()].rstrip()
    else:
        body = article

    url_to_num: dict[str, int] = {}
    num_to_entry: dict[int, tuple[str, str]] = {}

    def _sub(m: re.Match[str]) -> str:
        title = m.group(1).strip()
        url = m.group(2)
        n = url_to_num.get(url)
        if n is None:
            n = len(url_to_num) + 1
            url_to_num[url] = n
            num_to_entry[n] = (title, url)
        return f"[{n}]"

    new_body = _INLINE_LINK_RE.sub(_sub, body)

    if not num_to_entry:
        return new_body.rstrip() + "\n" if new_body else new_body

    refs = ["## References", ""]
    for n in sorted(num_to_entry):
        title, url = num_to_entry[n]
        refs.append(f"- [{n}] [{title}]({url})")
    return new_body.rstrip() + "\n\n" + "\n".join(refs) + "\n"


def _report_to_article(report: Any) -> str:
    """Convert a ResearchReport into a flat Markdown article with citations.

    DeepResearch-Bench expects a single Markdown string with inline
    citations for FACT / RACE evaluation. Inline citations are emitted as
    short numbered markers [N] with a clean numbered References
    section at the end, matching academic conventions and the clean
    reading-experience requirement.
    """
    parts: list[str] = []
    parts.append(f"# {report.title}\n")
    if report.summary:
        parts.append(f"{report.summary}\n")

    ref_lookup: dict[str, tuple[str, str]] = {}
    for ref in report.references:
        ref_lookup[ref.reference_id] = (ref.title or ref.reference_id, ref.url)

    cited_urls: set[str] = set()
    for section in report.sections:
        parts.append(f"## {section.title}\n")
        content = section.content
        for cit in section.citations:
            label, url = ref_lookup.get(cit.reference_id, (cit.reference_id, ""))
            if url:
                content = content.replace(
                    f"[{cit.reference_id}]",
                    f"[{label}]({url})",
                )
                cited_urls.add(url)

        # Second pass: resolve any [ref_xxx] the LLM wrote in content but
        # omitted from the structured citations array.  Only strip refs
        # that genuinely cannot be resolved to a URL.
        def _resolve_remaining(m: re.Match) -> str:
            rid = m.group(1)
            lbl, u = ref_lookup.get(rid, (rid, ""))
            if u:
                cited_urls.add(u)
                return f"[{lbl}]({u})"
            return ""

        content = re.sub(r"\[(ref_[a-f0-9]+)\](?!\()", _resolve_remaining, content)
        parts.append(content)
        parts.append("")

    # Only include references that are actually cited inline. Retrieved
    # sources that never made it into the prose should not appear in the
    # final References list, since they add noise without evidentiary value.
    cited_refs = [ref for ref in report.references if ref.url and ref.url in cited_urls]
    if cited_refs:
        parts.append("## References\n")
        for ref in cited_refs:
            parts.append(f"- [{ref.title}]({ref.url})")

    # Renumber verbose [title](url) citations to short [N] form.
    return _renumber_citations("\n".join(parts))


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


def _append_metrics(path: Path, result: BenchmarkResult) -> None:
    """Append per-case runtime metrics to a sidecar JSONL file."""
    entry = {
        "id": result.id,
        "prompt": result.prompt,
        "duration_seconds": result.duration_seconds,
        "search_elapsed_ms": result.search_elapsed_ms,
        "quality_score": result.quality_score,
        "error": result.error,
        "phase_timings_ms": result.phase_timings_ms,
        "section_input_metrics": result.section_input_metrics,
        "per_question_elapsed_ms": result.per_question_elapsed_ms,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _metrics_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.metrics{path.suffix}")


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
