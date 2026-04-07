#!/usr/bin/env python3
"""CLI for running HouYi Deep Research against DeepResearch-Bench.

Usage::

    # Full benchmark (100 queries, deep mode)
    python scripts/run_benchmark.py \\
        --queries data/prompt_data/query.jsonl \\
        --output data/test_data/raw_data/houyi.jsonl \\
        --depth deep --concurrency 3

    # Quick smoke test (first 5 queries)
    python scripts/run_benchmark.py \\
        --queries data/prompt_data/query.jsonl \\
        --output data/test_data/raw_data/houyi-smoke.jsonl \\
        --depth standard --concurrency 2 --limit 5

Environment variables:
    OPENAI_API_KEY / GEMINI_API_KEY   — LLM provider key
    BOCHA_API_KEY                     — Web search key
    LLM_PROVIDER                     — "openai" (default), "gemini", "anthropic"
    LLM_MODEL                        — model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from houyi.application.research.types import ResearchSettings
from houyi.arena.benchmark_runner import BenchmarkRunner


def _load_env():
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def _build_llm():
    """Build an LLM adapter from environment variables."""
    from houyi.adapters.llm.factory import LLMAdapterFactory

    provider = os.getenv("LLM_PROVIDER", os.getenv("DEFAULT_LLM_PROVIDER", "siliconflow"))
    return LLMAdapterFactory.create(provider=provider)


def _build_web_search():
    """Build a web search service from environment variables."""
    from houyi.skills.web_search.service import WebSearchService

    return WebSearchService.from_env()


def _trim_queries(query_path: Path, limit: int | None) -> Path:
    """If limit is set, create a temporary JSONL with only the first N queries."""
    if limit is None:
        return query_path
    tmp_path = query_path.parent / f"_houyi_bench_subset_{limit}.jsonl"
    with query_path.open(encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break
            fout.write(line)
    return tmp_path


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _load_env()

    llm = _build_llm()
    ws = _build_web_search()
    settings_kwargs = {
        "depth": args.depth,
        "orchestration_mode": args.mode,
    }
    if args.max_agents is not None:
        settings_kwargs["max_agents"] = args.max_agents
    settings = ResearchSettings(**settings_kwargs)

    query_path = _trim_queries(Path(args.queries), args.limit)

    runner = BenchmarkRunner(
        llm_adapter=llm,
        web_search=ws,
        settings=settings,
        concurrency=args.concurrency,
        session_timeout=args.timeout,
    )

    summary = await runner.run(
        query_path=query_path,
        output_path=args.output,
        resume=not args.no_resume,
    )

    print("\n" + "=" * 60)
    print("DeepResearch-Bench Run Summary")
    print("=" * 60)
    print(f"  Total:      {summary.total}")
    print(f"  Succeeded:  {summary.succeeded}")
    print(f"  Failed:     {summary.failed}")
    print(f"  Avg time:   {summary.avg_duration:.1f}s")
    print(f"  Avg quality: {summary.avg_quality:.1f}")
    print(f"  Output:     {args.output}")
    print("=" * 60)

    if summary.failed > 0:
        print("\nFailed queries:")
        for r in summary.results:
            if r.error:
                print(f"  [{r.id}] {r.error[:100]}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run HouYi against DeepResearch-Bench")
    parser.add_argument(
        "--queries",
        default="benchmark/data/query.jsonl",
        help="Path to benchmark query.jsonl",
    )
    parser.add_argument(
        "--output",
        default="benchmark/output/houyi.jsonl",
        help="Path to write results JSONL",
    )
    parser.add_argument("--depth", default="deep", choices=["quick", "standard", "deep"])
    parser.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "delegate", "autonomous"],
        help="Orchestration mode: direct (serial SC), delegate (parallel isolated SC), autonomous (parallel SC + SharedState)",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Max sub-agents for delegate/autonomous modes. Omit to keep historical default behavior.",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600, help="Per-query timeout (seconds)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N queries")
    parser.add_argument("--no-resume", action="store_true", help="Restart from scratch")
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
