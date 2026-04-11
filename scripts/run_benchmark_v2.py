#!/usr/bin/env python3
"""Run HouYi against DeepResearch Bench II using the upstream sidecar flow.

Acknowledgement:
This script reuses HouYi's in-repo article generation and then delegates scoring
orchestration to the public DeepResearch Bench II pipeline layout published by
https://github.com/Ayanami0730/deep_research_bench.

The scoring stages intentionally follow the upstream open-source entrypoints
instead of re-implementing a private evaluator from scratch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with env_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _build_llm():
    from houyi.adapters.llm.factory import LLMAdapterFactory

    provider = os.getenv("LLM_PROVIDER", os.getenv("DEFAULT_LLM_PROVIDER", "siliconflow"))
    return LLMAdapterFactory.create(provider=provider)


def _build_web_search():
    from houyi.skills.web_search.service import WebSearchService

    return WebSearchService.from_env()


def _trim_queries(query_path: Path, limit: int | None) -> Path:
    if limit is None:
        return query_path
    tmp_path = query_path.parent / f"_houyi_bench_subset_{limit}.jsonl"
    with query_path.open(encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx >= limit:
                break
            fout.write(line)
    return tmp_path


def _resolve_raw_data_path(args: argparse.Namespace) -> Path:
    if args.raw_data_path:
        return Path(args.raw_data_path).resolve()
    output_root = Path(args.bench_output_root).resolve()
    return output_root / "raw_data" / f"{args.target_model}.jsonl"


def _resolve_summary_path(args: argparse.Namespace) -> Path:
    if args.summary_path:
        return Path(args.summary_path).resolve()
    output_root = Path(args.bench_output_root).resolve()
    return output_root / f"{args.target_model}.summary.json"


def _stale_output_artifacts(output_root: Path, target_model: str) -> list[Path]:
    candidates = [
        output_root / "raw_data" / f"{target_model}.jsonl",
        output_root / "raw_data" / f"{target_model}.metrics.jsonl",
        output_root / "cleaned_data" / f"{target_model}.jsonl",
        output_root / "race" / target_model,
        output_root / "fact" / target_model,
        output_root / f"{target_model}.summary.json",
        output_root / f"{target_model}.bench2.log",
        output_root / "prepared_query.jsonl",
        output_root / "_bench2_compat",
    ]
    return [path for path in candidates if path.exists()]


def _guard_fresh_output_root(args: argparse.Namespace) -> None:
    if args.skip_generate or not args.no_resume:
        return
    output_root = Path(args.bench_output_root).resolve()
    stale = _stale_output_artifacts(output_root, args.target_model)
    if not stale:
        return
    formatted = "\n".join(f"  - {path}" for path in stale)
    raise RuntimeError(
        "Fresh Bench II generation was requested with --no-resume, but the output root already "
        "contains benchmark artifacts. Reuse here risks stale-result contamination. "
        "Choose a new --bench-output-root or manually clean these paths first:\n"
        f"{formatted}"
    )


async def main(args: argparse.Namespace) -> None:
    from houyi.application.research.types import ResearchSettings
    from houyi.arena.bench2_runner import (
        Bench2ExecutionContext,
        Bench2GenerationSummary,
        Bench2SidecarConfig,
        Bench2SidecarRunner,
        build_bench2_summary_payload,
    )
    from houyi.arena.benchmark_runner import BenchmarkRunner, _metrics_output_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _load_env()
    _guard_fresh_output_root(args)

    query_path = _trim_queries(Path(args.queries).resolve(), args.limit)
    raw_data_path = _resolve_raw_data_path(args)
    summary_path = _resolve_summary_path(args)
    sidecar_config = Bench2SidecarConfig(
        repo_root=Path(args.bench_repo).resolve(),
        output_root=Path(args.bench_output_root).resolve(),
        query_file=query_path,
        target_model=args.target_model,
        runtime_mode=args.bench_runtime,
        max_workers=args.bench_max_workers,
        limit=args.limit,
        skip_cleaning=args.skip_cleaning,
        only_zh=args.only_zh,
        only_en=args.only_en,
        force=args.force,
        run_race=not args.skip_race,
        run_fact=not args.skip_fact,
    )
    sidecar = Bench2SidecarRunner()
    sidecar.validate_preconditions(sidecar_config)
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    generation_summary: Bench2GenerationSummary | None = None
    metrics_path: Path | None = _metrics_output_path(raw_data_path)

    if not args.skip_generate:
        llm = _build_llm()
        ws = _build_web_search()
        settings_kwargs = {
            "depth": args.depth,
            "orchestration_mode": args.mode,
            "enable_quality_evaluation": args.inline_quality,
        }
        if args.max_agents is not None:
            settings_kwargs["max_agents"] = args.max_agents
        settings = ResearchSettings(**settings_kwargs)
        runner = BenchmarkRunner(
            llm_adapter=llm,
            web_search=ws,
            settings=settings,
            concurrency=args.concurrency,
            run_timeout=args.timeout,
        )
        summary = await runner.run(
            query_path=query_path,
            output_path=raw_data_path,
            resume=not args.no_resume,
        )
        print("\n" + "=" * 60)
        print("HouYi Article Generation Summary")
        print("=" * 60)
        print(f"  Total:       {summary.total}")
        print(f"  Succeeded:   {summary.succeeded}")
        print(f"  Failed:      {summary.failed}")
        print(f"  Avg time:    {summary.avg_duration:.1f}s")
        print(f"  Avg quality: {summary.avg_quality:.1f}")
        print(f"  Raw data:    {raw_data_path}")
        print(f"  Metrics:     {_metrics_output_path(raw_data_path)}")
        print(
            "  Inline quality: "
            + (
                "enabled (legacy internal proxy)"
                if args.inline_quality
                else "disabled (sidecar-only)"
            )
        )
        print("=" * 60)
        generation_summary = Bench2GenerationSummary(
            total=summary.total,
            succeeded=summary.succeeded,
            failed=summary.failed,
            avg_duration=summary.avg_duration,
            avg_quality=summary.avg_quality,
        )
    elif not raw_data_path.exists():
        raise FileNotFoundError(
            f"--skip-generate was set but raw data file does not exist: {raw_data_path}"
        )
    elif not metrics_path.exists():
        metrics_path = None

    sidecar_summary = sidecar.run(
        sidecar_config,
        raw_data_path=raw_data_path,
    )

    print("\n" + "=" * 60)
    print("DeepResearch Bench II Sidecar Summary")
    print("=" * 60)
    print(f"  Repo:        {sidecar_summary.workspace.repo_root}")
    print(f"  Output root: {sidecar_summary.workspace.output_root}")
    print(f"  Log:         {sidecar_summary.workspace.log_path}")
    print(f"  Steps:       {', '.join(sidecar_summary.executed_steps) or 'none'}")
    if sidecar_summary.race_scores:
        print("  RACE:")
        for key, value in sidecar_summary.race_scores.items():
            print(f"    - {key}: {value:.4f}")
    if sidecar_summary.fact_scores:
        print("  FACT:")
        for key, value in sidecar_summary.fact_scores.items():
            print(f"    - {key}: {value:.4f}")
    payload = build_bench2_summary_payload(
        context=Bench2ExecutionContext(
            target_model=args.target_model,
            runtime_mode=args.bench_runtime,
            query_file=query_path,
            raw_data_path=raw_data_path,
            metrics_path=metrics_path
            if metrics_path is not None and metrics_path.exists()
            else None,
            bench_repo=Path(args.bench_repo).resolve(),
            bench_output_root=Path(args.bench_output_root).resolve(),
            depth=args.depth,
            mode=args.mode,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout,
            limit=args.limit,
            skip_generate=args.skip_generate,
            inline_quality_enabled=args.inline_quality,
            skip_race=args.skip_race,
            skip_fact=args.skip_fact,
            bench_max_workers=args.bench_max_workers,
        ),
        sidecar_summary=sidecar_summary,
        generation_summary=generation_summary,
    )
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  Summary:     {summary_path}")
    print("=" * 60)


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run HouYi against DeepResearch Bench II with upstream sidecar scoring",
        epilog=(
            "Upstream sidecar requirements:\n"
            "  - GEMINI_API_KEY\n"
            "  - JINA_API_KEY\n"
            "These are required only in --bench-runtime official unless both --skip-race and --skip-fact are set."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--bench-repo",
        default="/Users/von/workspace/deep_research_bench",
        help="Path to the cloned DeepResearch Bench II repository",
    )
    parser.add_argument(
        "--bench-runtime",
        default="adapted_env",
        choices=["adapted_env", "official"],
        help="Bench II sidecar runtime: adapted_env uses the current HouYi environment; official keeps the upstream Gemini/Jina dependency path",
    )
    parser.add_argument(
        "--queries",
        default="/Users/von/workspace/deep_research_bench/data/prompt_data/query.jsonl",
        help="Path to benchmark query JSONL",
    )
    parser.add_argument(
        "--bench-output-root",
        default="benchmark/output/bench2",
        help="Output root for raw_data, cleaned_data, race, fact, and logs",
    )
    parser.add_argument(
        "--raw-data-path",
        default=None,
        help="Optional existing raw_data JSONL to score; defaults to <bench-output-root>/raw_data/<target-model>.jsonl",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path to write a machine-readable execution summary JSON",
    )
    parser.add_argument(
        "--target-model",
        default="houyi",
        help="Model name used by the official Bench II scripts for raw_data naming",
    )
    parser.add_argument("--depth", default="deep", choices=["quick", "standard", "deep"])
    parser.add_argument(
        "--mode",
        default="delegate",
        choices=["direct", "delegate", "autonomous"],
        help="HouYi orchestration mode for article generation",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Max sub-agents for delegate/autonomous modes",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600, help="Per-query timeout in seconds")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run first N benchmark queries"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Restart HouYi raw_data generation"
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip HouYi article generation and score an existing raw_data file",
    )
    parser.add_argument(
        "--inline-quality",
        action="store_true",
        help="Keep HouYi internal quality evaluation enabled during article generation",
    )
    parser.add_argument(
        "--bench-max-workers",
        type=int,
        default=5,
        help="Worker count forwarded to the official Bench II scripts",
    )
    parser.add_argument(
        "--skip-cleaning", action="store_true", help="Skip upstream RACE article cleaning"
    )
    parser.add_argument(
        "--only-zh", action="store_true", help="Only run Chinese tasks in upstream RACE"
    )
    parser.add_argument(
        "--only-en", action="store_true", help="Only run English tasks in upstream RACE"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force upstream Bench II re-evaluation"
    )
    parser.add_argument("--skip-race", action="store_true", help="Skip upstream RACE sidecar stage")
    parser.add_argument("--skip-fact", action="store_true", help="Skip upstream FACT sidecar stage")
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
