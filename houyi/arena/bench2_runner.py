"""Bench II sidecar orchestration for HouYi Deep Research.

Acknowledgement:
This module intentionally mirrors the public DeepResearch Bench II evaluation
flow published by the upstream open-source project
https://github.com/Ayanami0730/deep_research_bench.

In particular, the stage layout and entrypoints are aligned with the upstream
`run_benchmark.sh` wrapper plus these public scripts:
- `deepresearch_bench_race.py`
- `utils.extract`
- `utils.deduplicate`
- `utils.scrape`
- `utils.validate`
- `utils.stat`

HouYi keeps article generation in-repo, then runs the official-style Bench II
scoring pipeline as a sidecar so report generation and benchmark scoring remain
loosely coupled.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from houyi.arena.bench2_compat import api as bench2_compat_api
from houyi.arena.bench2_compat import extract as bench2_compat_extract
from houyi.arena.bench2_compat import stat as bench2_compat_stat
from houyi.arena.bench2_compat import validate as bench2_compat_validate


class SubprocessRunner(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class Bench2SidecarConfig:
    repo_root: Path
    output_root: Path
    query_file: Path
    target_model: str = "houyi"
    runtime_mode: str = "official"
    max_workers: int = 5
    limit: int | None = None
    skip_cleaning: bool = False
    only_zh: bool = False
    only_en: bool = False
    force: bool = False
    run_race: bool = True
    run_fact: bool = True


@dataclass(frozen=True, slots=True)
class Bench2Workspace:
    repo_root: Path
    output_root: Path
    query_file: Path
    raw_data_path: Path
    raw_data_dir: Path
    cleaned_data_dir: Path
    race_output_dir: Path
    fact_output_dir: Path
    log_path: Path

    def artifact_paths(self) -> dict[str, str]:
        return {
            "query_file": str(self.query_file),
            "raw_data_path": str(self.raw_data_path),
            "cleaned_data_dir": str(self.cleaned_data_dir),
            "race_output_dir": str(self.race_output_dir),
            "race_raw_results_path": str(self.race_output_dir / "raw_results.jsonl"),
            "race_result_path": str(self.race_output_dir / "race_result.txt"),
            "fact_output_dir": str(self.fact_output_dir),
            "fact_extracted_path": str(self.fact_output_dir / "extracted.jsonl"),
            "fact_deduplicated_path": str(self.fact_output_dir / "deduplicated.jsonl"),
            "fact_scraped_path": str(self.fact_output_dir / "scraped.jsonl"),
            "fact_validated_path": str(self.fact_output_dir / "validated.jsonl"),
            "fact_result_path": str(self.fact_output_dir / "fact_result.txt"),
            "log_path": str(self.log_path),
        }


@dataclass(frozen=True, slots=True)
class Bench2Command:
    name: str
    argv: tuple[str, ...]


@dataclass(slots=True)
class Bench2RunSummary:
    workspace: Bench2Workspace
    executed_steps: list[str] = field(default_factory=list)
    race_scores: dict[str, float] = field(default_factory=dict)
    fact_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed_steps": list(self.executed_steps),
            "race_scores": dict(self.race_scores),
            "fact_scores": dict(self.fact_scores),
            "artifacts": self.workspace.artifact_paths(),
        }


@dataclass(frozen=True, slots=True)
class Bench2GenerationSummary:
    total: int
    succeeded: int
    failed: int
    avg_duration: float
    avg_quality: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "avg_duration": self.avg_duration,
            "avg_quality": self.avg_quality,
        }


@dataclass(frozen=True, slots=True)
class Bench2ExecutionContext:
    target_model: str
    runtime_mode: str
    query_file: Path
    raw_data_path: Path
    metrics_path: Path | None
    bench_repo: Path
    bench_output_root: Path
    depth: str
    mode: str
    concurrency: int
    timeout_seconds: float
    limit: int | None
    skip_generate: bool
    inline_quality_enabled: bool
    skip_race: bool
    skip_fact: bool
    bench_max_workers: int


def build_bench2_summary_payload(
    *,
    context: Bench2ExecutionContext,
    sidecar_summary: Bench2RunSummary,
    generation_summary: Bench2GenerationSummary | None,
) -> dict[str, Any]:
    return {
        "target_model": context.target_model,
        "runtime_mode": context.runtime_mode,
        "query_file": str(context.query_file),
        "raw_data_path": str(context.raw_data_path),
        "metrics_path": str(context.metrics_path) if context.metrics_path is not None else None,
        "bench_repo": str(context.bench_repo),
        "bench_output_root": str(context.bench_output_root),
        "depth": context.depth,
        "mode": context.mode,
        "concurrency": context.concurrency,
        "timeout_seconds": context.timeout_seconds,
        "limit": context.limit,
        "skip_generate": context.skip_generate,
        "inline_quality_enabled": context.inline_quality_enabled,
        "skip_race": context.skip_race,
        "skip_fact": context.skip_fact,
        "bench_max_workers": context.bench_max_workers,
        "generation": generation_summary.to_dict() if generation_summary is not None else None,
        "sidecar": sidecar_summary.to_dict(),
    }


class Bench2SidecarRunner:
    """Run the public Bench II scoring flow against HouYi-generated articles."""

    def __init__(self, subprocess_runner: SubprocessRunner | None = None) -> None:
        self._subprocess_runner = subprocess_runner or subprocess.run

    def validate_preconditions(self, config: Bench2SidecarConfig) -> Path:
        repo_root = config.repo_root.resolve()
        self._validate_repo(repo_root)
        missing_env = [name for name in self._required_env_vars(config) if not os.environ.get(name)]
        if missing_env:
            missing_list = ", ".join(missing_env)
            raise OSError(
                "Bench II sidecar preflight failed. Missing upstream environment variables: "
                f"{missing_list}. The official Bench II repo requires these keys before scoring can start. "
                "Export the missing keys or rerun with --skip-race/--skip-fact."
            )
        return repo_root

    def run(self, config: Bench2SidecarConfig, *, raw_data_path: Path) -> Bench2RunSummary:
        self.validate_preconditions(config)
        workspace = self.prepare_workspace(config)
        prepared_raw_data = self._prepare_raw_data(raw_data_path.resolve(), workspace.raw_data_path)
        workspace = Bench2Workspace(
            repo_root=workspace.repo_root,
            output_root=workspace.output_root,
            query_file=workspace.query_file,
            raw_data_path=prepared_raw_data,
            raw_data_dir=workspace.raw_data_dir,
            cleaned_data_dir=workspace.cleaned_data_dir,
            race_output_dir=workspace.race_output_dir,
            fact_output_dir=workspace.fact_output_dir,
            log_path=workspace.log_path,
        )
        commands = self.build_commands(config, workspace)

        self._append_log(
            workspace.log_path,
            f"# Bench II sidecar run\n# repo_root={workspace.repo_root}\n"
            f"# raw_data_path={workspace.raw_data_path}\n# query_file={workspace.query_file}\n",
        )

        executed_steps: list[str] = []
        for command in commands:
            self._run_command(command, workspace.repo_root, workspace.log_path)
            executed_steps.append(command.name)

        return Bench2RunSummary(
            workspace=workspace,
            executed_steps=executed_steps,
            race_scores=_parse_score_file(workspace.race_output_dir / "race_result.txt"),
            fact_scores=_parse_score_file(workspace.fact_output_dir / "fact_result.txt"),
        )

    def prepare_workspace(self, config: Bench2SidecarConfig) -> Bench2Workspace:
        output_root = config.output_root.resolve()
        raw_data_dir = output_root / "raw_data"
        cleaned_data_dir = output_root / "cleaned_data"
        race_output_dir = output_root / "race" / config.target_model
        fact_output_dir = output_root / "fact" / config.target_model
        log_path = output_root / f"{config.target_model}.bench2.log"
        raw_data_path = raw_data_dir / f"{config.target_model}.jsonl"
        prepared_query_path = output_root / "prepared_query.jsonl"

        for path in (output_root, raw_data_dir, cleaned_data_dir, race_output_dir, fact_output_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._prepare_query_file(config.query_file.resolve(), prepared_query_path)

        return Bench2Workspace(
            repo_root=self._prepare_execution_root(config),
            output_root=output_root,
            query_file=prepared_query_path,
            raw_data_path=raw_data_path,
            raw_data_dir=raw_data_dir,
            cleaned_data_dir=cleaned_data_dir,
            race_output_dir=race_output_dir,
            fact_output_dir=fact_output_dir,
            log_path=log_path,
        )

    def build_commands(
        self,
        config: Bench2SidecarConfig,
        workspace: Bench2Workspace,
    ) -> list[Bench2Command]:
        query_file = str(workspace.query_file)
        commands: list[Bench2Command] = []

        if config.run_race:
            race_cmd = [
                sys.executable,
                "-u",
                "deepresearch_bench_race.py",
                config.target_model,
                "--raw_data_dir",
                str(workspace.raw_data_dir),
                "--cleaned_data_dir",
                str(workspace.cleaned_data_dir),
                "--max_workers",
                str(config.max_workers),
                "--query_file",
                query_file,
                "--output_dir",
                str(workspace.race_output_dir),
            ]
            if config.limit is not None:
                race_cmd.extend(["--limit", str(config.limit)])
            if config.skip_cleaning:
                race_cmd.append("--skip_cleaning")
            if config.only_zh:
                race_cmd.append("--only_zh")
            if config.only_en:
                race_cmd.append("--only_en")
            if config.force:
                race_cmd.append("--force")
            commands.append(Bench2Command(name="race", argv=tuple(race_cmd)))

        if config.run_fact:
            extracted = workspace.fact_output_dir / "extracted.jsonl"
            deduplicated = workspace.fact_output_dir / "deduplicated.jsonl"
            scraped = workspace.fact_output_dir / "scraped.jsonl"
            validated = workspace.fact_output_dir / "validated.jsonl"
            fact_result = workspace.fact_output_dir / "fact_result.txt"
            fact_workers = self._fact_workers(config)
            base_args = [
                "--query_data_path",
                query_file,
                "--n_total_process",
                str(fact_workers),
            ]
            commands.extend(
                [
                    Bench2Command(
                        name="fact_extract",
                        argv=(
                            sys.executable,
                            "-u",
                            "-m",
                            "utils.extract",
                            "--raw_data_path",
                            str(workspace.raw_data_path),
                            "--output_path",
                            str(extracted),
                            *base_args,
                        ),
                    ),
                    Bench2Command(
                        name="fact_deduplicate",
                        argv=(
                            sys.executable,
                            "-u",
                            "-m",
                            "utils.deduplicate",
                            "--raw_data_path",
                            str(extracted),
                            "--output_path",
                            str(deduplicated),
                            *base_args,
                        ),
                    ),
                    Bench2Command(
                        name="fact_scrape",
                        argv=(
                            sys.executable,
                            "-u",
                            "-m",
                            "utils.scrape",
                            "--raw_data_path",
                            str(deduplicated),
                            "--output_path",
                            str(scraped),
                            "--n_total_process",
                            str(fact_workers),
                        ),
                    ),
                    Bench2Command(
                        name="fact_validate",
                        argv=(
                            sys.executable,
                            "-u",
                            "-m",
                            "utils.validate",
                            "--raw_data_path",
                            str(scraped),
                            "--output_path",
                            str(validated),
                            *base_args,
                        ),
                    ),
                    Bench2Command(
                        name="fact_stat",
                        argv=(
                            sys.executable,
                            "-u",
                            "-m",
                            "utils.stat",
                            "--input_path",
                            str(validated),
                            "--output_path",
                            str(fact_result),
                        ),
                    ),
                ]
            )

        return commands

    def _run_command(self, command: Bench2Command, repo_root: Path, log_path: Path) -> None:
        rendered = " ".join(shlex.quote(part) for part in command.argv)
        self._append_log(log_path, f"\n# BEGIN {command.name}\n$ {rendered}\n")
        command_env = os.environ.copy()
        existing_pythonpath = command_env.get("PYTHONPATH", "")
        repo_parent = str(Path(__file__).resolve().parents[2])
        command_env["PYTHONPATH"] = (
            repo_parent
            if not existing_pythonpath
            else f"{repo_parent}{os.pathsep}{existing_pythonpath}"
        )
        if self._subprocess_runner is subprocess.run:
            proc = subprocess.Popen(
                list(command.argv),
                cwd=str(repo_root),
                env=command_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            def _pump_output() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._append_log(log_path, line)
                    sys.stdout.write(line)
                    sys.stdout.flush()

            output_thread = threading.Thread(target=_pump_output, daemon=True)
            output_thread.start()
            returncode = proc.wait()
            output_thread.join()
        else:
            proc = self._subprocess_runner(
                list(command.argv),
                cwd=str(repo_root),
                env=command_env,
                capture_output=True,
                text=True,
                check=False,
            )
            if getattr(proc, "stdout", None):
                self._append_log(log_path, proc.stdout)
            if getattr(proc, "stderr", None):
                self._append_log(log_path, proc.stderr)
            returncode = proc.returncode
        self._append_log(log_path, f"# END {command.name} rc={returncode}\n")
        if returncode != 0:
            raise RuntimeError(
                f"Bench II step '{command.name}' failed with exit code {returncode}. "
                f"See log: {log_path}"
            )

    def _validate_repo(self, repo_root: Path) -> None:
        required = [
            repo_root / "run_benchmark.sh",
            repo_root / "deepresearch_bench_race.py",
            repo_root / "utils" / "extract.py",
            repo_root / "utils" / "deduplicate.py",
            repo_root / "utils" / "scrape.py",
            repo_root / "utils" / "validate.py",
            repo_root / "utils" / "stat.py",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            missing_list = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Bench II repo is incomplete: {missing_list}")

    def _required_env_vars(self, config: Bench2SidecarConfig) -> tuple[str, ...]:
        if not config.run_race and not config.run_fact:
            return ()
        if config.runtime_mode == "adapted_env":
            return ()
        return ("GEMINI_API_KEY", "JINA_API_KEY")

    def _prepare_execution_root(self, config: Bench2SidecarConfig) -> Path:
        source_root = config.repo_root.resolve()
        if config.runtime_mode != "adapted_env":
            return source_root

        compat_root = config.output_root.resolve() / "_bench2_compat"
        if compat_root.exists():
            shutil.rmtree(compat_root)
        shutil.copytree(source_root, compat_root)
        stale_results_dir = compat_root / "results"
        if stale_results_dir.exists():
            shutil.rmtree(stale_results_dir)
        self._copy_compat_module(bench2_compat_api, compat_root / "utils" / "api.py")
        self._copy_compat_module(bench2_compat_extract, compat_root / "utils" / "extract.py")
        self._copy_compat_module(bench2_compat_validate, compat_root / "utils" / "validate.py")
        self._copy_compat_module(bench2_compat_stat, compat_root / "utils" / "stat.py")
        return compat_root

    def _copy_compat_module(self, module: Any, target_path: Path) -> None:
        source_path = Path(module.__file__).resolve()
        target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _fact_workers(self, config: Bench2SidecarConfig) -> int:
        if config.runtime_mode == "adapted_env":
            return 1
        return max(1, config.max_workers)

    def _prepare_query_file(self, source_path: Path, target_path: Path) -> Path:
        with (
            source_path.open(encoding="utf-8") as src,
            target_path.open("w", encoding="utf-8") as dst,
        ):
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "id" in data:
                    data["id"] = str(data["id"])
                dst.write(json.dumps(data, ensure_ascii=False) + "\n")
        return target_path

    def _prepare_raw_data(self, source_path: Path, target_path: Path) -> Path:
        if not source_path.exists():
            raise FileNotFoundError(f"Raw benchmark data not found: {source_path}")
        if source_path == target_path:
            return target_path
        shutil.copyfile(source_path, target_path)
        return target_path

    def _append_log(self, log_path: Path, content: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(content)


def _parse_score_file(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    scores: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        try:
            scores[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return scores
