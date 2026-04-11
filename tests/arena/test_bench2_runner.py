from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from houyi.arena.bench2_runner import (
    Bench2ExecutionContext,
    Bench2GenerationSummary,
    Bench2RunSummary,
    Bench2SidecarConfig,
    Bench2SidecarRunner,
    build_bench2_summary_payload,
)


def _create_bench_repo(root: Path) -> Path:
    (root / "utils").mkdir(parents=True, exist_ok=True)
    for rel_path in (
        "run_benchmark.sh",
        "deepresearch_bench_race.py",
        "utils/extract.py",
        "utils/deduplicate.py",
        "utils/scrape.py",
        "utils/validate.py",
        "utils/stat.py",
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder\n", encoding="utf-8")
    return root


def _config(repo_root: Path, output_root: Path, query_file: Path) -> Bench2SidecarConfig:
    return Bench2SidecarConfig(
        repo_root=repo_root,
        output_root=output_root,
        query_file=query_file,
        target_model="houyi",
        runtime_mode="official",
        max_workers=4,
        limit=3,
        skip_cleaning=True,
        only_en=True,
        force=True,
    )


class TestBench2SidecarRunner:
    def test_prepare_workspace(self, tmp_path: Path) -> None:
        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        output_root = tmp_path / "output"

        runner = Bench2SidecarRunner()
        workspace = runner.prepare_workspace(_config(repo_root, output_root, query_file))

        assert workspace.raw_data_path == output_root / "raw_data" / "houyi.jsonl"
        assert workspace.cleaned_data_dir.is_dir()
        assert workspace.race_output_dir.is_dir()
        assert workspace.fact_output_dir.is_dir()
        assert workspace.artifact_paths()["fact_result_path"].endswith("fact/houyi/fact_result.txt")

    def test_normalizes_query_ids(self, tmp_path: Path) -> None:
        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": 1, "prompt": "Q", "language": "zh"}\n', encoding="utf-8")

        runner = Bench2SidecarRunner()
        workspace = runner.prepare_workspace(_config(repo_root, tmp_path / "output", query_file))

        prepared_lines = workspace.query_file.read_text(encoding="utf-8").splitlines()
        prepared = json.loads(prepared_lines[0])
        assert prepared["id"] == "1"

    def test_uses_compat_root(self, tmp_path: Path) -> None:
        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        output_root = tmp_path / "output"
        config = Bench2SidecarConfig(
            repo_root=repo_root,
            output_root=output_root,
            query_file=query_file,
            target_model="houyi",
            runtime_mode="adapted_env",
        )

        runner = Bench2SidecarRunner()
        workspace = runner.prepare_workspace(config)

        assert workspace.repo_root == output_root / "_bench2_compat"
        assert (workspace.repo_root / "utils" / "api.py").exists()
        assert (workspace.repo_root / "utils" / "stat.py").exists()

    def test_compat_root_clears_stale_upstream_results(self, tmp_path: Path) -> None:
        repo_root = _create_bench_repo(tmp_path / "bench")
        stale = repo_root / "results" / "fact" / "claude-3-7-sonnet-latest"
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "fact_result.txt").write_text("valid_rate: 0.9999\n", encoding="utf-8")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        output_root = tmp_path / "output"
        config = Bench2SidecarConfig(
            repo_root=repo_root,
            output_root=output_root,
            query_file=query_file,
            target_model="houyi",
            runtime_mode="adapted_env",
        )

        runner = Bench2SidecarRunner()
        workspace = runner.prepare_workspace(config)

        assert not (workspace.repo_root / "results").exists()

    def test_limits_fact_workers(self, tmp_path: Path) -> None:
        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q", "language": "zh"}\n', encoding="utf-8")
        output_root = tmp_path / "output"
        config = Bench2SidecarConfig(
            repo_root=repo_root,
            output_root=output_root,
            query_file=query_file,
            target_model="houyi",
            runtime_mode="adapted_env",
            max_workers=5,
        )

        runner = Bench2SidecarRunner()
        workspace = runner.prepare_workspace(config)
        commands = runner.build_commands(config, workspace)

        fact_extract = list(
            commands[0].argv if commands[0].name == "fact_extract" else commands[1].argv
        )
        assert "--n_total_process" in fact_extract
        assert fact_extract[fact_extract.index("--n_total_process") + 1] == "1"

    def test_build_commands(self, tmp_path: Path) -> None:
        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        output_root = tmp_path / "output"

        runner = Bench2SidecarRunner()
        config = _config(repo_root, output_root, query_file)
        workspace = runner.prepare_workspace(config)
        commands = runner.build_commands(config, workspace)

        assert [command.name for command in commands] == [
            "race",
            "fact_extract",
            "fact_deduplicate",
            "fact_scrape",
            "fact_validate",
            "fact_stat",
        ]
        race_cmd = list(commands[0].argv)
        assert race_cmd[:4] == [sys.executable, "-u", "deepresearch_bench_race.py", "houyi"]
        assert "--skip_cleaning" in race_cmd
        assert "--only_en" in race_cmd
        assert "--force" in race_cmd
        assert workspace.raw_data_dir.as_posix() in race_cmd
        assert str(workspace.query_file) in race_cmd

    def test_run_copies(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
        monkeypatch.setenv("JINA_API_KEY", "test-jina")

        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        output_root = tmp_path / "output"
        raw_data_path = tmp_path / "external.jsonl"
        raw_data_path.write_text('{"id": "1", "prompt": "Q", "article": "A"}\n', encoding="utf-8")

        def fake_subprocess(
            args: list[str],
            *,
            cwd: str,
            env: dict[str, str],
            capture_output: bool,
            text: bool,
            check: bool,
        ) -> subprocess.CompletedProcess[str]:
            assert cwd == str(repo_root.resolve())
            assert capture_output is True
            assert text is True
            assert check is False
            if "deepresearch_bench_race.py" in args:
                race_result = output_root / "race" / "houyi" / "race_result.txt"
                race_result.parent.mkdir(parents=True, exist_ok=True)
                race_result.write_text(
                    "Overall Score: 0.5123\nReadability: 0.6000\n", encoding="utf-8"
                )
            if "utils.stat" in args:
                fact_result = output_root / "fact" / "houyi" / "fact_result.txt"
                fact_result.parent.mkdir(parents=True, exist_ok=True)
                fact_result.write_text(
                    "total_citations: 12.0\nvalid_rate: 0.9500\n",
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")

        runner = Bench2SidecarRunner(subprocess_runner=fake_subprocess)
        summary = runner.run(
            _config(repo_root, output_root, query_file), raw_data_path=raw_data_path
        )

        prepared_raw_data = output_root / "raw_data" / "houyi.jsonl"
        assert prepared_raw_data.read_text(encoding="utf-8") == raw_data_path.read_text(
            encoding="utf-8"
        )
        assert summary.workspace.query_file.exists()
        assert summary.executed_steps == [
            "race",
            "fact_extract",
            "fact_deduplicate",
            "fact_scrape",
            "fact_validate",
            "fact_stat",
        ]
        assert summary.race_scores["Overall Score"] == pytest.approx(0.5123)
        assert summary.fact_scores["valid_rate"] == pytest.approx(0.95)
        log_text = summary.workspace.log_path.read_text(encoding="utf-8")
        assert "# BEGIN race" in log_text
        assert "# END fact_stat rc=0" in log_text
        assert "deepresearch_bench_race.py" in log_text
        assert "utils.stat" in log_text

    def test_requires_complete_repo(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "bench"
        repo_root.mkdir(parents=True, exist_ok=True)
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        raw_data_path = tmp_path / "raw.jsonl"
        raw_data_path.write_text('{"id": "1", "prompt": "Q", "article": "A"}\n', encoding="utf-8")

        runner = Bench2SidecarRunner()
        with pytest.raises(FileNotFoundError, match="Bench II repo is incomplete"):
            runner.run(
                _config(repo_root, tmp_path / "output", query_file), raw_data_path=raw_data_path
            )

    def test_needs_env_vars(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("JINA_API_KEY", raising=False)

        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")

        runner = Bench2SidecarRunner()
        with pytest.raises(OSError, match="GEMINI_API_KEY, JINA_API_KEY"):
            runner.validate_preconditions(_config(repo_root, tmp_path / "output", query_file))

    def test_skips_env_check(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("JINA_API_KEY", raising=False)

        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        config = Bench2SidecarConfig(
            repo_root=repo_root,
            output_root=tmp_path / "output",
            query_file=query_file,
            target_model="houyi",
            runtime_mode="adapted_env",
        )

        runner = Bench2SidecarRunner()
        assert runner.validate_preconditions(config) == repo_root.resolve()

    def test_allows_skip_all(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("JINA_API_KEY", raising=False)

        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        config = Bench2SidecarConfig(
            repo_root=repo_root,
            output_root=tmp_path / "output",
            query_file=query_file,
            target_model="houyi",
            run_race=False,
            run_fact=False,
        )

        runner = Bench2SidecarRunner()
        assert runner.validate_preconditions(config) == repo_root.resolve()

    def test_build_summary_payload(self, tmp_path: Path) -> None:
        repo_root = _create_bench_repo(tmp_path / "bench")
        query_file = tmp_path / "query.jsonl"
        query_file.write_text('{"id": "1", "prompt": "Q"}\n', encoding="utf-8")
        output_root = tmp_path / "output"

        runner = Bench2SidecarRunner()
        workspace = runner.prepare_workspace(_config(repo_root, output_root, query_file))
        sidecar_summary = Bench2RunSummary(
            workspace=workspace,
            executed_steps=["race", "fact_extract"],
            race_scores={"Overall Score": 0.51},
            fact_scores={"valid_rate": 0.95},
        )

        payload = build_bench2_summary_payload(
            context=Bench2ExecutionContext(
                target_model="houyi",
                runtime_mode="adapted_env",
                query_file=query_file,
                raw_data_path=workspace.raw_data_path,
                metrics_path=output_root / "raw_data" / "houyi.metrics.jsonl",
                bench_repo=repo_root,
                bench_output_root=output_root,
                depth="deep",
                mode="delegate",
                concurrency=2,
                timeout_seconds=600.0,
                limit=3,
                skip_generate=False,
                inline_quality_enabled=False,
                skip_race=False,
                skip_fact=False,
                bench_max_workers=4,
            ),
            sidecar_summary=sidecar_summary,
            generation_summary=Bench2GenerationSummary(
                total=3,
                succeeded=2,
                failed=1,
                avg_duration=12.5,
                avg_quality=88.0,
            ),
        )

        assert payload["target_model"] == "houyi"
        assert payload["runtime_mode"] == "adapted_env"
        assert payload["generation"]["succeeded"] == 2
        assert payload["sidecar"]["race_scores"]["Overall Score"] == pytest.approx(0.51)
        assert payload["sidecar"]["artifacts"]["race_result_path"].endswith(
            "race/houyi/race_result.txt"
        )
