"""Tests for houyi.arena.benchmark_runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.arena.benchmark_runner import (
    BenchmarkResult,
    BenchmarkRunner,
    _append_metrics,
    _append_result,
    _build_summary,
    _load_done_ids,
    _load_queries,
    _metrics_output_path,
    _report_to_article,
)


class TestLoadQueries:
    def test_loads_standard_format(self, tmp_path: Path):
        p = tmp_path / "q.jsonl"
        p.write_text(
            '{"id": "1", "prompt": "What is AI?"}\n{"id": "2", "prompt": "Explain LLMs"}\n'
        )
        qs = _load_queries(p)
        assert len(qs) == 2
        assert qs[0].id == "1"
        assert qs[1].prompt == "Explain LLMs"

    def test_skips_blank_lines(self, tmp_path: Path):
        p = tmp_path / "q.jsonl"
        p.write_text('{"id": "1", "prompt": "Q1"}\n\n{"id": "2", "prompt": "Q2"}\n')
        assert len(_load_queries(p)) == 2

    def test_query_field_fallback(self, tmp_path: Path):
        p = tmp_path / "q.jsonl"
        p.write_text('{"id": "1", "query": "Fallback field"}\n')
        qs = _load_queries(p)
        assert qs[0].prompt == "Fallback field"


class TestLoadDoneIds:
    def test_missing_file(self, tmp_path: Path):
        assert _load_done_ids(tmp_path / "nope.jsonl") == set()

    def test_existing_results(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        p.write_text('{"id": "a"}\n{"id": "b"}\n')
        assert _load_done_ids(p) == {"a", "b"}

    def test_malformed_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        p.write_text('{"id": "ok"}\nnot-json\n')
        assert _load_done_ids(p) == {"ok"}


class TestAppendResult:
    def test_appends_jsonl(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        r = BenchmarkResult(id="t1", prompt="Q", article="A")
        _append_result(p, r)
        _append_result(p, BenchmarkResult(id="t2", prompt="Q2", article="A2"))
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "t1"


class TestAppendMetrics:
    def test_appends_metrics_jsonl(self, tmp_path: Path):
        out_path = tmp_path / "out.jsonl"
        metrics_path = _metrics_output_path(out_path)
        result = BenchmarkResult(
            id="t1",
            prompt="Q",
            article="A",
            duration_seconds=12.5,
            search_elapsed_ms=210.4,
            quality_score=88.8,
            phase_timings_ms={"report_generate_ms": 100.1, "total_ms": 300.3},
        )

        _append_metrics(metrics_path, result)

        lines = metrics_path.read_text().strip().split("\n")
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["id"] == "t1"
        assert row["duration_seconds"] == 12.5
        assert row["search_elapsed_ms"] == 210.4
        assert row["phase_timings_ms"]["total_ms"] == 300.3


class TestBuildSummary:
    def test_all_succeed(self):
        rs = [
            BenchmarkResult(id="1", prompt="", article="", quality_score=80, duration_seconds=10),
            BenchmarkResult(id="2", prompt="", article="", quality_score=60, duration_seconds=20),
        ]
        s = _build_summary(rs)
        assert s.total == 2
        assert s.succeeded == 2
        assert s.failed == 0
        assert s.avg_quality == 70.0

    def test_with_failures(self):
        rs = [
            BenchmarkResult(id="1", prompt="", article="", quality_score=80, duration_seconds=5),
            BenchmarkResult(id="2", prompt="", article="", error="timeout", duration_seconds=600),
        ]
        s = _build_summary(rs)
        assert s.succeeded == 1
        assert s.failed == 1
        assert s.avg_quality == 80.0


class TestReportToArticle:
    def test_converts_with_citations(self):
        report = MagicMock()
        report.title = "Test Report"
        report.summary = "Summary text"

        ref = MagicMock()
        ref.reference_id = "ref_1"
        ref.url = "https://example.com"
        ref.title = "Source 1"
        report.references = [ref]

        section = MagicMock()
        section.title = "Section 1"
        section.content = "Claim A [ref_1]."
        cit = MagicMock()
        cit.reference_id = "ref_1"
        section.citations = [cit]
        report.sections = [section]

        article = _report_to_article(report)
        assert "# Test Report" in article
        assert "[ref_1](https://example.com)" in article
        assert "## References" in article


class TestRunnerInit:
    def test_default_settings(self):
        llm = MagicMock()
        ws = MagicMock()
        runner = BenchmarkRunner(llm_adapter=llm, web_search=ws)
        assert runner._settings.depth == "deep"
        assert runner._concurrency == 3


@pytest.mark.asyncio
class TestRunnerExecute:
    async def test_resume_skips_done(self, tmp_path: Path):
        q_path = tmp_path / "q.jsonl"
        q_path.write_text('{"id": "1", "prompt": "Q1"}\n{"id": "2", "prompt": "Q2"}\n')
        out_path = tmp_path / "out.jsonl"
        out_path.write_text('{"id": "1", "prompt": "Q1", "article": "done"}\n')

        llm = MagicMock()
        ws = MagicMock()
        runner = BenchmarkRunner(llm_adapter=llm, web_search=ws, concurrency=1)

        with patch.object(runner, "_execute_query", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = BenchmarkResult(id="2", prompt="Q2", article="result")
            summary = await runner.run(q_path, out_path, resume=True)

        assert summary.total == 1
        mock_exec.assert_called_once()
        assert mock_exec.call_args[0][0].id == "2"
        metrics_path = _metrics_output_path(out_path)
        metrics = metrics_path.read_text().strip().split("\n")
        assert len(metrics) == 1
        metrics_row = json.loads(metrics[0])
        assert metrics_row["id"] == "2"
        assert metrics_row["search_elapsed_ms"] == 0.0
        assert metrics_row["phase_timings_ms"] == {}

    async def test_error_captured(self, tmp_path: Path):
        q_path = tmp_path / "q.jsonl"
        q_path.write_text('{"id": "1", "prompt": "Q1"}\n')
        out_path = tmp_path / "out.jsonl"

        llm = MagicMock()
        ws = MagicMock()
        runner = BenchmarkRunner(llm_adapter=llm, web_search=ws, concurrency=1)

        with patch.object(runner, "_execute_query", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = BenchmarkResult(
                id="1", prompt="Q1", article="", error="timeout"
            )
            summary = await runner.run(q_path, out_path, resume=False)

        assert summary.failed == 1
