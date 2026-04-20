from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import houyi.arena.benchmark_runner as benchmark_runner_module
from houyi.application.research.types import ResearchSettings
from houyi.arena.benchmark_runner import (
    BenchmarkResult,
    BenchmarkRunner,
    _append_metrics,
    _append_result,
    _build_summary,
    _load_done_ids,
    _load_queries,
    _metrics_output_path,
    _renumber_citations,
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
            section_input_metrics=[
                {
                    "section_id": "sec_1",
                    "title": "Overview",
                    "relevant_source_count": 7,
                    "intermediate_context_chars": 320,
                }
            ],
        )

        _append_metrics(metrics_path, result)

        lines = metrics_path.read_text().strip().split("\n")
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["id"] == "t1"
        assert row["duration_seconds"] == 12.5
        assert row["search_elapsed_ms"] == 210.4
        assert row["phase_timings_ms"]["total_ms"] == 300.3
        assert row["section_input_metrics"][0]["relevant_source_count"] == 7
        assert row["section_input_metrics"][0]["intermediate_context_chars"] == 320


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
        assert "Claim A [1]." in article
        assert "## References" in article
        assert "- [1] [Source 1](https://example.com)" in article

    def test_strips_orphan_refs(self):
        report = MagicMock()
        report.title = "Test"
        report.summary = ""

        ref = MagicMock()
        ref.reference_id = "ref_aaa"
        ref.url = "https://a.example"
        ref.title = "A"
        report.references = [ref]

        section = MagicMock()
        section.title = "S1"
        section.content = "Known [ref_aaa] and orphan [ref_bbb123] token."
        cit = MagicMock()
        cit.reference_id = "ref_aaa"
        section.citations = [cit]
        report.sections = [section]

        article = _report_to_article(report)
        assert "Known [1] and orphan  token." in article
        assert "[ref_bbb123]" not in article
        assert "- [1] [A](https://a.example)" in article

    def test_drops_uncited_references(self):
        report = MagicMock()
        report.title = "Test"
        report.summary = ""

        cited = MagicMock()
        cited.reference_id = "ref_cited"
        cited.url = "https://cited.example"
        cited.title = "Cited"

        uncited = MagicMock()
        uncited.reference_id = "ref_uncited"
        uncited.url = "https://unrelated.example"
        uncited.title = "Unrelated"

        report.references = [cited, uncited]

        section = MagicMock()
        section.title = "S1"
        section.content = "Claim A [ref_cited]."
        cit = MagicMock()
        cit.reference_id = "ref_cited"
        section.citations = [cit]
        report.sections = [section]

        article = _report_to_article(report)
        assert "Claim A [1]." in article
        assert "- [1] [Cited](https://cited.example)" in article
        assert "https://unrelated.example" not in article

    def test_uses_id_without_title(self):
        report = MagicMock()
        report.title = "Test Report"
        report.summary = ""

        ref = MagicMock()
        ref.reference_id = "ref_1"
        ref.url = "https://example.com"
        ref.title = ""
        report.references = [ref]

        section = MagicMock()
        section.title = "Section 1"
        section.content = "Claim A [ref_1]."
        cit = MagicMock()
        cit.reference_id = "ref_1"
        section.citations = [cit]
        report.sections = [section]

        article = _report_to_article(report)
        assert "Claim A [1]." in article
        # Title falls back to the reference id when no human-readable title.
        assert "- [1] [ref_1](https://example.com)" in article

    def test_resolves_refs(self):
        """Refs written in content but omitted from the structured citations
        array should still resolve via the second-pass ref_lookup."""
        report = MagicMock()
        report.title = "Test"
        report.summary = ""

        ref_a = MagicMock()
        ref_a.reference_id = "ref_aaa"
        ref_a.url = "https://a.example"
        ref_a.title = "Source A"
        ref_b = MagicMock()
        ref_b.reference_id = "ref_bbb"
        ref_b.url = "https://b.example"
        ref_b.title = "Source B"
        report.references = [ref_a, ref_b]

        section = MagicMock()
        section.title = "S1"
        # ref_bbb is in the content but NOT in section.citations
        section.content = "Fact A [ref_aaa]. Fact B [ref_bbb]."
        cit = MagicMock()
        cit.reference_id = "ref_aaa"
        section.citations = [cit]  # only ref_aaa
        report.sections = [section]

        article = _report_to_article(report)
        assert "Fact A [1]. Fact B [2]." in article
        assert "[ref_bbb]" not in article
        assert "- [1] [Source A](https://a.example)" in article
        assert "- [2] [Source B](https://b.example)" in article


class TestRenumberCitations:
    def test_numbers_by_first_appearance(self):
        article = (
            "# Title\n\nClaim A [Source B](https://b.example). "
            "Claim B [Source A](https://a.example)."
        )
        out = _renumber_citations(article)
        # The first appearance wins: B=1, A=2.
        assert "Claim A [1]. Claim B [2]." in out
        assert "- [1] [Source B](https://b.example)" in out
        assert "- [2] [Source A](https://a.example)" in out

    def test_dedups_repeated_url(self):
        article = (
            "# Title\n\nFact [Shared](https://same.example) and "
            "again [Other Title](https://same.example)."
        )
        out = _renumber_citations(article)
        # Same URL reuses the same number; the first-appearance title wins.
        assert "Fact [1] and again [1]." in out
        assert out.count("- [1]") == 1
        assert "- [1] [Shared](https://same.example)" in out

    def test_noop_without_links(self):
        article = "# Plain\n\nNo citations here."
        out = _renumber_citations(article)
        assert out.rstrip() == article.rstrip()
        assert "## References" not in out


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

    async def test_keeps_phase_timings(self, tmp_path: Path):
        q_path = tmp_path / "q.jsonl"
        q_path.write_text('{"id": "1", "prompt": "Q1"}\n')
        out_path = tmp_path / "out.jsonl"

        llm = MagicMock()
        ws = MagicMock()
        runner = BenchmarkRunner(
            llm_adapter=llm, web_search=ws, settings=ResearchSettings(depth="quick"), concurrency=1
        )

        runtime = AsyncMock()
        runtime.start = AsyncMock(return_value=None)
        runtime.confirm_plan = AsyncMock(return_value=None)
        runtime.execute = AsyncMock(side_effect=RuntimeError("boom"))
        runtime.phase_timings_ms = {"partial_total_ms": 321.0}
        runtime.aggregate_ms = 12.0
        runtime.intermediate_ms = 0.0
        runtime.search_elapsed_ms = 45.0
        runtime._runtime_timeout = Mock(return_value=1)

        with patch.object(benchmark_runner_module, "ResearchRuntime", return_value=runtime):
            result = await runner._execute_query(_load_queries(q_path)[0])

        assert result.error == "boom"
        assert result.phase_timings_ms["partial_total_ms"] == 321.0
        assert result.phase_timings_ms["aggregate_ms"] == 12.0
