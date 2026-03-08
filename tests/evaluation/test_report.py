"""Tests for report generation."""

import json

from houyi.assurance.evaluation.base import EvaluationResult, EvaluationSummary
from houyi.assurance.evaluation.report import ReportGenerator


class TestReportGenerator:
    """Test ReportGenerator class."""

    def test_generate_html_basic(self, tmp_path):
        """Test basic HTML report generation."""
        results = [
            EvaluationResult(
                evaluator="accuracy",
                input="test1",
                output="output1",
                expected_output="output1",
                passed=True,
                score=1.0,
            ),
            EvaluationResult(
                evaluator="accuracy",
                input="test2",
                output="output2",
                expected_output="output3",
                passed=False,
                score=0.0,
            ),
        ]

        summary = EvaluationSummary(
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            pass_rate=0.5,
            avg_score=0.5,
            avg_cost=0.01,
            avg_latency=100.0,
            results=results,
        )

        output_file = tmp_path / "report.html"
        ReportGenerator.generate_html(summary, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "Evaluation Report" in content
        assert "50.0%" in content or "50%" in content
        assert "accuracy" in content

    def test_generate_html_with_title(self, tmp_path):
        """Test HTML report with custom title."""
        results = [
            EvaluationResult(
                evaluator="test", input="test", output="output", passed=True, score=1.0
            )
        ]

        summary = EvaluationSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            pass_rate=1.0,
            avg_score=1.0,
            avg_cost=0.0,
            avg_latency=50.0,
            results=results,
        )

        output_file = tmp_path / "custom_report.html"
        ReportGenerator.generate_html(summary, str(output_file), title="Custom Report")

        assert output_file.exists()
        content = output_file.read_text()
        assert "Custom Report" in content

    def test_generate_json_basic(self, tmp_path):
        """Test basic JSON report generation."""
        results = [
            EvaluationResult(
                evaluator="accuracy", input="test1", output="output1", passed=True, score=1.0
            )
        ]

        summary = EvaluationSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            pass_rate=1.0,
            avg_score=1.0,
            avg_cost=0.01,
            avg_latency=100.0,
            results=results,
        )

        output_file = tmp_path / "report.json"
        ReportGenerator.generate_json(summary, str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        assert data["summary"]["total_cases"] == 1
        assert data["summary"]["passed_cases"] == 1
        assert data["summary"]["pass_rate"] == 1.0
        assert len(data["results"]) == 1

    def test_generate_markdown_basic(self, tmp_path):
        """Test basic Markdown report generation."""
        results = [
            EvaluationResult(
                evaluator="accuracy", input="test1", output="output1", passed=True, score=1.0
            ),
            EvaluationResult(
                evaluator="latency",
                input="test1",
                output="output1",
                passed=True,
                score=0.9,
                duration_ms=50.0,
            ),
        ]

        summary = EvaluationSummary(
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            pass_rate=1.0,
            avg_score=0.95,
            avg_cost=0.01,
            avg_latency=50.0,
            results=results,
        )

        output_file = tmp_path / "report.md"
        ReportGenerator.generate_markdown(summary, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "# Evaluation Report" in content
        assert "Total Cases" in content
        assert "Pass Rate" in content
        assert "accuracy" in content
        assert "latency" in content

    def test_generate_markdown_with_title(self, tmp_path):
        """Test Markdown report with custom title."""
        results = [
            EvaluationResult(
                evaluator="test", input="test", output="output", passed=True, score=1.0
            )
        ]

        summary = EvaluationSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            pass_rate=1.0,
            avg_score=1.0,
            avg_cost=0.0,
            avg_latency=50.0,
            results=results,
        )

        output_file = tmp_path / "custom.md"
        ReportGenerator.generate_markdown(summary, str(output_file), title="My Custom Report")

        assert output_file.exists()
        content = output_file.read_text()
        assert "# My Custom Report" in content

    def test_generate_html_multiple_evaluators(self, tmp_path):
        """Test HTML report with multiple evaluators."""
        results = [
            EvaluationResult(
                evaluator="accuracy", input="test1", output="output1", passed=True, score=1.0
            ),
            EvaluationResult(
                evaluator="latency", input="test1", output="output1", passed=True, score=0.9
            ),
            EvaluationResult(
                evaluator="accuracy", input="test2", output="output2", passed=False, score=0.0
            ),
        ]

        summary = EvaluationSummary(
            total_cases=3,
            passed_cases=2,
            failed_cases=1,
            pass_rate=0.67,
            avg_score=0.63,
            avg_cost=0.01,
            avg_latency=100.0,
            results=results,
        )

        output_file = tmp_path / "multi_eval.html"
        ReportGenerator.generate_html(summary, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "accuracy" in content
        assert "latency" in content

    def test_generate_json_preserves_all_fields(self, tmp_path):
        """Test JSON report preserves all result fields."""
        results = [
            EvaluationResult(
                evaluator="test",
                input="test input",
                output="test output",
                expected_output="expected",
                passed=True,
                score=0.95,
                cost=0.001,
                duration_ms=50.0,
                metrics={"custom": "value"},
            )
        ]

        summary = EvaluationSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            pass_rate=1.0,
            avg_score=0.95,
            avg_cost=0.001,
            avg_latency=50.0,
            results=results,
        )

        output_file = tmp_path / "detailed.json"
        ReportGenerator.generate_json(summary, str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        result = data["results"][0]
        assert result["evaluator"] == "test"
        assert result["input"] == "test input"
        assert result["output"] == "test output"
        assert result["expected_output"] == "expected"
        assert result["passed"] is True
        assert result["score"] == 0.95
