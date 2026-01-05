"""Report generation for evaluation results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from houyi.evaluation.base import EvaluationSummary


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    @staticmethod
    def generate_html(
        summary: EvaluationSummary, output_path: str, title: str | None = None
    ) -> None:
        """Generate HTML report from evaluation summary.

        Args:
            summary: Evaluation summary
            output_path: Path to save HTML report
            title: Report title (default: "Evaluation Report")
        """
        title = title or "Evaluation Report"

        # Group results by evaluator
        results_by_evaluator = {}
        for result in summary.results:
            evaluator = result.evaluator
            if evaluator not in results_by_evaluator:
                results_by_evaluator[evaluator] = []
            results_by_evaluator[evaluator].append(result)

        # Calculate per-evaluator stats
        evaluator_stats = {}
        for evaluator, results in results_by_evaluator.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            avg_score = sum(r.score for r in results) / total if total > 0 else 0.0
            evaluator_stats[evaluator] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0.0,
                "avg_score": avg_score,
            }

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2em;
        }}

        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .metric {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}

        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .pass {{ color: #27ae60; }}
        .fail {{ color: #e74c3c; }}

        h2 {{
            color: #2c3e50;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}

        .evaluator-section {{
            margin-bottom: 30px;
        }}

        .evaluator-header {{
            background: #3498db;
            color: white;
            padding: 15px 20px;
            border-radius: 6px 6px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .evaluator-name {{
            font-size: 1.2em;
            font-weight: bold;
        }}

        .evaluator-stats {{
            font-size: 0.9em;
        }}

        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}

        .results-table th {{
            background: #ecf0f1;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #2c3e50;
        }}

        .results-table td {{
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }}

        .results-table tr:hover {{
            background: #f8f9fa;
        }}

        .score-bar {{
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
        }}

        .score-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            transition: width 0.3s ease;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-pass {{
            background: #d4edda;
            color: #155724;
        }}

        .badge-fail {{
            background: #f8d7da;
            color: #721c24;
        }}

        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{summary.total_cases}</div>
                <div class="metric-label">Total Cases</div>
            </div>
            <div class="metric">
                <div class="metric-value pass">{summary.passed_cases}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value fail">{summary.failed_cases}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.pass_rate:.1%}</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.avg_score:.2f}</div>
                <div class="metric-label">Avg Score</div>
            </div>
        </div>

        <h2>Results by Evaluator</h2>
"""

        # Add evaluator sections
        for evaluator, results in results_by_evaluator.items():
            stats = evaluator_stats[evaluator]
            html += f"""
        <div class="evaluator-section">
            <div class="evaluator-header">
                <div class="evaluator-name">{evaluator}</div>
                <div class="evaluator-stats">
                    {stats["passed"]}/{stats["total"]} passed ({stats["pass_rate"]:.1%}) |
                    Avg Score: {stats["avg_score"]:.2f}
                </div>
            </div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th style="width: 40%">Input</th>
                        <th style="width: 20%">Score</th>
                        <th style="width: 15%">Status</th>
                        <th style="width: 25%">Feedback</th>
                    </tr>
                </thead>
                <tbody>
"""

            for result in results:
                status_badge = (
                    '<span class="badge badge-pass">PASS</span>'
                    if result.passed
                    else '<span class="badge badge-fail">FAIL</span>'
                )
                input_preview = (
                    result.input[:100] + "..." if len(result.input) > 100 else result.input
                )

                html += f"""
                    <tr>
                        <td>{input_preview}</td>
                        <td>
                            <div class="score-bar">
                                <div class="score-fill" style="width: {result.score * 100}%"></div>
                            </div>
                            <div style="margin-top: 5px; font-size: 0.9em;">{result.score:.2%}</div>
                        </td>
                        <td>{status_badge}</td>
                        <td style="font-size: 0.9em;">{result.feedback}</td>
                    </tr>
"""

            html += """
                </tbody>
            </table>
        </div>
"""

        # Footer
        html += f"""
        <div class="footer">
            Generated by HouYi Evaluation System |
            {len(results_by_evaluator)} evaluators |
            {summary.total_cases} test cases
        </div>
    </div>
</body>
</html>
"""

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")

    @staticmethod
    def generate_json(summary: EvaluationSummary, output_path: str) -> None:
        """Generate JSON report from evaluation summary.

        Args:
            summary: Evaluation summary
            output_path: Path to save JSON report
        """
        # Convert summary to dict
        data = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_cases": summary.total_cases,
                "passed_cases": summary.passed_cases,
                "failed_cases": summary.failed_cases,
                "pass_rate": summary.pass_rate,
                "avg_score": summary.avg_score,
                "avg_cost": summary.avg_cost,
                "avg_latency": summary.avg_latency,
            },
            "results": [
                {
                    "evaluator": r.evaluator,
                    "input": r.input,
                    "output": r.output,
                    "expected_output": r.expected_output,
                    "score": r.score,
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "feedback": r.feedback,
                    "cost": r.cost,
                    "duration_ms": r.duration_ms,
                }
                for r in summary.results
            ],
        }

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_markdown(
        summary: EvaluationSummary, output_path: str, title: str | None = None
    ) -> None:
        """Generate Markdown report from evaluation summary.

        Args:
            summary: Evaluation summary
            output_path: Path to save Markdown report
            title: Report title (default: "Evaluation Report")
        """
        title = title or "Evaluation Report"

        # Group results by evaluator
        results_by_evaluator = {}
        for result in summary.results:
            evaluator = result.evaluator
            if evaluator not in results_by_evaluator:
                results_by_evaluator[evaluator] = []
            results_by_evaluator[evaluator].append(result)

        # Generate Markdown
        md = f"""# {title}

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| Metric | Value |
|--------|-------|
| Total Cases | {summary.total_cases} |
| Passed | {summary.passed_cases} |
| Failed | {summary.failed_cases} |
| Pass Rate | {summary.pass_rate:.1%} |
| Avg Score | {summary.avg_score:.2f} |
| Avg Cost | ${summary.avg_cost:.4f} |
| Avg Latency | {summary.avg_latency:.0f}ms |

## Results by Evaluator

"""

        # Add evaluator sections
        for evaluator, results in results_by_evaluator.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            avg_score = sum(r.score for r in results) / total if total > 0 else 0.0

            md += f"""### {evaluator}

**Stats**: {passed}/{total} passed ({passed / total:.1%}) | Avg Score: {avg_score:.2f}

| Input | Score | Status | Feedback |
|-------|-------|--------|----------|
"""

            for result in results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                input_preview = (
                    result.input[:50] + "..." if len(result.input) > 50 else result.input
                )
                md += f"| {input_preview} | {result.score:.2%} | {status} | {result.feedback} |\n"

            md += "\n"

        # Footer
        md += f"""---

*Generated by HouYi Evaluation System | {len(results_by_evaluator)} evaluators | {summary.total_cases} test cases*
"""

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(md, encoding="utf-8")
