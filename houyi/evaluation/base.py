"""Base classes for evaluation system (Strategy Pattern).

Designed to be extensible for 13+ evaluators (AWS re:Invent 2025 style)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Result of evaluating a single test case."""

    evaluator: str = Field("", description="Name of the evaluator")
    input: str = Field(..., description="Test input")
    output: str = Field(..., description="Actual output")
    expected_output: str | None = Field(default=None, description="Expected output")
    score: float = Field(..., description="Evaluation score (0-1)")
    passed: bool = Field(..., description="Whether the test passed")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Additional metrics")
    feedback: str = Field(default="", description="Evaluation feedback")
    duration_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    cost: float = Field(default=0.0, description="Execution cost in USD")


class EvaluationSummary(BaseModel):
    """Summary of evaluation results across multiple test cases."""

    total_cases: int = Field(..., description="Total number of test cases")
    passed_cases: int = Field(..., description="Number of passed cases")
    failed_cases: int = Field(..., description="Number of failed cases")
    pass_rate: float = Field(..., description="Pass rate (0-1)")
    avg_score: float = Field(..., description="Average score across all cases")
    avg_cost: float = Field(..., description="Average cost per case")
    avg_latency: float = Field(..., description="Average latency in milliseconds")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Aggregated metrics")
    results: list[EvaluationResult] = Field(default_factory=list, description="Individual results")

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""Evaluation Results:
  Total Cases: {self.total_cases}
  Passed: {self.passed_cases} ({self.pass_rate:.1%})
  Failed: {self.failed_cases}
  Avg Score: {self.avg_score:.2f}
  Avg Cost: ${self.avg_cost:.4f}
  Avg Latency: {self.avg_latency:.0f}ms"""

    def save_report(self, path: str, format: str = "html", title: str | None = None) -> None:
        """Save evaluation report to file.

        Args:
            path: Output file path
            format: Report format (html, json, markdown)
            title: Report title (for HTML/Markdown)
        """
        # Lazy import to avoid circular dependency
        from houyi.evaluation.report import ReportGenerator

        if format == "html":
            ReportGenerator.generate_html(self, path, title)
        elif format == "json":
            ReportGenerator.generate_json(self, path)
        elif format == "markdown" or format == "md":
            ReportGenerator.generate_markdown(self, path, title)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'html', 'json', or 'markdown'")


class Evaluator(ABC):
    """Base class for all evaluators (Strategy Pattern).

    Extensible design to support 13+ evaluators:
    - Accuracy, Cost, Latency, SkillUsage (current)
    - Quality, Coherence, Relevance, Completeness (future)
    - Toxicity, Bias, Hallucination, Factuality (future)
    - CustomEvaluator (user-defined)
    """

    @abstractmethod
    def evaluate(
        self,
        input: str,
        output: str,
        expected: str | None = None,
        metadata: dict | None = None,
    ) -> EvaluationResult:
        """Evaluate a single test case.

        Args:
            input: Test input
            output: Actual output
            expected: Expected output (optional)
            metadata: Additional metadata (execution time, cost, etc.)

        Returns:
            EvaluationResult
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Evaluator name."""
        pass
