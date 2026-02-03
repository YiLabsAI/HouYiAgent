"""Tests for evaluation/base.py"""

import pytest

from houyi.evaluation.base import EvaluationResult, EvaluationSummary, Evaluator


class MockEvaluator(Evaluator):
    """Mock evaluator for testing."""

    @property
    def name(self) -> str:
        return "mock_evaluator"

    def evaluate(self, input: str, output: str, expected: str = None, **kwargs) -> EvaluationResult:
        """Mock evaluation."""
        passed = output == expected if expected else True
        return EvaluationResult(
            evaluator=self.name,
            input=input,
            output=output,
            expected_output=expected,
            passed=passed,
            score=1.0 if passed else 0.0,
        )


def test_evaluation_result_creation():
    """Test EvaluationResult creation."""
    result = EvaluationResult(
        evaluator="test_eval",
        input="test input",
        output="test output",
        expected_output="test output",
        passed=True,
        score=0.95,
    )

    assert result.evaluator == "test_eval"
    assert result.passed is True
    assert result.score == 0.95


def test_evaluation_result_failed():
    """Test failed EvaluationResult."""
    result = EvaluationResult(
        evaluator="test_eval",
        input="test input",
        output="wrong output",
        expected_output="correct output",
        passed=False,
        score=0.3,
    )

    assert result.passed is False
    assert result.score == 0.3


def test_evaluation_summary_creation():
    """Test EvaluationSummary creation."""
    summary = EvaluationSummary(
        total_cases=10,
        passed_cases=8,
        failed_cases=2,
        pass_rate=0.8,
        avg_score=0.85,
        avg_cost=0.01,
        avg_latency=100.0,
    )

    assert summary.total_cases == 10
    assert summary.passed_cases == 8
    assert summary.pass_rate == 0.8


def test_evaluator_abstract():
    """Test that Evaluator is abstract."""
    with pytest.raises(TypeError):
        Evaluator()


def test_mock_evaluator():
    """Test MockEvaluator."""
    evaluator = MockEvaluator()

    assert evaluator.name == "mock_evaluator"

    result = evaluator.evaluate(
        input="test input", output="expected output", expected="expected output"
    )

    assert result.passed is True
    assert result.score == 1.0


def test_mock_evaluator_failed():
    """Test MockEvaluator with failed case."""
    evaluator = MockEvaluator()

    result = evaluator.evaluate(
        input="test input", output="wrong output", expected="expected output"
    )

    assert result.passed is False
    assert result.score == 0.0


def test_evaluation_result_with_metadata():
    """Test EvaluationResult with metadata."""
    result = EvaluationResult(
        evaluator="test",
        input="test input",
        output="test output",
        passed=True,
        score=1.0,
        metrics={"latency": 0.5, "tokens": 100},
    )

    assert result.metrics["latency"] == 0.5
    assert result.metrics["tokens"] == 100


def test_evaluation_summary_with_metrics():
    """Test EvaluationSummary with additional metrics."""
    summary = EvaluationSummary(
        total_cases=100,
        passed_cases=85,
        failed_cases=15,
        pass_rate=0.85,
        avg_score=0.87,
        avg_cost=0.02,
        avg_latency=150.0,
        metrics={
            "exact_match": {"passed": 80, "failed": 20},
            "semantic": {"passed": 90, "failed": 10},
        },
    )

    assert summary.metrics["exact_match"]["passed"] == 80
    assert summary.metrics["semantic"]["passed"] == 90
