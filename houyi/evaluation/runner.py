"""Evaluation runner for batch evaluation."""

from __future__ import annotations

from typing import Any

from houyi.evaluation.base import EvaluationSummary, Evaluator
from houyi.evaluation.dataset import Dataset
from houyi.evaluation.evaluators import (
    # Phase 1
    AccuracyEvaluator,
    # Phase 3
    BiasEvaluator,
    CoherenceEvaluator,
    # Phase 2
    CompletenessEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    CostEvaluator,
    CustomEvaluator,
    FactualityEvaluator,
    FaithfulnessEvaluator,
    GroundednessEvaluator,
    HallucinationEvaluator,
    LatencyEvaluator,
    LLMJudgeEvaluator,
    RelevanceEvaluator,
    SafetyEvaluator,
    SemanticSimilarityEvaluator,
    SkillUsageEvaluator,
    ToxicityEvaluator,
)


def _get_evaluator(name_or_instance: str | Evaluator) -> Evaluator:
    """Get evaluator instance from name or return instance directly."""
    if isinstance(name_or_instance, Evaluator):
        return name_or_instance

    # Map evaluator names to classes (19 evaluators total)
    evaluators = {
        # Phase 1: Core (4)
        "accuracy": AccuracyEvaluator,
        "cost": CostEvaluator,
        "latency": LatencyEvaluator,
        "skill_usage": SkillUsageEvaluator,
        # Phase 2: Quality & Safety (6)
        "completeness": CompletenessEvaluator,
        "relevance": RelevanceEvaluator,
        "toxicity": ToxicityEvaluator,
        "hallucination": HallucinationEvaluator,
        "semantic_similarity": SemanticSimilarityEvaluator,
        "llm_judge": LLMJudgeEvaluator,
        # Phase 3: Advanced (9)
        "bias": BiasEvaluator,
        "safety": SafetyEvaluator,
        "factuality": FactualityEvaluator,
        "groundedness": GroundednessEvaluator,
        "coherence": CoherenceEvaluator,
        "context_precision": ContextPrecisionEvaluator,
        "context_recall": ContextRecallEvaluator,
        "faithfulness": FaithfulnessEvaluator,
        "custom": CustomEvaluator,
    }

    evaluator_class = evaluators.get(name_or_instance.lower())
    if not evaluator_class:
        raise ValueError(f"Unknown evaluator: {name_or_instance}. Available: {', '.join(evaluators.keys())}")

    return evaluator_class()


def evaluate(
    agent: Any,
    test_cases: list[dict] | Dataset = None,
    evaluators: list[str] | list[Evaluator] = None,
    dataset: Dataset | None = None,
) -> EvaluationSummary:
    """Evaluate agent performance on test cases.

    Args:
        agent: Agent to evaluate
        test_cases: List of test cases (dicts) or Dataset instance (deprecated, use dataset param)
        evaluators: List of evaluator names or Evaluator instances
        dataset: Dataset instance (preferred over test_cases)

    Returns:
        EvaluationSummary

    Example:
        >>> from houyi.evaluation import evaluate, Dataset
        >>> dataset = Dataset.from_file("tests/dataset.json")
        >>> results = evaluate(agent, dataset=dataset, evaluators=["accuracy", "coherence"])
        >>> results.save_report("report.html")
    """
    # Handle dataset parameter
    if dataset is not None:
        from houyi.evaluation.dataset import Dataset
        if isinstance(dataset, Dataset):
            test_cases = [case.model_dump() for case in dataset.test_cases]
    elif test_cases is not None:
        # Check if test_cases is actually a Dataset
        from houyi.evaluation.dataset import Dataset
        if isinstance(test_cases, Dataset):
            test_cases = [case.model_dump() for case in test_cases.test_cases]
    if evaluators is None:
        evaluators = ["accuracy", "cost", "latency"]

    # Convert string evaluator names to instances
    evaluator_instances = [_get_evaluator(ev) for ev in evaluators]

    # Run evaluation
    all_results = []
    total_cost = 0.0
    total_latency = 0.0

    for test_case in test_cases:
        input_text = test_case["input"]
        expected = test_case.get("expected_output")

        # Execute agent and collect metrics
        import time
        start_time = time.time()

        try:
            output = agent.run(input_text)
            duration_ms = (time.time() - start_time) * 1000

            # Extract metadata from agent execution
            metadata = {
                "cost": getattr(agent, '_last_cost', 0.0),
                "duration_ms": duration_ms,
                "used_skills": getattr(agent, '_used_skills', []),
                "expected_skills": test_case.get("expected_skills", []),
            }
        except Exception as e:
            print(f"Agent execution failed: {e}")
            output = f"Error: {e}"
            metadata = {
                "cost": 0.0,
                "duration_ms": (time.time() - start_time) * 1000,
                "used_skills": [],
                "expected_skills": test_case.get("expected_skills", []),
                "error": str(e),
            }

        # Evaluate with each evaluator
        for evaluator in evaluator_instances:
            result = evaluator.evaluate(input_text, str(output), expected, metadata)
            all_results.append(result)
            total_cost += result.cost
            total_latency += result.duration_ms

    # Calculate summary
    passed_count = sum(1 for r in all_results if r.passed)
    total_count = len(all_results)

    return EvaluationSummary(
        total_cases=total_count,
        passed_cases=passed_count,
        failed_cases=total_count - passed_count,
        pass_rate=passed_count / total_count if total_count > 0 else 0.0,
        avg_score=sum(r.score for r in all_results) / total_count if total_count > 0 else 0.0,
        avg_cost=total_cost / len(test_cases) if test_cases else 0.0,
        avg_latency=total_latency / len(test_cases) if test_cases else 0.0,
        results=all_results,
    )
