"""Error-path tests for evaluation/runner.py."""

import pytest
from pydantic import BaseModel

from houyi import AgentSpec, SkillSpec
from houyi.assurance.evaluation.runner import evaluate


class TestRunnerErrorHandling:
    def test_evaluate_with_empty_test_cases(self):
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )

        agent = AgentSpec(role="Test", skills=[skill])

        results = evaluate(agent=agent, test_cases=[], evaluators=["accuracy"])
        assert results.total_cases == 0

    def test_evaluate_with_invalid_evaluator(self):
        class Input(BaseModel):
            x: int

        class Output(BaseModel):
            y: int

        def func(input: Input) -> Output:
            return Output(y=input.x)

        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )

        agent = AgentSpec(role="Test", skills=[skill])

        with pytest.raises((KeyError, ValueError)):
            evaluate(
                agent=agent,
                test_cases=[{"input": "test", "expected_output": "test"}],
                evaluators=["nonexistent_evaluator"],
            )
