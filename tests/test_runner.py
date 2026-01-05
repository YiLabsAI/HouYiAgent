"""Tests for evaluation runner (batch evaluation)."""

import pytest
from pydantic import BaseModel
from unittest.mock import patch, AsyncMock

from houyi import AgentSpec, SkillSpec
from houyi.evaluation.runner import evaluate
from houyi.evaluation.evaluators import AccuracyEvaluator, CustomEvaluator, LatencyEvaluator
from houyi.evaluation.base import EvaluationResult
from houyi.evaluation.dataset import Dataset, TestCase
from houyi.runtime.agent import Agent


class TestEvaluate:
    """Test evaluate() function for batch evaluation."""

    def test_evaluate_single_case(self):
        """Test evaluating a single test case."""
        # Simple agent
        class Input(BaseModel):
            query: str
        
        class Output(BaseModel):
            result: str
        
        def search(input: Input) -> Output:
            return Output(result="test result")
        
        skill = SkillSpec(
            name="search",
            description="Search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )
        
        agent = AgentSpec(role="Test Agent", skills=[skill])
        
        # Evaluate
        results = evaluate(
            agent=agent,
            test_cases=[
                {
                    "input": "test query",
                    "expected_output": "test result"
                }
            ],
            evaluators=["accuracy"]
        )
        
        assert results is not None
        assert len(results.results) == 1
        assert results.results[0].evaluator == "accuracy"

    def test_evaluate_multiple_cases(self):
        """Test evaluating multiple test cases."""
        class Input(BaseModel):
            value: int
        
        class Output(BaseModel):
            doubled: int
        
        def doubler(input: Input) -> Output:
            return Output(doubled=input.value * 2)
        
        skill = SkillSpec(
            name="doubler",
            description="Double",
            input_schema=Input,
            output_schema=Output,
            executor=doubler,
        )
        
        agent = AgentSpec(role="Test Agent", skills=[skill])
        
        results = evaluate(
            agent=agent,
            test_cases=[
                {"input": "1", "expected_output": "2"},
                {"input": "5", "expected_output": "10"},
            ],
            evaluators=["accuracy"]
        )
        
        assert len(results.results) >= 2

    def test_evaluate_multiple_evaluators(self):
        """Test using multiple evaluators."""
        class Input(BaseModel):
            query: str
        
        class Output(BaseModel):
            result: str
        
        def search(input: Input) -> Output:
            return Output(result="answer")
        
        skill = SkillSpec(
            name="search",
            description="Search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )
        
        agent = AgentSpec(role="Test Agent", skills=[skill])
        
        results = evaluate(
            agent=agent,
            test_cases=[{"input": "query", "expected_output": "answer"}],
            evaluators=["accuracy", "completeness"]
        )
        
        # Should have results from both evaluators
        evaluator_names = [r.evaluator for r in results.results]
        assert "accuracy" in evaluator_names
        assert "completeness" in evaluator_names

    def test_evaluate_with_evaluator_instance(self):
        """Test using evaluator instance."""
        class Input(BaseModel):
            query: str
        
        class Output(BaseModel):
            result: str
        
        def search(input: Input) -> Output:
            return Output(result="test")
        
        skill = SkillSpec(
            name="search",
            description="Search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )
        
        agent = AgentSpec(role="Test Agent", skills=[skill])
        
        # Use evaluator instance
        accuracy_eval = AccuracyEvaluator()
        
        results = evaluate(
            agent=agent,
            test_cases=[{"input": "query", "expected_output": "test"}],
            evaluators=[accuracy_eval]
        )
        
        assert len(results.results) > 0

    def test_evaluate_summary_stats(self):
        """Test evaluation summary statistics."""
        class Input(BaseModel):
            value: int
        
        class Output(BaseModel):
            result: int
        
        def processor(input: Input) -> Output:
            return Output(result=input.value * 2)
        
        skill = SkillSpec(
            name="processor",
            description="Process",
            input_schema=Input,
            output_schema=Output,
            executor=processor,
        )
        
        agent = AgentSpec(role="Test Agent", skills=[skill])
        
        results = evaluate(
            agent=agent,
            test_cases=[
                {"input": "1", "expected_output": "2"},
                {"input": "2", "expected_output": "4"},
            ],
            evaluators=["accuracy"]
        )
        
        # Check summary exists
        assert results is not None
        assert len(results.results) >= 2

    def test_evaluate_empty_test_cases(self):
        """Test handling empty test cases."""
        class Input(BaseModel):
            query: str
        
        class Output(BaseModel):
            result: str
        
        def search(input: Input) -> Output:
            return Output(result="test")
        
        skill = SkillSpec(
            name="search",
            description="Search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )
        
        agent = AgentSpec(role="Test Agent", skills=[skill])
        
        results = evaluate(
            agent=agent,
            test_cases=[],
            evaluators=["accuracy"]
        )
        
        assert results.total_cases == 0
        assert len(results.results) == 0


def test_evaluate_with_dataset():
    """Test evaluate with Dataset object."""
    # Create a mock agent
    class MockAgent:
        def run(self, input_text):
            return "mocked output"
    
    agent = MockAgent()
    
    dataset = Dataset(
        name="Test Dataset",
        test_cases=[
            TestCase(input="test1", expected_output="output1"),
            TestCase(input="test2", expected_output="output2")
        ]
    )
    
    results = evaluate(
        agent=agent,
        dataset=dataset,
        evaluators=[AccuracyEvaluator()]
    )
    
    assert results.total_cases == 2
    assert len(results.results) == 2


def test_evaluate_with_multiple_evaluators():
    """Test evaluate with multiple evaluators."""
    # Create a mock agent
    class MockAgent:
        def run(self, input_text):
            return "output1"
    
    agent = MockAgent()
    
    test_cases = [
        {"input": "test1", "expected_output": "output1"}
    ]
    
    results = evaluate(
        agent=agent,
        test_cases=test_cases,
        evaluators=[AccuracyEvaluator(), LatencyEvaluator()]
    )
    
    # 1 test case * 2 evaluators = 2 results
    assert results.total_cases == 2
    assert len(results.results) == 2


def test_evaluate_with_string_evaluator():
    """Test evaluate with string evaluator names."""
    # Create a mock agent
    class MockAgent:
        def run(self, input_text):
            return "output1"
    
    agent = MockAgent()
    
    test_cases = [
        {"input": "test1", "expected_output": "output1"}
    ]
    
    results = evaluate(
        agent=agent,
        test_cases=test_cases,
        evaluators=["accuracy", "latency"]
    )
    
    # 1 test case * 2 evaluators = 2 results
    assert results.total_cases == 2
    assert len(results.results) == 2


def test_evaluate_empty_test_cases():
    """Test evaluate with empty test cases."""
    # Create a mock agent
    class MockAgent:
        def run(self, input_text):
            return "output1"
    
    agent = MockAgent()
    
    results = evaluate(
        agent=agent,
        test_cases=[],
        evaluators=[AccuracyEvaluator()]
    )
    
    assert len(results.results) == 0


def test_evaluate_preserves_metadata():
    """Test that evaluate preserves test case metadata."""
    # Create a mock agent
    class MockAgent:
        def run(self, input_text):
            return "output1"
    
    agent = MockAgent()
    
    test_cases = [
        {
            "input": "test1",
            "expected_output": "output1",
            "metadata": {"category": "basic"},
            "expected_skills": ["skill1"]
        }
    ]
    
    results = evaluate(
        agent=agent,
        test_cases=test_cases,
        evaluators=[AccuracyEvaluator()]
    )
    
    assert results.total_cases == 1
    assert len(results.results) == 1


def test_evaluate_records_latency():
    """Test that evaluate records execution latency."""
    # Create a mock agent
    class MockAgent:
        def run(self, input_text):
            return "output1"
    
    agent = MockAgent()
    
    test_cases = [
        {"input": "test1", "expected_output": "output1"}
    ]
    
    results = evaluate(
        agent=agent,
        test_cases=test_cases,
        evaluators=[LatencyEvaluator()]
    )
    
    assert results.total_cases == 1
    assert len(results.results) == 1
    assert results.results[0].duration_ms > 0
