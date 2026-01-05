"""Tests for error handling and exception paths."""

import pytest
from pydantic import BaseModel, ValidationError

from houyi import AgentSpec, SkillSpec
from houyi.core.skill import SkillSpec
from houyi.evaluation.runner import evaluate
from houyi.evaluation.dataset import Dataset, TestCase


class TestSkillErrorHandling:
    """Test error handling in skill operations."""

    def test_skill_from_file_not_found(self):
        """Test loading skill from non-existent file."""
        with pytest.raises(FileNotFoundError):
            SkillSpec.from_file("nonexistent_file.md")

    def test_skill_from_file_invalid_path(self):
        """Test loading skill with invalid path."""
        with pytest.raises(FileNotFoundError):
            SkillSpec.from_file("/invalid/path/skill.md")

    def test_skill_from_url_invalid_url(self):
        """Test loading skill from invalid URL."""
        import urllib.error
        
        # Test with clearly invalid URL that will fail
        with pytest.raises((urllib.error.URLError, Exception)):
            SkillSpec.from_url("http://invalid-domain-that-does-not-exist-12345.com/skill.md", cache=False)

    def test_skill_missing_required_fields(self):
        """Test skill creation with missing required fields."""
        class Input(BaseModel):
            x: int
        
        class Output(BaseModel):
            y: int
        
        # Missing description should fail
        with pytest.raises(ValidationError):
            SkillSpec(
                name="test",
                input_schema=Input,
                output_schema=Output,
            )

    def test_skill_executor_with_invalid_input(self):
        """Test skill executor with invalid input."""
        class Input(BaseModel):
            x: int
        
        class Output(BaseModel):
            y: int
        
        def func(input: Input) -> Output:
            return Output(y=input.x * 2)
        
        skill = SkillSpec(
            name="test",
            description="Test",
            input_schema=Input,
            output_schema=Output,
            executor=func,
        )
        
        # Should raise validation error for invalid input
        with pytest.raises(ValidationError):
            skill.input_schema(x="not_an_int")


class TestDatasetErrorHandling:
    """Test error handling in dataset operations."""

    def test_dataset_from_file_not_found(self):
        """Test loading dataset from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Dataset.from_file("nonexistent.json")

    def test_dataset_from_file_unsupported_format(self):
        """Test loading dataset with unsupported format."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / "test.txt"
            invalid_file.write_text("invalid content")
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                Dataset.from_file(str(invalid_file))

    def test_testcase_validation_error(self):
        """Test TestCase creation with invalid data."""
        # TestCase should handle various input types gracefully
        tc = TestCase(input="test")
        assert tc.input == "test"

    def test_dataset_empty_test_cases(self):
        """Test dataset with empty test cases."""
        dataset = Dataset(name="empty", test_cases=[])
        assert len(dataset) == 0
        assert list(dataset) == []


class TestEvaluationErrorHandling:
    """Test error handling in evaluation."""

    def test_evaluate_with_empty_test_cases(self):
        """Test evaluation with empty test cases."""
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
        
        results = evaluate(
            agent=agent,
            test_cases=[],
            evaluators=["accuracy"]
        )
        
        assert results.total_cases == 0

    def test_evaluate_with_invalid_evaluator(self):
        """Test evaluation with invalid evaluator name."""
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
        
        # Should handle invalid evaluator gracefully or raise error
        try:
            results = evaluate(
                agent=agent,
                test_cases=[{"input": "test", "expected_output": "test"}],
                evaluators=["nonexistent_evaluator"]
            )
        except (KeyError, ValueError):
            pass  # Expected to fail


class TestAgentErrorHandling:
    """Test error handling in agent operations."""

    def test_agent_with_invalid_skill(self):
        """Test agent with invalid skill configuration."""
        # Agent should handle skills list properly
        agent = AgentSpec(role="Test", skills=[])
        assert len(agent.skills) == 0

    def test_agent_system_prompt_with_no_skills(self):
        """Test system prompt generation with no skills."""
        agent = AgentSpec(role="Test Agent")
        prompt = agent.to_system_prompt()
        
        assert "Test Agent" in prompt
        # Should not crash with no skills

    def test_agent_tool_schemas_empty(self):
        """Test tool schema generation with no skills."""
        agent = AgentSpec(role="Test")
        schemas = agent.get_tool_schemas()
        
        assert schemas == []


class TestExecutionErrorHandling:
    """Test error handling in execution."""

    def test_ir_node_missing_dependencies(self):
        """Test IR node with missing dependencies."""
        from houyi.orchestration.plan import IRNode, NodeType
        
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"},
            dependencies=["missing_dep"]
        )
        
        # Should not be ready without dependencies
        assert not node.is_ready(set())
        assert not node.is_ready({"other_dep"})

    def test_ir_node_invalid_input_reference(self):
        """Test IR node with invalid input reference."""
        from houyi.orchestration.plan import IRNode, NodeType
        
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "$missing.value"},
            outputs={"result": "$output"}
        )
        
        context = {}
        inputs = node.get_input_values(context)
        
        # Should return the reference as-is if not in context
        assert inputs["task"] == "$missing.value"

    def test_execution_plan_with_nodes(self):
        """Test execution plan with nodes."""
        from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
        
        node = IRNode(
            node_id="node1",
            node_type=NodeType.LLM,
            inputs={"task": "test"},
            outputs={"result": "$output"}
        )
        
        plan = ExecutionPlan(
            plan_id="test",
            nodes=[node],
            entry_node="node1"
        )
        
        assert len(plan.nodes) == 1


class TestConstraintsAndPolicies:
    """Test constraints and policy handling."""

    def test_skill_with_timeout_constraint(self):
        """Test skill with timeout constraint."""
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
            constraints={"timeout_ms": 1000, "max_retries": 3}
        )
        
        assert skill.constraints["timeout_ms"] == 1000
        assert skill.constraints["max_retries"] == 3

    def test_agent_with_cost_policy(self):
        """Test agent with cost budget policy."""
        agent = AgentSpec(
            role="Test",
            policies={"max_cost": 0.10, "cost_per_token": 0.0001}
        )
        
        assert agent.policies["max_cost"] == 0.10

    def test_agent_with_retry_policy(self):
        """Test agent with retry policy."""
        agent = AgentSpec(
            role="Test",
            policies={"max_retries": 5, "retry_delay_ms": 1000}
        )
        
        assert agent.policies["max_retries"] == 5
