"""Tests for Tool node execution - critical business logic."""

import pytest
from pydantic import BaseModel

from houyi.core.skill import SkillSpec
from houyi.core.agent import AgentSpec
from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.state import SessionState
from houyi.runtime.executor import LocalExecutor


class TestToolNodeExecution:
    """Test Tool node execution in runtime executor."""

    @pytest.mark.asyncio
    async def test_tool_node_with_skill(self) -> None:
        """Test executing a tool node with a real skill."""
        
        # Define skill input/output schemas
        class CalculatorInput(BaseModel):
            expression: str
        
        class CalculatorOutput(BaseModel):
            result: float
        
        # Create a real skill with executor
        def calculator_executor(expression: str) -> CalculatorOutput:
            result = eval(expression)  # Simple calculator
            return CalculatorOutput(result=result)
        
        skill = SkillSpec(
            name="calculator",
            description="Calculate math expressions",
            input_schema=CalculatorInput,
            output_schema=CalculatorOutput,
            executor=calculator_executor
        )
        
        # Create agent with skill
        agent = AgentSpec(
            role="Calculator Agent",
            skills=[skill]
        )
        
        # Create execution plan with tool node
        plan = ExecutionPlan(
            plan_id="test_plan",
            entry_node="calc_node",
            nodes=[
                IRNode(
                    node_id="calc_node",
                    node_type=NodeType.TOOL,
                    skill_ref=skill,
                    inputs={"expression": "2 + 2 * 3"},
                    metadata={"skill": skill}
                )
            ]
        )
        
        # Execute
        executor = LocalExecutor()
        initial_state = SessionState(
            session_id="test_session",
            agent_id="test_agent"
        )
        
        result = await executor.execute(plan, initial_state)
        
        # Verify tool execution
        assert result.status.value == "succeeded"
        assert "calc_node" in result.output
        assert result.output["calc_node"]["type"] == "tool_result"
        assert result.output["calc_node"]["output"].result == 8.0

    @pytest.mark.asyncio
    async def test_tool_node_with_input_from_previous_node(self) -> None:
        """Test tool node receiving input from previous node."""
        
        class Input(BaseModel):
            value: int
        
        class Output(BaseModel):
            doubled: int
        
        def doubler(value: int) -> Output:
            return Output(doubled=value * 2)
        
        skill = SkillSpec(
            name="doubler",
            description="Double a number",
            input_schema=Input,
            output_schema=Output,
            executor=doubler
        )
        
        # Plan: LLM node -> Tool node
        plan = ExecutionPlan(
            plan_id="test_plan",
            entry_node="llm_node",
            nodes=[
                IRNode(
                    node_id="llm_node",
                    node_type=NodeType.LLM,
                    inputs={"task": "generate number"},
                    dependencies=[]
                ),
                IRNode(
                    node_id="tool_node",
                    node_type=NodeType.TOOL,
                    skill_ref=skill,
                    inputs={"value": 5},  # Literal value for testing
                    dependencies=["llm_node"],
                    metadata={"skill": skill}
                )
            ]
        )
        
        executor = LocalExecutor()
        initial_state = SessionState(
            session_id="test_session",
            agent_id="test_agent"
        )
        
        result = await executor.execute(plan, initial_state)
        
        assert result.status.value == "succeeded"
        assert result.output["tool_node"]["output"].doubled == 10

    @pytest.mark.asyncio
    async def test_tool_node_execution_failure(self) -> None:
        """Test tool node handling execution errors."""
        
        class Input(BaseModel):
            value: str
        
        def failing_skill(value: str):
            raise ValueError(f"Intentional error: {value}")
        
        skill = SkillSpec(
            name="failing_skill",
            description="A skill that fails",
            input_schema=Input,
            output_schema=Input,
            executor=failing_skill
        )
        
        plan = ExecutionPlan(
            plan_id="test_plan",
            entry_node="fail_node",
            nodes=[
                IRNode(
                    node_id="fail_node",
                    node_type=NodeType.TOOL,
                    skill_ref=skill,
                    inputs={"value": "test"},
                    metadata={"skill": skill}
                )
            ]
        )
        
        executor = LocalExecutor()
        initial_state = SessionState(
            session_id="test_session",
            agent_id="test_agent"
        )
        
        result = await executor.execute(plan, initial_state)
        
        # Should capture error
        assert result.status.value == "failed"
        assert result.error is not None
        assert "Intentional error" in result.error
