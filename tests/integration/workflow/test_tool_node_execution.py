"""Tests for Tool node execution - critical business logic."""

import pytest
from pydantic import BaseModel

from houyi.application.workflow.executor import LocalExecutor
from houyi.application.workflow.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.application.workflow.orchestration.state import SessionState, TaskStatus
from houyi.domain.skill.spec import SkillSpec


class TestToolNodeExecution:
    """Test Tool node execution in runtime executor."""

    @pytest.mark.asyncio
    async def test_tool_node_with_skill(self) -> None:
        """Test executing a tool node with a real skill."""

        # Define skill input/output schemas
        class CalculatorInput(BaseModel):
            task: str

        class CalculatorOutput(BaseModel):
            result: float

        # Create a real skill with executor
        def calculator_executor(task: str) -> CalculatorOutput:
            result = eval(task)  # Simple calculator
            return CalculatorOutput(result=result)

        skill = SkillSpec(
            name="calculator",
            description="Calculate math expressions",
            input_schema=CalculatorInput,
            output_schema=CalculatorOutput,
            executor=calculator_executor,
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
                    inputs={"task": "2 + 2 * 3"},
                    outputs={"result": "$answer"},
                    metadata={"skill": skill, "direct_execution": True},
                )
            ],
        )

        # Execute
        executor = LocalExecutor()
        initial_state = SessionState(session_id="test_session", agent_id="test_agent")

        result = await executor.execute(plan, initial_state)

        # Verify tool execution
        assert result.status == TaskStatus.SUCCEEDED
        assert result.success is True
        assert result.output["result"] == 8.0

    @pytest.mark.asyncio
    async def test_with_input_from_previous(self) -> None:
        """Test tool node receiving input from previous node."""

        class Input(BaseModel):
            task: str

        class Output(BaseModel):
            result: int

        def doubler(task: str) -> Output:
            return Output(result=int(task) * 2)

        skill = SkillSpec(
            name="doubler",
            description="Double a number",
            input_schema=Input,
            output_schema=Output,
            executor=doubler,
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
                    dependencies=[],
                ),
                IRNode(
                    node_id="tool_node",
                    node_type=NodeType.TOOL,
                    skill_ref=skill,
                    inputs={"task": "5"},
                    outputs={"result": "$answer"},
                    dependencies=["llm_node"],
                    metadata={"skill": skill, "direct_execution": True},
                ),
            ],
        )

        executor = LocalExecutor()
        initial_state = SessionState(session_id="test_session", agent_id="test_agent")

        result = await executor.execute(plan, initial_state)

        assert result.status == TaskStatus.SUCCEEDED
        assert result.success is True
        assert result.output["result"] == 10

    @pytest.mark.asyncio
    async def test_tool_node_execution_failure(self) -> None:
        """Test tool node handling execution errors."""

        class Input(BaseModel):
            task: str

        class Output(BaseModel):
            result: str

        def failing_skill(task: str):
            raise ValueError(f"Intentional error: {task}")

        skill = SkillSpec(
            name="failing_skill",
            description="A skill that fails",
            input_schema=Input,
            output_schema=Output,
            executor=failing_skill,
        )

        plan = ExecutionPlan(
            plan_id="test_plan",
            entry_node="fail_node",
            nodes=[
                IRNode(
                    node_id="fail_node",
                    node_type=NodeType.TOOL,
                    skill_ref=skill,
                    inputs={"task": "test"},
                    outputs={"result": "$answer"},
                    metadata={"skill": skill, "direct_execution": True},
                )
            ],
        )

        executor = LocalExecutor()
        initial_state = SessionState(session_id="test_session", agent_id="test_agent")

        result = await executor.execute(plan, initial_state)

        # Should capture error
        assert result.status == TaskStatus.FAILED
        assert result.success is False
        assert result.error is not None
        assert "Intentional error" in result.error
