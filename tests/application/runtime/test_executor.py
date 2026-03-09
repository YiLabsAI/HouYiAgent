"""Tests for executor."""

import pytest
from pydantic import BaseModel

from houyi import AgentSpec, SkillSpec
from houyi.application.workflow.executor import LocalExecutor
from houyi.application.workflow.orchestration.planner import DAGPlanner
from houyi.application.workflow.orchestration.state import SessionState, TaskStatus


class TestLocalExecutor:
    """Test LocalExecutor."""

    @pytest.mark.asyncio
    async def test_executor_runs_plan(self) -> None:
        """Test that executor can run a simple plan."""

        class Input(BaseModel):
            task: str

        class Output(BaseModel):
            result: str

        def search(task: str) -> Output:
            return Output(result=f"test result:{task}")

        skill = SkillSpec(
            name="search",
            description="Search the web",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = AgentSpec(role="Test Agent", skills=[skill])

        # Create plan
        planner = DAGPlanner()
        plan = planner.plan("Search for HouYi", agent)

        # Create initial state
        initial_state = SessionState(
            session_id="test_session",
            agent_id="test_agent",
        )

        # Execute plan
        executor = LocalExecutor()
        result = await executor.execute(plan, initial_state)

        assert result.status == TaskStatus.SUCCEEDED
        assert result.trace_id.startswith("trace_")
        assert result.metrics.total_duration_ms >= 0
        assert len(result.metrics.node_durations) >= 1

    @pytest.mark.asyncio
    async def test_executor_result_structure(self) -> None:
        """Test that executor returns properly structured results."""

        class Input(BaseModel):
            task: str

        class Output(BaseModel):
            result: str

        def search(task: str) -> Output:
            return Output(result=f"test result:{task}")

        skill = SkillSpec(
            name="search",
            description="Search skill",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = AgentSpec(role="Test Agent", skills=[skill])

        planner = DAGPlanner()
        plan = planner.plan("Search test", agent)

        initial_state = SessionState(
            session_id="test_session",
            agent_id="test_agent",
        )

        executor = LocalExecutor()
        result = await executor.execute(plan, initial_state)

        # Verify result structure
        assert hasattr(result, "task_id")
        assert hasattr(result, "status")
        assert hasattr(result, "output")
        assert hasattr(result, "final_state")
        assert hasattr(result, "metrics")
        assert hasattr(result, "trace_id")
        assert result.task_id.startswith("task_")
        assert result.trace_id.startswith("trace_")

    @pytest.mark.asyncio
    async def test_executor_with_multiple_nodes(self) -> None:
        """Test executor with multiple independent nodes."""

        class Input(BaseModel):
            task: str

        class Output(BaseModel):
            result: str

        def skill1(task: str) -> Output:
            return Output(result=f"skill1:{task}")

        def skill2(task: str) -> Output:
            return Output(result=f"skill2:{task}")

        skill_a = SkillSpec(
            name="skill1",
            description="Double the value",
            input_schema=Input,
            output_schema=Output,
            executor=skill1,
        )

        skill_b = SkillSpec(
            name="skill2",
            description="Add 10 to value",
            input_schema=Input,
            output_schema=Output,
            executor=skill2,
        )

        agent = AgentSpec(role="Test Agent", skills=[skill_a, skill_b])

        planner = DAGPlanner()
        plan = planner.plan("Execute both skills", agent)

        initial_state = SessionState(
            session_id="test_session",
            agent_id="test_agent",
        )

        executor = LocalExecutor()
        result = await executor.execute(plan, initial_state)

        assert result.status == TaskStatus.SUCCEEDED
        assert result.output is not None
        assert len(result.metrics.node_durations) >= 1

    @pytest.mark.asyncio
    async def test_executor_metrics_collection(self) -> None:
        """Test that executor collects execution metrics."""

        class Input(BaseModel):
            task: str

        class Output(BaseModel):
            result: str

        def search(task: str) -> Output:
            return Output(result=f"test result:{task}")

        skill = SkillSpec(
            name="search",
            description="Search skill",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = AgentSpec(role="Test Agent", skills=[skill])

        planner = DAGPlanner()
        plan = planner.plan("Search test", agent)

        initial_state = SessionState(
            session_id="test_session",
            agent_id="test_agent",
        )

        executor = LocalExecutor()
        result = await executor.execute(plan, initial_state)

        # Verify metrics
        assert result.metrics.total_duration_ms >= 0
        assert len(result.metrics.node_durations) > 0
        for node_id, duration in result.metrics.node_durations.items():
            assert duration >= 0
            assert isinstance(node_id, str)

    @pytest.mark.asyncio
    async def test_executor_state_management(self) -> None:
        """Test that executor properly manages session state."""

        class Input(BaseModel):
            task: str

        class Output(BaseModel):
            result: str

        def search(task: str) -> Output:
            return Output(result=f"test result:{task}")

        skill = SkillSpec(
            name="search",
            description="Search skill",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        agent = AgentSpec(role="Test Agent", skills=[skill])

        planner = DAGPlanner()
        plan = planner.plan("Search test", agent)

        initial_state = SessionState(
            session_id="test_session_123",
            agent_id="test_agent_456",
        )

        executor = LocalExecutor()
        result = await executor.execute(plan, initial_state)

        # Verify state is properly maintained
        assert result.final_state.session_id == "test_session_123"
        assert result.final_state.agent_id == "test_agent_456"
        assert result.final_state.current_plan_id == plan.plan_id
        assert result.final_state.parent_state_id == "test_session_123"
