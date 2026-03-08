"""Tests for DAG planner."""

from pydantic import BaseModel

from houyi.application.workflow.orchestration.plan import NodeType
from houyi.application.workflow.orchestration.planner import DAGPlanner
from houyi.application.workflow.orchestration.state import SessionState
from houyi.domain.agent import AgentSpec
from houyi.domain.skill.spec import SkillSpec


class TestDAGPlanner:
    """Test DAGPlanner class."""

    def test_plan_simple_llm_only(self):
        """Test planning for agent without skills."""
        planner = DAGPlanner()
        agent = AgentSpec(name="simple_agent", role="assistant", goal="Answer questions")

        plan = planner.plan(task="What is 2+2?", agent=agent)

        assert plan.plan_id.startswith("plan_")
        assert len(plan.nodes) == 1
        assert plan.entry_node == "llm_main"
        assert plan.nodes[0].node_type == NodeType.LLM
        assert plan.nodes[0].node_id == "llm_main"
        assert plan.nodes[0].inputs["task"] == "What is 2+2?"
        assert plan.metadata["num_skills"] == 0

    def test_plan_with_skills(self):
        """Test planning for agent with skills."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="result")

        skill = SkillSpec(
            name="search",
            description="Search skill",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        planner = DAGPlanner()
        agent = AgentSpec(
            name="skilled_agent", role="assistant", goal="Search and answer", skills=[skill]
        )

        plan = planner.plan(task="Search for Python", agent=agent)

        assert plan.plan_id.startswith("plan_")
        # Without LLM, planner uses direct execution (1 node)
        assert len(plan.nodes) >= 1
        assert plan.entry_node.startswith("tool_")

        # Check TOOL node (direct execution)
        tool_node = plan.nodes[0]
        assert tool_node.node_id == "tool_search"
        assert tool_node.node_type == NodeType.TOOL
        assert tool_node.skill_ref == skill
        assert plan.metadata["num_skills"] == 1
        assert plan.metadata.get("direct_execution") is True

    def test_plan_with_multiple_skills(self):
        """Test planning for agent with multiple skills."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="search result")

        def calculate(input: Input) -> Output:
            return Output(result="calc result")

        skill1 = SkillSpec(
            name="search",
            description="Search",
            input_schema=Input,
            output_schema=Output,
            executor=search,
        )

        skill2 = SkillSpec(
            name="calculate",
            description="Calculate",
            input_schema=Input,
            output_schema=Output,
            executor=calculate,
        )

        planner = DAGPlanner()
        agent = AgentSpec(
            name="multi_skilled",
            role="assistant",
            goal="Search and calculate",
            skills=[skill1, skill2],
        )

        plan = planner.plan(task="Find and sum numbers", agent=agent)

        # Without LLM, planner uses direct execution with first skill
        assert len(plan.nodes) >= 1
        assert plan.entry_node.startswith("tool_")
        assert plan.metadata["num_skills"] == 2
        assert plan.metadata.get("direct_execution") is True

        # Check first tool node exists (direct execution uses first skill)
        tool_node = plan.nodes[0]
        assert tool_node.node_type == NodeType.TOOL
        assert tool_node.skill_ref in [skill1, skill2]

    def test_plan_with_session_state(self):
        """Test planning with session state."""
        planner = DAGPlanner()
        agent = AgentSpec(name="agent", role="assistant", goal="Test")

        session_state = SessionState(
            session_id="test_session", agent_id="test_agent", context={"user": "test_user"}
        )

        plan = planner.plan(task="Test task", agent=agent, session_state=session_state)

        assert plan is not None
        assert plan.metadata["task"] == "Test task"
        assert plan.metadata["agent_role"] == "assistant"

    def test_plan_metadata(self):
        """Test plan metadata is correctly set."""
        planner = DAGPlanner()
        agent = AgentSpec(name="test_agent", role="data_analyst", goal="Analyze data")

        plan = planner.plan(task="Analyze sales data", agent=agent)

        assert plan.metadata["task"] == "Analyze sales data"
        assert plan.metadata["agent_role"] == "data_analyst"
        assert plan.metadata["num_skills"] == 0
