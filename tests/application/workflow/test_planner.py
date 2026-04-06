"""Tests for DAG planner."""

from pydantic import BaseModel

from houyi.application.workflow.orchestration.plan import NodeType
from houyi.application.workflow.orchestration.planner import DAGPlanner
from houyi.application.workflow.orchestration.state import SessionState
from houyi.assurance.verification import VerificationConfig
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

    def test_plan_skills_with_llm(self):
        """Test planning for agent with skills AND LLM produces decide+tool+synthesize."""

        class In(BaseModel):
            query: str

        class Out(BaseModel):
            result: str

        skill = SkillSpec(
            name="search",
            description="S",
            input_schema=In,
            output_schema=Out,
            executor=lambda i: Out(result="r"),
        )

        planner = DAGPlanner()
        agent = AgentSpec(
            name="a",
            role="assistant",
            goal="G",
            skills=[skill],
            policies={"llm": "some_model"},
        )

        plan = planner.plan(task="Find X", agent=agent)

        assert plan.entry_node == "llm_decide"
        node_ids = [n.node_id for n in plan.nodes]
        assert "llm_decide" in node_ids
        assert "tool_search" in node_ids
        assert "llm_synthesize" in node_ids

    def test_verify_node_sql(self):
        """_create_verify_node inserts SQL verification rule."""

        class In(BaseModel):
            q: str

        class Out(BaseModel):
            r: str

        skill = SkillSpec(
            name="sql_runner",
            description="Run SQL",
            input_schema=In,
            output_schema=Out,
            executor=lambda i: Out(r="ok"),
        )

        vc = VerificationConfig(enabled=True)
        planner = DAGPlanner(verification_config=vc)

        agent = AgentSpec(
            name="a",
            role="r",
            goal="g",
            skills=[skill],
            policies={"llm": "model"},
        )
        plan = planner.plan("Run query", agent)

        verify_nodes = [n for n in plan.nodes if n.node_type == NodeType.VERIFY]
        assert len(verify_nodes) == 1
        assert verify_nodes[0].verification_rules
        assert verify_nodes[0].verification_rules[0].verifier_type == "sql"

    def test_verify_node_python(self):
        """_create_verify_node inserts Python verification rule."""

        class In(BaseModel):
            q: str

        class Out(BaseModel):
            r: str

        skill = SkillSpec(
            name="python_exec",
            description="Execute Python",
            input_schema=In,
            output_schema=Out,
            executor=lambda i: Out(r="ok"),
        )

        vc = VerificationConfig(enabled=True)
        planner = DAGPlanner(verification_config=vc)

        agent = AgentSpec(
            name="a",
            role="r",
            goal="g",
            skills=[skill],
            policies={"llm": "model"},
        )
        plan = planner.plan("Run code", agent)

        verify_nodes = [n for n in plan.nodes if n.node_type == NodeType.VERIFY]
        assert len(verify_nodes) == 1
        assert verify_nodes[0].verification_rules[0].verifier_type == "python"

    def test_should_verify_disabled(self):
        """Verification not added when config disabled."""
        vc = VerificationConfig(enabled=False)
        planner = DAGPlanner(verification_config=vc)

        class In(BaseModel):
            q: str

        class Out(BaseModel):
            r: str

        skill = SkillSpec(
            name="sql_runner",
            description="s",
            input_schema=In,
            output_schema=Out,
            executor=lambda i: Out(r="ok"),
        )

        agent = AgentSpec(name="a", role="r", goal="g", skills=[skill], policies={"llm": "m"})
        plan = planner.plan("q", agent)
        verify_nodes = [n for n in plan.nodes if n.node_type == NodeType.VERIFY]
        assert len(verify_nodes) == 0

    def test_direct_exec_with_verify(self):
        """Direct execution path (no LLM) can also include verify."""

        class In(BaseModel):
            q: str

        class Out(BaseModel):
            r: str

        skill = SkillSpec(
            name="sql_tool",
            description="s",
            input_schema=In,
            output_schema=Out,
            executor=lambda i: Out(r="ok"),
        )

        vc = VerificationConfig(enabled=True)
        planner = DAGPlanner(verification_config=vc)
        agent = AgentSpec(name="a", role="r", goal="g", skills=[skill])
        plan = planner.plan("q", agent)

        assert plan.metadata.get("direct_execution") is True
        verify_nodes = [n for n in plan.nodes if n.node_type == NodeType.VERIFY]
        assert len(verify_nodes) == 1
