"""Tests for orchestration layer."""

from pydantic import BaseModel

from houyi import AgentSpec, SkillSpec
from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.planner import DAGPlanner
from houyi.orchestration.state import SessionState, TaskState, TaskStatus


class TestExecutionPlan:
    """Test ExecutionPlan."""

    def test_plan_creation(self) -> None:
        """Test creating an execution plan."""
        nodes = [
            IRNode(
                node_id="node1",
                node_type=NodeType.LLM,
                dependencies=[],
            ),
            IRNode(
                node_id="node2",
                node_type=NodeType.TOOL,
                dependencies=["node1"],
            ),
        ]

        plan = ExecutionPlan(
            plan_id="test_plan",
            nodes=nodes,
            entry_node="node1",
        )

        assert plan.plan_id == "test_plan"
        assert len(plan.nodes) > 0
        assert plan.entry_node is not None

    def test_get_ready_nodes(self) -> None:
        """Test getting ready nodes."""
        nodes = [
            IRNode(node_id="node1", node_type=NodeType.LLM, dependencies=[]),
            IRNode(node_id="node2", node_type=NodeType.TOOL, dependencies=["node1"]),
            IRNode(node_id="node3", node_type=NodeType.VERIFY, dependencies=["node2"]),
        ]

        plan = ExecutionPlan(
            plan_id="test_plan",
            nodes=nodes,
            entry_node="node1",
        )

        # Initially, only node1 is ready
        ready = plan.get_ready_nodes(set())
        assert len(ready) == 1
        assert ready[0].node_id == "node1"

        # After node1 completes, node2 is ready
        ready = plan.get_ready_nodes({"node1"})
        assert len(ready) == 1
        assert ready[0].node_id == "node2"

        # After node1 and node2 complete, node3 is ready
        ready = plan.get_ready_nodes({"node1", "node2"})
        assert len(ready) == 1
        assert ready[0].node_id == "node3"

    def test_plan_with_metadata(self) -> None:
        """Test plan with metadata."""
        nodes = [
            IRNode(node_id="node1", node_type=NodeType.LLM, dependencies=[]),
        ]

        plan = ExecutionPlan(
            plan_id="test_plan",
            nodes=nodes,
            entry_node="node1",
            metadata={"task": "test task", "priority": "high"},
        )

        assert plan.metadata["task"] == "test task"
        assert plan.metadata["priority"] == "high"

    def test_irnode_creation(self) -> None:
        """Test IRNode creation with all fields."""
        node = IRNode(
            node_id="test_node",
            node_type=NodeType.TOOL,
            inputs={"query": "test"},
            outputs={"result": "$result"},
            dependencies=["dep1"],
            metadata={"skill": "search"},
        )

        assert node.node_id == "test_node"
        assert node.node_type == NodeType.TOOL
        assert node.inputs["query"] == "test"
        assert "result" in node.outputs
        assert "dep1" in node.dependencies
        assert node.metadata["skill"] == "search"


class TestDAGPlanner:
    """Test DAGPlanner."""

    def test_planner_creates_plan(self) -> None:
        """Test that planner creates a valid plan."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        def search(input: Input) -> Output:
            return Output(result="test")

        agent = AgentSpec(
            role="Research Assistant",
            skills=[
                SkillSpec(
                    name="search",
                    description="Search the web",
                    input_schema=Input,
                    output_schema=Output,
                    executor=search,
                )
            ],
        )

        planner = DAGPlanner()
        plan = planner.plan("Search for HouYi framework", agent)

        assert plan.plan_id.startswith("plan_")
        # Without LLM, planner uses direct execution (1 node)
        assert len(plan.nodes) >= 1
        assert plan.entry_node.startswith("tool_")
        assert plan.metadata.get("direct_execution") is True


class TestSessionState:
    """Test SessionState."""

    def test_state_creation(self) -> None:
        """Test creating a session state."""
        state = SessionState(
            session_id="session_1",
            agent_id="agent_1",
        )

        assert state.session_id == "session_1"
        assert state.agent_id == "agent_1"
        assert state.current_plan_id is None
        assert len(state.memory_stack) == 0


class TestTaskState:
    """Test TaskState."""

    def test_task_state_creation(self) -> None:
        """Test creating a task state."""
        state = TaskState(
            task_id="task_1",
            status=TaskStatus.PENDING,
        )

        assert state.task_id == "task_1"
        assert state.status == TaskStatus.PENDING
        assert state.retry_count == 0
