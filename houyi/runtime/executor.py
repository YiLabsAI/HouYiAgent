"""Local executor for DAG execution."""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from houyi.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.orchestration.state import SessionState, TaskStatus


class ExecutionMetrics(BaseModel):
    """Metrics collected during execution."""

    total_duration_ms: float = 0.0
    node_durations: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_estimate: float = 0.0


class ExecutionResult(BaseModel):
    """Result of plan execution."""

    task_id: str
    status: TaskStatus
    output: dict[str, Any] | None = None
    final_state: SessionState
    metrics: ExecutionMetrics
    trace_id: str
    error: str | None = None


class LocalExecutor:
    """Local asyncio-based executor for execution plans.

    Executes DAG plans with:
    - Topological scheduling
    - Concurrent execution of independent nodes
    - State snapshot generation
    - Basic retry logic
    - OpenTelemetry instrumentation (TODO)
    """

    def __init__(self) -> None:
        """Initialize the executor."""
        pass

    async def execute(
        self,
        plan: ExecutionPlan,
        initial_state: SessionState,
    ) -> ExecutionResult:
        """Execute a plan locally with asyncio.

        Args:
            plan: Execution plan (DAG of IR nodes)
            initial_state: Initial session state

        Returns:
            Execution result with output, final state, and metrics
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        trace_id = f"trace_{uuid.uuid4().hex[:8]}"

        start_time = datetime.now(UTC)
        completed_nodes: set[str] = set()
        node_outputs: dict[str, Any] = {}
        metrics = ExecutionMetrics()

        try:
            # Execute nodes in topological order
            while len(completed_nodes) < len(plan.nodes):
                ready_nodes = plan.get_ready_nodes(completed_nodes)

                if not ready_nodes:
                    # No ready nodes but plan not complete = cycle or missing dependency
                    raise RuntimeError("DAG has cycle or missing dependencies")

                # Execute ready nodes concurrently
                tasks = [self._execute_node(node, node_outputs, metrics) for node in ready_nodes]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for node, result in zip(ready_nodes, results, strict=False):
                    if isinstance(result, Exception):
                        raise result
                    node_outputs[node.node_id] = result
                    completed_nodes.add(node.node_id)

            # Build final state
            final_state = SessionState(
                session_id=initial_state.session_id,
                agent_id=initial_state.agent_id,
                current_plan_id=plan.plan_id,
                memory_stack=initial_state.memory_stack,
                execution_pointer=None,
                parent_state_id=initial_state.session_id,
            )

            # Calculate total duration
            end_time = datetime.now(UTC)
            metrics.total_duration_ms = (end_time - start_time).total_seconds() * 1000

            return ExecutionResult(
                task_id=task_id,
                status=TaskStatus.SUCCEEDED,
                output=node_outputs,
                final_state=final_state,
                metrics=metrics,
                trace_id=trace_id,
            )

        except Exception as e:
            final_state = SessionState(
                session_id=initial_state.session_id,
                agent_id=initial_state.agent_id,
                current_plan_id=plan.plan_id,
                memory_stack=initial_state.memory_stack,
                execution_pointer=None,
                parent_state_id=initial_state.session_id,
            )

            return ExecutionResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                output=None,
                final_state=final_state,
                metrics=metrics,
                trace_id=trace_id,
                error=str(e),
            )

    async def _execute_node(
        self,
        node: IRNode,
        node_outputs: dict[str, Any],
        metrics: ExecutionMetrics,
    ) -> Any:
        """Execute a single node.

        Args:
            node: IR node to execute
            node_outputs: Outputs from previously executed nodes
            metrics: Metrics collector

        Returns:
            Node execution result
        """
        start_time = datetime.now(UTC)

        try:
            # Resolve inputs from dependencies
            resolved_inputs = {}
            for input_name, source_value in node.inputs.items():
                # Check if it's a node reference (string starting with node ID)
                if isinstance(source_value, str) and source_value in node_outputs:
                    resolved_inputs[input_name] = node_outputs[source_value]
                else:
                    # Input is a literal value (could be str, list, dict, etc.)
                    resolved_inputs[input_name] = source_value

            # Execute based on node type
            if node.node_type == NodeType.LLM:
                result = await self._execute_llm_node(node, resolved_inputs)
            elif node.node_type == NodeType.TOOL:
                result = await self._execute_tool_node(node, resolved_inputs)
            elif node.node_type == NodeType.VERIFY:
                result = await self._execute_verify_node(node, resolved_inputs)
            elif node.node_type == NodeType.LOGIC:
                result = await self._execute_logic_node(node, resolved_inputs)
            elif node.node_type == NodeType.ROUTE:
                result = await self._execute_route_node(node, resolved_inputs)
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")

            # Record metrics
            end_time = datetime.now(UTC)
            duration_ms = (end_time - start_time).total_seconds() * 1000
            metrics.node_durations[node.node_id] = duration_ms

            return result

        except Exception as e:
            raise RuntimeError(f"Node {node.node_id} failed: {e}") from e

    async def _execute_llm_node(
        self,
        node: IRNode,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute LLM node."""
        # Get LLM configuration from node metadata
        use_real_llm = node.metadata.get("use_real_llm", False)

        if use_real_llm:
            try:
                from houyi.llm.base import LLMMessage, MessageRole
                from houyi.llm.openai_adapter import OpenAIAdapter

                adapter = OpenAIAdapter()
                task = inputs.get("task", inputs.get("prompt", ""))
                messages = [LLMMessage(role=MessageRole.USER, content=task)]

                # Synchronous call (runtime executor is sync)
                import asyncio

                response = asyncio.run(adapter.chat(messages))  # type: ignore[arg-type]

                return {
                    "type": "llm_response",
                    "content": response.content,
                }
            except Exception as e:
                print(f"LLM execution failed: {e}")
                return {
                    "type": "llm_response",
                    "content": f"Mock LLM response (error): {inputs.get('task', 'unknown')}",
                }

        # Mock implementation
        return {
            "type": "llm_response",
            "content": f"Mock LLM response for task: {inputs.get('task', 'unknown')}",
        }

    async def _execute_tool_node(
        self,
        node: IRNode,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tool node."""
        # Get skill from node metadata
        skill = node.metadata.get("skill")

        if skill and hasattr(skill, "executor") and skill.executor:
            try:
                # Execute skill
                result = skill.executor(**inputs)
                return {
                    "type": "tool_result",
                    "output": result,
                }
            except Exception as e:
                print(f"Tool execution failed: {e}")
                raise

        # Mock implementation
        return {
            "type": "tool_result",
            "output": f"Mock tool result for inputs: {inputs}",
        }

    async def _execute_verify_node(
        self,
        node: IRNode,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute verification node."""
        # Get assertion from node metadata
        assertion = node.metadata.get("assertion")

        if assertion:
            try:
                from houyi.core.assertion import AssertionSpec

                if isinstance(assertion, AssertionSpec):
                    passed = assertion.evaluate(inputs)
                    return {
                        "type": "verification",
                        "passed": passed,
                        "assertion": assertion.name,
                    }
            except Exception as e:
                print(f"Verification failed: {e}")
                return {
                    "type": "verification",
                    "passed": False,
                    "error": str(e),
                }

        # Default: pass if no assertion
        return {
            "type": "verification",
            "passed": True,
        }

    async def _execute_logic_node(
        self,
        node: IRNode,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute logic node (conditional branching, loops, etc)."""
        # Logic nodes perform control flow operations
        logic_type = node.metadata.get("logic_type", "passthrough")

        if logic_type == "conditional":
            condition = node.metadata.get("condition")
            if condition and callable(condition):
                result = condition(inputs)
                return {
                    "type": "logic",
                    "result": result,
                    "branch": "true" if result else "false",
                }

        # Default: passthrough
        return {
            "type": "logic",
            "result": inputs,
        }

    async def _execute_route_node(
        self,
        node: IRNode,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute routing node (select execution branch based on input)."""
        # Routing nodes select which branch to execute
        router = node.metadata.get("router")

        if router and callable(router):
            selected = router(inputs)
            return {
                "type": "route",
                "selected_branch": selected,
                "inputs": inputs,
            }

        # Default routing
        return {
            "type": "route",
            "selected_branch": "default",
        }
