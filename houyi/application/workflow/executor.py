"""Local workflow executor with DAG execution support."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from houyi.application.workflow.orchestration.plan import ExecutionPlan, IRNode, NodeType
from houyi.application.workflow.orchestration.state import SessionState, TaskStatus
from houyi.application.workflow.skill_executor import SkillExecutor
from houyi.domain.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(UTC)


class ExecutionMetrics(BaseModel):
    """Metrics collected during execution."""

    total_duration_ms: float = 0.0
    node_durations: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_estimate: float = 0.0


class ExecutionResult(BaseModel):
    """Result of execution."""

    success: bool
    output: Any
    final_state: SessionState
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus | None = None
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    task_id: str = ""
    trace_id: str = ""
    error: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.status is None:
            self.status = TaskStatus.SUCCEEDED if self.success else TaskStatus.FAILED


class LocalExecutor:
    """Local executor with DAG execution support.

    Executes ExecutionPlan using topological sort and async concurrency.
    """

    def __init__(self, trace_manager: Any = None):
        self.context: dict[str, Any] = {}
        self.trace_manager = trace_manager

    async def execute(self, plan: ExecutionPlan, initial_state: SessionState) -> ExecutionResult:
        """Execute the plan.

        Args:
            plan: Execution plan (DAG)
            initial_state: Initial session state

        Returns:
            ExecutionResult with output and final state
        """
        completed_node_ids: set[str] = set()
        self.context = {"task": plan.metadata.get("task", "")}
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        trace_id = f"trace_{uuid.uuid4().hex[:8]}"
        started_at = _now_utc()
        metrics = ExecutionMetrics()

        while len(completed_node_ids) < len(plan.nodes):
            ready_nodes = plan.get_ready_nodes(completed_node_ids)

            if not ready_nodes:
                if len(completed_node_ids) < len(plan.nodes):
                    raise RuntimeError(
                        f"Circular dependency detected. "
                        f"Completed: {len(completed_node_ids)}/{len(plan.nodes)}"
                    )
                break

            results = await asyncio.gather(
                *[self._execute_node(node, self.context, metrics) for node in ready_nodes],
                return_exceptions=True,
            )

            node_error = self._extract_node_error(ready_nodes, results)
            if node_error is not None:
                metrics.total_duration_ms = self._duration_ms(started_at)
                return self._build_failure_result(
                    initial_state=initial_state,
                    plan=plan,
                    completed_node_ids=completed_node_ids,
                    metrics=metrics,
                    task_id=task_id,
                    trace_id=trace_id,
                    error=node_error,
                )

            for node, result in zip(ready_nodes, results, strict=False):
                result_dict = result if isinstance(result, dict) else {}
                for output_key, var_name in node.outputs.items():
                    if var_name.startswith("$"):
                        self.context[var_name[1:]] = result_dict.get(output_key)

                completed_node_ids.add(node.node_id)

        metrics.total_duration_ms = self._duration_ms(started_at)
        updated_state = self._build_completed_state(initial_state, plan)
        final_output = self.context.get("answer", self.context)

        return ExecutionResult(
            success=True,
            status=TaskStatus.SUCCEEDED,
            output=final_output,
            final_state=updated_state,
            metadata={
                "nodes_executed": len(completed_node_ids),
                "context": self.context,
            },
            metrics=metrics,
            task_id=task_id,
            trace_id=trace_id,
        )

    async def _execute_node(
        self,
        node: IRNode,
        context: dict[str, Any],
        metrics: ExecutionMetrics,
    ) -> dict[str, Any]:
        """Execute a single node.

        Args:
            node: IR node to execute
            context: Execution context

        Returns:
            Node execution result
        """
        started_at = _now_utc()
        inputs = node.get_input_values(context)

        try:
            if self.trace_manager:
                span_name = f"node.{node.node_type.value}"
                with self.trace_manager.start_span(
                    span_name,
                    attributes={
                        "node.id": node.node_id,
                        "node.type": node.node_type.value,
                    },
                ) as span:
                    result = await self._execute_node_impl(node, inputs)
                    span.set_attribute("node.success", True)
            else:
                result = await self._execute_node_impl(node, inputs)
        except Exception as exc:
            if self.trace_manager:
                logger.exception("Node execution failed: %s", node.node_id)
            raise RuntimeError(f"Node {node.node_id} failed: {exc}") from exc

        metrics.node_durations[node.node_id] = self._duration_ms(started_at)
        return result

    async def _execute_node_impl(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute node implementation.

        Args:
            node: IR node to execute
            inputs: Resolved inputs

        Returns:
            Node execution result
        """
        if node.node_type == NodeType.LLM:
            return await self._execute_llm_node(node, inputs)
        elif node.node_type == NodeType.TOOL:
            return await self._execute_tool_node(node, inputs)
        elif node.node_type == NodeType.VERIFY:
            return await self._execute_verify_node(node, inputs)
        elif node.node_type == NodeType.AGENT:
            return await self._execute_agent_node(node, inputs)
        else:
            raise ValueError(f"Unsupported node type: {node.node_type}")

    async def _execute_llm_node(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute LLM node.

        Args:
            node: LLM node
            inputs: Resolved inputs

        Returns:
            LLM response
        """
        # Get LLM adapter from node metadata or use default
        use_real_llm = node.metadata.get("use_real_llm", False)
        node.metadata.get("model", "gpt-4")

        if use_real_llm:
            # Use real LLM adapter
            from houyi.adapters.llm.base import LLMMessage, MessageRole
            from houyi.adapters.llm.openai_adapter import OpenAIAdapter

            try:
                adapter = OpenAIAdapter()

                # Build messages
                task = inputs.get("task", "")
                messages = [LLMMessage(role=MessageRole.USER, content=task)]

                # Call LLM
                response = await adapter.chat(messages)  # type: ignore[arg-type]

                return {"answer": response.content}
            except Exception as e:
                logger.warning("LLM call failed (%s), using mock response", e)
                return {"answer": f"Mock LLM response (LLM unavailable): {inputs.get('task', '')}"}

        # Mock implementation when LLM is not configured
        task = inputs.get("task", inputs.get("prompt", ""))
        purpose = node.metadata.get("purpose", "reasoning")

        return {"answer": f"Mock {purpose} response: {task}"}

    async def _execute_tool_node(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute TOOL node.

        Args:
            node: TOOL node
            inputs: Resolved inputs

        Returns:
            Tool execution result
        """
        if not node.skill_ref:
            raise ValueError(f"TOOL node {node.node_id} has no skill_ref")

        skill = node.skill_ref

        # Use SkillExecutor if skill has executor
        if skill.executor:
            executor = SkillExecutor()

            # Check if this is direct execution mode (no LLM)
            is_direct_mode = node.metadata.get("direct_execution", False)

            if is_direct_mode:
                task_input = inputs.get("task", "")
                nested_params = inputs.get("params", {})
                if not task_input and isinstance(nested_params, dict):
                    nested_task = nested_params.get("task", "")
                    task_input = nested_task if isinstance(nested_task, str) else str(nested_task)
                params = self._extract_params_from_task(task_input, skill)
                if not params and isinstance(nested_params, dict):
                    params = nested_params
            else:
                # Use params from LLM decision
                params = inputs.get("params", {})

            try:
                result = await executor.execute(skill, params)
                return {"result": result}
            except Exception as e:
                logger.warning("Skill execution failed (%s)", e)
                raise

        # Placeholder implementation
        return {"result": f"Result from {skill.name}"}

    async def _execute_agent_node(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute AGENT node by spawning a sub-agent via SubAgentManager.

        Requires ``agent_id`` on the node and an ``AgentRegistry`` available
        via ``self.agent_registry``.  Falls back to a mock result when the
        registry is not configured.
        """
        from houyi.application.runtime.sub_agent import SubAgentManager
        from houyi.domain.agent.spec import AgentSpec

        task_input = inputs.get("task", "")
        agent_id = node.agent_id or node.metadata.get("agent_id", "")

        registry = getattr(self, "agent_registry", None)
        if registry is not None:
            config = registry.get(agent_id)
            if config and config.default_spec:
                spec = config.default_spec
            else:
                spec = AgentSpec(role=agent_id or "sub_agent")
        else:
            spec = AgentSpec(role=agent_id or "sub_agent")

        mgr = SubAgentManager()
        handle = await mgr.spawn(spec, task_input)
        result = await mgr.join(handle)

        output: dict[str, Any] = {
            "result": result.output,
            "agent_id": result.agent_id,
            "success": result.success,
        }

        if node.handoff_to:
            output["handoff_to"] = node.handoff_to

        return output

    def _extract_params_from_task(self, task: str, skill: SkillSpec) -> dict[str, Any]:
        """Extract skill parameters from task string.

        Simple heuristic-based parameter extraction for fallback mode.

        Args:
            task: Task description string
            skill: Skill specification

        Returns:
            Dictionary of extracted parameters
        """
        import inspect
        from typing import get_type_hints

        params = {}

        # Get skill input schema fields
        if hasattr(skill.input_schema, "model_fields"):
            fields = skill.input_schema.model_fields

            # Simple heuristic: use task as first string parameter
            for field_name, field_info in fields.items():
                field_type = field_info.annotation

                # Handle string parameters
                if field_type is str or str(field_type) == "<class 'str'>":
                    params[field_name] = task
                    break
                # Handle list parameters
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:  # type: ignore[union-attr]
                    params[field_name] = [task]  # type: ignore[assignment]
                    break

        # If no params extracted, try using original function signature
        if not params and hasattr(skill, "_original_func"):
            sig = inspect.signature(skill._original_func)
            hints = get_type_hints(skill._original_func)

            for param_name, _param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = hints.get(param_name, str)

                if param_type is str:
                    params[param_name] = task
                    break
                if hasattr(param_type, "__origin__") and param_type.__origin__ is list:
                    params[param_name] = [task]  # type: ignore[assignment]
                    break

        return params

    async def _execute_verify_node(self, node: IRNode, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute VERIFY node.

        Args:
            node: VERIFY node
            inputs: Resolved inputs

        Returns:
            Verification result
        """
        from houyi.assurance.verification import ConstraintChecker, PythonVerifier, SQLVerifier

        if not node.verification_rules:
            return {"verified": True}

        # Get output to verify
        output = inputs.get("output")
        if output is None:
            return {"verified": False, "error": "No output to verify"}

        # Run all verification rules
        all_passed = True
        errors = []

        for rule in node.verification_rules:
            # Select verifier based on type
            verifier: SQLVerifier | PythonVerifier | ConstraintChecker | None = None
            if rule.verifier_type == "sql":
                verifier = SQLVerifier()
            elif rule.verifier_type == "python":
                verifier = PythonVerifier()
            elif rule.verifier_type == "constraint":
                verifier = ConstraintChecker()

            if verifier:
                result = await verifier.verify(output, rule)
                if not result.passed:
                    all_passed = False
                    errors.append(
                        {
                            "rule_id": result.rule_id,
                            "error_type": result.error_type,
                            "error_message": result.error_message,
                        }
                    )

        return {
            "verified": all_passed,
            "errors": errors if not all_passed else None,
        }

    def _build_completed_state(
        self,
        initial_state: SessionState,
        plan: ExecutionPlan,
    ) -> SessionState:
        return SessionState(
            session_id=initial_state.session_id,
            agent_id=initial_state.agent_id,
            current_plan_id=plan.plan_id,
            memory_stack=[*initial_state.memory_stack, self.context],
            execution_pointer=None,
            parent_state_id=initial_state.session_id,
            metadata={**initial_state.metadata, "completed": True},
        )

    def _build_failure_result(
        self,
        initial_state: SessionState,
        plan: ExecutionPlan,
        completed_node_ids: set[str],
        metrics: ExecutionMetrics,
        task_id: str,
        trace_id: str,
        error: Exception,
    ) -> ExecutionResult:
        failed_state = SessionState(
            session_id=initial_state.session_id,
            agent_id=initial_state.agent_id,
            current_plan_id=plan.plan_id,
            memory_stack=[*initial_state.memory_stack, self.context],
            execution_pointer=None,
            parent_state_id=initial_state.session_id,
            metadata={**initial_state.metadata, "completed": False},
        )
        return ExecutionResult(
            success=False,
            status=TaskStatus.FAILED,
            output=self.context,
            final_state=failed_state,
            metadata={
                "nodes_executed": len(completed_node_ids),
                "context": self.context,
            },
            metrics=metrics,
            task_id=task_id,
            trace_id=trace_id,
            error=str(error),
        )

    @staticmethod
    def _extract_node_error(
        ready_nodes: list[IRNode],
        results: list[dict[str, Any] | BaseException],
    ) -> Exception | None:
        for node, result in zip(ready_nodes, results, strict=False):
            if isinstance(result, BaseException):
                return RuntimeError(f"Node {node.node_id} failed: {result}")
        return None

    @staticmethod
    def _duration_ms(started_at: datetime) -> float:
        return (_now_utc() - started_at).total_seconds() * 1000
