"""Execution engine for console server."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from houyi.adapters.llm import LLMAdapterFactory
from houyi.application.tool_calling.retry_policy import (
    RetryPolicy,
    calculate_retry_delay,
    get_num_retries_from_policy,
)
from houyi.application.workflow.config_service import ConfigService
from houyi.application.workflow.execution_backend_resolver import ExecutionBackendResolver
from houyi.application.workflow.execution_backends import ExecutionBackend
from houyi.application.workflow.execution_order_service import ExecutionOrderService
from houyi.infrastructure.observability import Span, SpanType, TraceContext
from houyi.interface.protocol.ir import (
    CheckpointTrigger,
    ExecutionIR,
    ExecutionStatus,
    NodeExecutionIR,
    NodeStatus,
    PlanIR,
)

from ..gateway.event_bus import EventBus
from ..gateway.events import RetryFailedEvent, RetryStatusEvent, RetrySuccessEvent, SpanUpdateEvent
from ..rag import get_knowledge_service
from ..tooling.coordinator import ToolCallCoordinator
from .agent_comm_service import AgentCommService
from .checkpoint_service import CheckpointService
from .context import ExecutionContext
from .context_factory import ExecutionContextFactory
from .context_service import ContextService
from .lifecycle_hooks import LifecycleHook
from .lifecycle_service import ExecutionLifecycleService
from .llm_execution_flow import LLMExecutionFlow
from .llm_gateway_service import LLMGatewayService
from .mcp_gateway import MCPGateway
from .memory_service import MemoryService
from .node_execution_flow import NodeExecutionFlow
from .node_execution_service import NodeExecutionService
from .node_executor_factory import NodeExecutorFactory
from .observation_service import ObservationService
from .plan_execution_loop import PlanExecutionLoop
from .plan_service import PlanService
from .stores import CheckpointStore, ExecutionStore, PlanStore
from .workflow_service import WorkflowService

if TYPE_CHECKING:
    from ..gateway.websocket import ConnectionManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Engine for executing plans and managing execution state."""

    def __init__(self, connection_manager: ConnectionManager) -> None:
        """Initialize execution engine.

        Args:
            connection_manager: WebSocket connection manager for sending events
        """
        self.connection_manager = connection_manager
        self._execution_tasks: dict[str, asyncio.Task] = {}
        self.execution_store = ExecutionStore()
        self.checkpoint_store = CheckpointStore()
        self.plan_store = PlanStore(plans_dir=Path("data/plans"))
        # Backward-compatible aliases
        self.executions = self.execution_store.executions
        self.checkpoints = self.checkpoint_store.checkpoints
        self.plans = self.plan_store.plans
        self.session_plans = self.plan_store.session_plans
        self.llm_call_logs: dict[str, list] = {}  # Track LLM calls per execution for replay
        self.tool_call_coordinator = ToolCallCoordinator()
        self.config_service = ConfigService()
        self.event_bus = EventBus()
        self.observation_service = ObservationService(
            connection_manager=self.connection_manager,
            event_bus=self.event_bus,
        )
        self.memory_service = MemoryService()
        self.rag_service = get_knowledge_service()
        self.context_service = ContextService(
            memory_service=self.memory_service,
            rag_service=self.rag_service,
        )
        self.mcp_gateway = MCPGateway()
        self.agent_comm_service = AgentCommService()
        self.execution_context_factory = ExecutionContextFactory(
            context_service=self.context_service,
            memory_service=self.memory_service,
            rag_service=self.rag_service,
            mcp_gateway=self.mcp_gateway,
            agent_comm_service=self.agent_comm_service,
            observation_service=self.observation_service,
            tool_cache=self.tool_call_coordinator.tool_call_cache,
        )
        self.execution_backend: ExecutionBackend = ExecutionBackendResolver().resolve()
        self.lifecycle_hooks: list[LifecycleHook] = []
        self.plan_service = PlanService(self.plan_store)
        self.execution_lifecycle_service = ExecutionLifecycleService(
            execution_store=self.execution_store,
            checkpoint_store=self.checkpoint_store,
            plan_store=self.plan_store,
            observation_service=self.observation_service,
            execution_backend=self.execution_backend,
            execution_tasks=self._execution_tasks,
            llm_call_logs=self.llm_call_logs,
            normalize_run_settings=self._normalize_run_settings,
            execute_plan=self._execute_plan,
            restore_checkpoint=self._restore_checkpoint,
        )
        self.execution_order_service = ExecutionOrderService()
        self.llm_execution_flow = LLMExecutionFlow(
            observation_service=self.observation_service,
            tool_call_service=None,
            adapter_factory=lambda: LLMAdapterFactory.create(),
        )
        self.llm_execution_flow.llm_call_logs = self.llm_call_logs
        self.llm_gateway_service = LLMGatewayService(self.llm_execution_flow)
        self.node_executor_registry = NodeExecutorFactory(
            config_service=self.config_service,
            execute_llm_real=self._execute_llm_real,
            execute_llm_mock=self._execute_llm_mock,
        ).build_registry()
        self.tool_call_service = self.tool_call_coordinator.build_service(
            connection_manager=self.connection_manager,
            record_llm_call=self.llm_gateway_service.record_llm_call,
        )
        self.llm_execution_flow.set_tool_call_service(self.tool_call_service)
        self.node_execution_flow = NodeExecutionFlow(
            node_executor_registry=self.node_executor_registry,
            observation_service=self.observation_service,
            notify_lifecycle=self._notify_lifecycle,
            context_factory=self._build_execution_context,
        )
        self.node_execution_service = NodeExecutionService(self.node_execution_flow)
        self.plan_execution_loop = PlanExecutionLoop(
            plan_getter=self._get_plan_for_session,
            get_execution_order=self._get_execution_order,
            node_executor=self._execute_node,
            checkpoint_callback=self._create_checkpoint,
            observation_service=self.observation_service,
            notify_lifecycle=self._notify_lifecycle,
            context_factory=self._build_execution_context,
        )
        self.checkpoint_service = CheckpointService(
            checkpoint_store=self.checkpoint_store,
            execution_store=self.execution_store,
            plan_store=self.plan_store,
            observation_service=self.observation_service,
            execution_tasks=self._execution_tasks,
            llm_call_logs=self.llm_call_logs,
        )

        # Plan persistence directory (session-based, temporary)
        self.plan_store.plans_dir.mkdir(parents=True, exist_ok=True)

        # Workflow persistence directory (user-saved, permanent)
        workflows_dir_env = os.getenv("HOUYI_WORKFLOWS_DIR")
        self.workflows_dir = (
            Path(workflows_dir_env) if workflows_dir_env else Path("data/workflows")
        )
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_service = WorkflowService(self.workflows_dir)

        logger.info("Plan storage directory: %s", self.plan_store.plans_dir.absolute())
        logger.info("Workflow storage directory: %s", self.workflows_dir.absolute())

    def register_lifecycle_hook(self, hook: LifecycleHook) -> None:
        self.lifecycle_hooks.append(hook)

    async def _notify_lifecycle(self, method_name: str, *args: Any) -> None:
        for hook in self.lifecycle_hooks:
            handler = getattr(hook, method_name, None)
            if handler is None:
                continue
            try:
                await handler(*args)
            except Exception:
                logger.exception("Lifecycle hook failed: %s", method_name)

    def _normalize_run_settings(self, run_settings: dict[str, Any] | None) -> dict[str, Any]:
        return self.config_service.normalize_run_settings(run_settings)

    def _build_execution_context(
        self, session_id: str, execution: ExecutionIR, plan: PlanIR
    ) -> ExecutionContext:
        """Build a shared execution context with all required services."""
        return self.execution_context_factory.build(session_id, execution, plan)

    def _get_plan_for_session(self, session_id: str, fallback_plan: PlanIR) -> PlanIR:
        """Return the latest plan for a session, falling back to the provided plan."""
        return self.plan_service.get_plan_for_session(session_id, fallback_plan)

    async def start_execution(
        self,
        session_id: str,
        plan: PlanIR,
        run_settings: dict[str, Any] | None = None,
    ) -> ExecutionIR:
        """Start executing a plan."""
        return await self.execution_lifecycle_service.start_execution(
            session_id=session_id,
            plan=plan,
            run_settings=run_settings,
        )

    async def pause_execution(self, execution_id: str) -> None:
        """Pause an execution."""
        await self.execution_lifecycle_service.pause_execution(execution_id)

    async def resume_execution(self, execution_id: str) -> None:
        """Resume a paused execution with potentially modified plan."""
        await self.execution_lifecycle_service.resume_execution(execution_id)

    async def abort_execution(self, execution_id: str) -> None:
        """Abort an execution."""
        await self.execution_lifecycle_service.abort_execution(execution_id)

    async def abort_session_executions(self, session_id: str) -> None:
        """Abort any running executions for a session."""
        for execution in list(self.execution_store.executions.values()):
            if execution.metadata.get("session_id") != session_id:
                continue
            if execution.status != ExecutionStatus.RUNNING:
                continue
            await self.abort_execution(execution.execution_id)

    async def retry_node(
        self,
        *,
        session_id: str,
        execution_id: str,
        node_id: str,
        new_inputs: dict[str, Any],
    ) -> None:
        """Retry a failed node using configurable retry policy."""
        execution = self.execution_store.get(execution_id)
        if not execution:
            logger.warning("Execution not found: %s", execution_id)
            return

        if execution.status in {ExecutionStatus.ABORTED, ExecutionStatus.COMPLETED}:
            logger.warning("Cannot retry node for execution in state: %s", execution.status)
            return

        plan = self.plan_store.get_cached(session_id)
        if not plan:
            logger.error("No plan found for session: %s", session_id)
            return

        node_exec = execution.node_executions.get(node_id)
        if not node_exec or node_exec.status != NodeStatus.FAILED:
            logger.warning("Node not in failed state: %s", node_id)
            return

        node_exec.inputs = new_inputs
        policy = self._resolve_retry_policy(execution)
        max_retries = max(0, policy.default_retries)

        try:
            await self._retry_node_with_policy(
                session_id=session_id,
                execution=execution,
                plan=plan,
                node_id=node_id,
                max_retries=max_retries,
                policy=policy,
            )
        except Exception as exc:
            node_exec.error = str(exc)
            node_exec.status = NodeStatus.FAILED
            node_exec.completed_at = datetime.now()
            logger.exception("Retry node failed: %s", node_id)

    def _resolve_retry_policy(self, execution: ExecutionIR) -> RetryPolicy:
        run_settings = execution.metadata.get("run_settings") or {}
        settings = run_settings.get("retry_policy") if isinstance(run_settings, dict) else None
        policy = RetryPolicy.from_settings(settings if isinstance(settings, dict) else None)
        policy.default_retries = max(0, policy.default_retries)
        return policy

    async def _retry_node_with_policy(
        self,
        *,
        session_id: str,
        execution: ExecutionIR,
        plan: PlanIR,
        node_id: str,
        max_retries: int,
        policy: RetryPolicy,
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                await self.node_execution_service.execute_node(
                    session_id=session_id,
                    execution=execution,
                    plan=plan,
                    node_id=node_id,
                    sleep_func=asyncio.sleep,
                )
                if attempt > 0:
                    await self.observation_service.emit(
                        RetrySuccessEvent(
                            event_id=f"evt_{uuid4().hex[:8]}",
                            session_id=session_id,
                            execution_id=execution.execution_id,
                            node_id=node_id,
                            attempt=attempt + 1,
                        )
                    )
                return
            except Exception as exc:
                last_error = exc
                policy_retries = get_num_retries_from_policy(exc, policy)
                max_retries = max(max_retries, policy_retries)
                if attempt >= max_retries:
                    break

                # Emit retry span so the attempt is visible in the waterfall
                retry_span = Span(
                    name=f"retry.{node_id}",
                    span_type=SpanType.RETRY,
                    trace_id=execution.execution_id,
                    node_id=node_id,
                    attributes={
                        "retry.attempt": attempt + 1,
                        "retry.max_retries": max_retries,
                        "retry.error": str(exc),
                    },
                )
                # Attach to current trace context if available
                parent = TraceContext.current()
                if parent:
                    retry_span.parent_id = parent.span_id
                    retry_span.trace_id = parent.trace_id
                retry_span.set_status("error", str(exc))
                retry_span.end()
                await self.observation_service.emit(
                    SpanUpdateEvent.from_span(
                        retry_span,
                        session_id=session_id,
                        execution_id=execution.execution_id,
                    )
                )

                await self.observation_service.emit(
                    RetryStatusEvent(
                        event_id=f"evt_{uuid4().hex[:8]}",
                        session_id=session_id,
                        execution_id=execution.execution_id,
                        node_id=node_id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(exc),
                    )
                )
                delay = self._retry_delay(attempt)
                await asyncio.sleep(delay)

        if last_error:
            await self.observation_service.emit(
                RetryFailedEvent(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    execution_id=execution.execution_id,
                    node_id=node_id,
                    max_retries=max_retries,
                    error=str(last_error),
                )
            )
            raise last_error

    def _retry_delay(self, attempt: int) -> float:
        min_delay = float(os.getenv("HOUYI_RETRY_MIN_DELAY", 0.5))
        max_delay = float(os.getenv("HOUYI_RETRY_MAX_DELAY", 8.0))
        jitter = float(os.getenv("HOUYI_RETRY_JITTER", 0.75))
        return calculate_retry_delay(
            attempt=attempt,
            min_delay=min_delay,
            max_delay=max_delay,
            jitter=jitter,
        )

    async def _execute_plan(
        self,
        session_id: str,
        execution: ExecutionIR,
        plan: PlanIR,
    ) -> None:
        """Execute a plan (internal)."""
        self.plan_execution_loop.set_sleep_func(asyncio.sleep)
        await self.plan_execution_loop.execute(session_id, execution, plan)

    async def _execute_node(
        self,
        context: ExecutionContext | None = None,
        node_id: str | None = None,
        *,
        session_id: str | None = None,
        execution: ExecutionIR | None = None,
        plan: PlanIR | None = None,
    ) -> None:
        """Execute a single node."""
        await self.node_execution_service.execute_node(
            context=context,
            node_id=node_id,
            session_id=session_id,
            execution=execution,
            plan=plan,
            sleep_func=asyncio.sleep,
        )

    async def _execute_llm_real(
        self,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        enable_tool_calls: bool = False,
        tool_names: list[str] | None = None,
        tool_choice: Any | None = None,
        max_tool_calls: int = 6,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        prompt_cache_key: str | None = None,
    ) -> None:
        """Execute LLM node with real API call or deterministic replay."""
        self.llm_execution_flow.set_sleep_func(asyncio.sleep)
        await self.llm_gateway_service.execute_llm_real(
            session_id=session_id,
            execution=execution,
            node_id=node_id,
            node_exec=node_exec,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
            thinking_budget=thinking_budget,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_tool_calls=enable_tool_calls,
            tool_names=tool_names,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_key=prompt_cache_key,
        )

    async def _execute_llm_tool_calls(
        self,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
        prompt: str,
        system_prompt: str | None,
        user_prompt: str | None,
        model: str | None,
        tool_names: list[str],
        tool_choice: Any | None,
        max_tool_calls: int,
        max_tokens: int | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        prompt_cache_key: str | None = None,
    ) -> bool:
        return await self.llm_gateway_service.execute_llm_tool_calls(
            session_id=session_id,
            execution=execution,
            node_id=node_id,
            node_exec=node_exec,
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            tool_names=tool_names,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_key=prompt_cache_key,
        )

    async def _execute_llm_mock(
        self,
        session_id: str,
        execution: ExecutionIR,
        node_id: str,
        node_exec: NodeExecutionIR,
    ) -> None:
        """Execute LLM node with mock streaming (for testing UI)."""
        self.llm_execution_flow.set_sleep_func(asyncio.sleep)
        await self.llm_gateway_service.execute_llm_mock(
            session_id=session_id,
            execution=execution,
            node_id=node_id,
            node_exec=node_exec,
        )

    async def _create_checkpoint(
        self,
        session_id: str,
        execution: ExecutionIR,
        trigger: CheckpointTrigger,
        node_id: str | None = None,
    ) -> None:
        """Create a checkpoint."""
        await self.checkpoint_service.create_checkpoint(
            session_id=session_id,
            execution=execution,
            trigger=trigger,
            node_id=node_id,
        )

    async def restore_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        replay_mode: str = "deterministic",
        execution_id: str | None = None,
    ) -> None:
        """Restore execution from a checkpoint."""
        await self.execution_lifecycle_service.restore_checkpoint(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            replay_mode=replay_mode,
            execution_id=execution_id,
        )

    async def _restore_checkpoint(
        self,
        *,
        session_id: str,
        checkpoint_id: str,
        replay_mode: str = "deterministic",
        execution_id: str | None = None,
    ) -> None:
        """Restore execution from a checkpoint (internal)."""
        await self.checkpoint_service.restore_checkpoint(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            replay_mode=replay_mode,
            execution_id=execution_id,
        )

    def record_llm_call(
        self,
        execution_id: str,
        node_id: str,
        model: str,
        prompt: str | list[dict[str, Any]],
        response: str,
        metadata: dict | None = None,
    ) -> None:
        """Record an LLM call for deterministic replay."""
        self.llm_gateway_service.record_llm_call(
            execution_id=execution_id,
            node_id=node_id,
            model=model,
            prompt=prompt,
            response=response,
            metadata=metadata,
        )

    def get_recorded_llm_response(
        self,
        execution_id: str,
        node_id: str,
    ) -> str | None:
        """Get recorded LLM response for deterministic replay."""
        return self.llm_gateway_service.get_recorded_llm_response(execution_id, node_id)

    def _get_execution_order(self, plan: PlanIR) -> list[str]:
        """Get execution order using topological sort.

        Args:
            plan: Plan to sort

        Returns:
            List of node IDs in execution order
        """
        return self.execution_order_service.get_execution_order(plan)
