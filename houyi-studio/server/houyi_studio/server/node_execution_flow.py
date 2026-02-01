"""Node execution flow for the console execution engine."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import datetime
from typing import Any
from uuid import uuid4

from houyi.execution.node_execution_utils import (
    extract_output_payload,
    resolve_inputs,
    resolve_value,
)
from houyi.protocol.ir import ExecutionIR, NodeExecutionIR, NodeStatus, PlanIR
from houyi.protocol.ir.plan_ir import NodeType

from .events import NodeStatusEvent
from .execution_context import ExecutionContext
from .node_executor_registry import NodeExecutorRegistry
from .observation_service import ObservationService

logger = logging.getLogger(__name__)

ContextFactory = Callable[[str, ExecutionIR, PlanIR], ExecutionContext]
LifecycleNotifier = Callable[..., Awaitable[None]]
SleepFunc = Callable[[float], Awaitable[None]]


class NodeExecutionFlow:
    """Handle node lifecycle execution and event emission."""

    def __init__(
        self,
        *,
        node_executor_registry: NodeExecutorRegistry,
        observation_service: ObservationService,
        notify_lifecycle: LifecycleNotifier,
        context_factory: ContextFactory,
        sleep_func: SleepFunc | None = None,
    ) -> None:
        self._node_executor_registry = node_executor_registry
        self._observation_service = observation_service
        self._notify_lifecycle = notify_lifecycle
        self._context_factory = context_factory
        self._sleep_func = sleep_func or asyncio.sleep

    def set_sleep_func(self, sleep_func: SleepFunc) -> None:
        self._sleep_func = sleep_func

    def _resolve_value(self, value: Any, context_values: dict[str, Any]) -> Any:
        return resolve_value(value, context_values)

    def _resolve_inputs(
        self, inputs: dict[str, Any], context_values: dict[str, Any]
    ) -> dict[str, Any]:
        return resolve_inputs(inputs, context_values)

    @staticmethod
    def _extract_output_payload(outputs: dict[str, Any]) -> dict[str, Any]:
        return extract_output_payload(outputs)

    def _apply_outputs_to_context(
        self, node: Any, node_exec: NodeExecutionIR, context: ExecutionContext
    ) -> None:
        if not node:
            return
        if node.node_type == NodeType.LLM:
            tool_calls = (node_exec.outputs or {}).get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    tool_name = call.get("tool_name") if isinstance(call, dict) else None
                    args = call.get("args") if isinstance(call, dict) else None
                    if isinstance(args, dict):
                        for key, value in args.items():
                            if key not in context.execution.context:
                                context.execution.context[key] = value
                    result = call.get("result") if isinstance(call, dict) else None
                    raw = result.get("raw") if isinstance(result, dict) else None
                    if isinstance(raw, dict):
                        payload = raw.get("result", raw)
                        if isinstance(payload, dict):
                            for key, value in payload.items():
                                if key not in context.execution.context:
                                    context.execution.context[key] = value
                        else:
                            if tool_name and tool_name not in context.execution.context:
                                context.execution.context[tool_name] = payload
                            if tool_name == "get_date":
                                context.execution.context["date"] = payload
                    elif (
                        raw is not None and tool_name and tool_name not in context.execution.context
                    ):
                        context.execution.context[tool_name] = raw
                        if tool_name == "get_date":
                            context.execution.context["date"] = raw
        output_payload = self._extract_output_payload(node_exec.outputs or {})
        if not node.outputs:
            if isinstance(output_payload, dict):
                for key, value in output_payload.items():
                    if key not in context.execution.context:
                        context.execution.context[key] = value
            return
        for output_key, var_name in node.outputs.items():
            if not var_name:
                continue
            value = output_payload.get(output_key)
            if value is None and output_key == "result":
                value = output_payload
            if value is not None:
                context.execution.context[var_name] = value

    async def execute(
        self,
        context: ExecutionContext | None = None,
        node_id: str | None = None,
        *,
        session_id: str | None = None,
        execution: ExecutionIR | None = None,
        plan: PlanIR | None = None,
    ) -> None:
        """Execute a single node and emit lifecycle events.

        Args:
            context: Execution context
            node_id: Node identifier
        """
        if context is None:
            if not (session_id and execution and plan and node_id):
                raise ValueError("execute requires context or session_id/execution/plan/node_id")
            context = self._context_factory(session_id, execution, plan)
        elif node_id is None:
            raise ValueError("execute requires node_id when context is provided")

        session_id = context.session_id
        execution = context.execution
        plan = context.plan

        existing_exec = execution.node_executions.get(node_id)
        if existing_exec is None:
            node_exec = NodeExecutionIR(
                node_id=node_id,
                status=NodeStatus.RUNNING,
                started_at=datetime.now(),
                completed_at=None,
                inputs={},
                outputs={},
                error=None,
                streaming_output="",
                metadata={},
            )
        else:
            node_exec = existing_exec
            # Reset execution state but preserve caller-provided inputs for retries.
            node_exec.status = NodeStatus.RUNNING
            node_exec.started_at = datetime.now()
            node_exec.completed_at = None
            node_exec.outputs = {}
            node_exec.error = None
            node_exec.streaming_output = ""
            node_exec.metadata = node_exec.metadata or {}
        execution.node_executions[node_id] = node_exec

        logger.info("[%s] Node started at %s", node_id, node_exec.started_at.strftime("%H:%M:%S"))

        node = next((n for n in plan.nodes if n.node_id == node_id), None)
        executor = self._node_executor_registry.resolve(node.node_type) if node else None
        if executor and not node_exec.inputs:
            build_inputs = getattr(executor, "build_inputs", None)
            if callable(build_inputs):
                try:
                    node_exec.inputs = build_inputs(context, node)
                except Exception:
                    logger.exception("Failed to build inputs for node %s", node_id)
                    node_exec.inputs = node_exec.inputs or {}
        if node and node.inputs:
            resolved_inputs = self._resolve_inputs(node.inputs, context.execution.context)
            if resolved_inputs:
                node_exec.inputs = {**node_exec.inputs, **resolved_inputs}

        logger.debug(
            "[%s] Node inputs prepared: node_type=%s node_inputs=%s resolved_inputs=%s exec_inputs=%s context_keys=%s",
            node_id,
            getattr(node, "node_type", None),
            node.inputs if node else None,
            resolved_inputs if node and node.inputs else {},
            node_exec.inputs or {},
            sorted(context.execution.context.keys()),
        )

        # NOTE(core): Minimal observability envelope (OTel-aligned placeholder).
        # We use execution_id as a stable trace_id surrogate for now.
        # Later phases will generate true trace/span IDs and attach llm/tool spans with cost/cache fields.
        trace_id = execution.execution_id
        node_span_id = f"span_{uuid4().hex[:16]}"
        execution_span_id = f"span_{uuid4().hex[:16]}"
        start_time_iso = (
            node_exec.started_at.isoformat() if node_exec.started_at else datetime.now().isoformat()
        )

        start_event = NodeStatusEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            execution_id=execution.execution_id,
            node_id=node_id,
            status=NodeStatus.RUNNING,
            inputs=node_exec.inputs or {},
            execution_metadata=execution.metadata or {},
            observation={
                "trace_id": trace_id,
                "span_id": node_span_id,
                "parent_span_id": execution_span_id,
                "span_type": "node",
                "start_time": start_time_iso,
                "end_time": None,
                "status": "running",
            },
        )
        await self._observation_service.emit(start_event)
        await self._notify_lifecycle("before_node", context, node_id, node_exec)

        if context.context_service:
            try:
                # Resolve external context before node execution.
                context.context_bundle = await context.context_service.resolve(context)
                if context.context_bundle is not None:
                    # Persist context bundle for UI visibility and checkpoint snapshots.
                    context.execution.metadata["context_bundle"] = asdict(context.context_bundle)
            except Exception:
                logger.exception("Context resolution failed for node %s", node_id)

        try:
            if executor is not None:
                await executor.execute(context, node, node_exec)
            elif node and node.node_type in {NodeType.LLM, NodeType.TOOL}:
                raise RuntimeError(f"No executor registered for node type {node.node_type}")
            else:
                # Non-LLM nodes just wait briefly.
                await self._sleep_func(2)

            if node_exec.error:
                node_exec.status = NodeStatus.FAILED
                node_exec.completed_at = datetime.now()

                end_time_iso = (
                    node_exec.completed_at.isoformat()
                    if node_exec.completed_at
                    else datetime.now().isoformat()
                )

                fail_event = NodeStatusEvent(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    execution_id=execution.execution_id,
                    node_id=node_id,
                    status=NodeStatus.FAILED,
                    inputs=node_exec.inputs or {},
                    outputs=node_exec.outputs,
                    error=node_exec.error,
                    execution_metadata=execution.metadata or {},
                    observation={
                        "trace_id": trace_id,
                        "span_id": node_span_id,
                        "parent_span_id": execution_span_id,
                        "span_type": "node",
                        "start_time": start_time_iso,
                        "end_time": end_time_iso,
                        "status": "failed",
                    },
                )
                await self._observation_service.emit(fail_event)
                raise RuntimeError(f"Node {node_id} failed: {node_exec.error}")

            node_exec.status = NodeStatus.COMPLETED
            node_exec.completed_at = datetime.now()
            if not node_exec.outputs:
                node_exec.outputs = {
                    "result": node_exec.streaming_output or f"Output from {node_id}"
                }
            self._apply_outputs_to_context(node, node_exec, context)

            end_time_iso = (
                node_exec.completed_at.isoformat()
                if node_exec.completed_at
                else datetime.now().isoformat()
            )

            complete_event = NodeStatusEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                execution_id=execution.execution_id,
                node_id=node_id,
                status=NodeStatus.COMPLETED,
                inputs=node_exec.inputs or {},
                outputs=node_exec.outputs,
                execution_metadata=execution.metadata or {},
                observation={
                    "trace_id": trace_id,
                    "span_id": node_span_id,
                    "parent_span_id": execution_span_id,
                    "span_type": "node",
                    "start_time": start_time_iso,
                    "end_time": end_time_iso,
                    "status": "completed",
                },
            )
            await self._observation_service.emit(complete_event)
        finally:
            await self._notify_lifecycle("after_node", context, node_id, node_exec)
