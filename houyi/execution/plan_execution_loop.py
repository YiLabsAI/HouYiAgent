"""Plan execution loop.

This module contains the core pause/resume execution loop logic.
It must remain independent of any server transport/event emission.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from houyi.protocol.ir import CheckpointTrigger, ExecutionIR, ExecutionStatus, NodeStatus, PlanIR

logger = logging.getLogger(__name__)

PlanGetter = Callable[[str, PlanIR], PlanIR]
ExecutionOrderResolver = Callable[[PlanIR], list[str]]
NodeExecutor = Callable[[Any, str], Awaitable[None]]
CheckpointCallback = Callable[..., Awaitable[None]]
LifecycleNotifier = Callable[..., Awaitable[None]]
SleepFunc = Callable[[float], Awaitable[None]]
ContextFactory = Callable[[str, ExecutionIR, PlanIR], Any]


class PlanExecutionLoop:
    """Execute plan nodes in order with pause/resume handling."""

    def __init__(
        self,
        *,
        plan_getter: PlanGetter,
        get_execution_order: ExecutionOrderResolver,
        node_executor: NodeExecutor,
        checkpoint_callback: CheckpointCallback,
        notify_lifecycle: LifecycleNotifier,
        context_factory: ContextFactory,
        sleep_func: SleepFunc | None = None,
    ) -> None:
        self._plan_getter = plan_getter
        self._get_execution_order = get_execution_order
        self._node_executor = node_executor
        self._checkpoint_callback = checkpoint_callback
        self._notify_lifecycle = notify_lifecycle
        self._context_factory = context_factory
        self._sleep_func = sleep_func or asyncio.sleep

    def set_sleep_func(self, sleep_func: SleepFunc) -> None:
        self._sleep_func = sleep_func

    async def execute(self, session_id: str, execution: ExecutionIR, plan: PlanIR) -> None:
        context: Any | None = None
        try:
            executed_nodes = {
                node_id
                for node_id, node_exec in (execution.node_executions or {}).items()
                if getattr(node_exec, "status", None) == NodeStatus.COMPLETED
            }
            context = self._context_factory(session_id, execution, plan)
            await self._notify_lifecycle("on_execution_start", context)

            while True:
                current_plan = self._plan_getter(session_id, plan)
                if hasattr(context, "update_plan"):
                    context.update_plan(current_plan)

                node_order = self._get_execution_order(current_plan)

                next_node = None
                for node_id in node_order:
                    if node_id not in executed_nodes:
                        next_node = node_id
                        break

                if next_node is None:
                    break

                if execution.status == ExecutionStatus.ABORTED:
                    logger.info("Execution aborted: %s", execution.execution_id)
                    return

                while execution.status == ExecutionStatus.PAUSED:
                    await self._sleep_func(0.1)
                    if execution.status == ExecutionStatus.ABORTED:
                        logger.info("Execution aborted while paused: %s", execution.execution_id)
                        return

                await self._node_executor(context, next_node)
                executed_nodes.add(next_node)

                await self._invoke_checkpoint_callback(
                    session_id=session_id,
                    execution=execution,
                    trigger=CheckpointTrigger.NODE_COMPLETED,
                    node_id=next_node,
                )

            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = datetime.now()
            logger.info("Execution completed: %s", execution.execution_id)

        except asyncio.CancelledError:
            logger.info("Execution task cancelled: %s", execution.execution_id)
            return

        except Exception as exc:
            logger.error(
                "Execution failed: %s - %s",
                execution.execution_id,
                exc,
                exc_info=True,
            )
            execution.status = ExecutionStatus.FAILED
            execution.error = str(exc)
            execution.completed_at = datetime.now()
            raise

        finally:
            if context is not None:
                await self._notify_lifecycle("on_execution_end", context)

    async def _invoke_checkpoint_callback(
        self,
        *,
        session_id: str,
        execution: ExecutionIR,
        trigger: CheckpointTrigger,
        node_id: str | None,
    ) -> None:
        callback = self._checkpoint_callback
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            await callback(session_id, execution, trigger)
            return

        params = list(signature.parameters.values())
        supports_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
        supports_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)

        if (
            supports_kwargs
            or supports_varargs
            or "node_id" in signature.parameters
            or len(params) >= 4
        ):
            await callback(session_id, execution, trigger, node_id=node_id)
        else:
            await callback(session_id, execution, trigger)
