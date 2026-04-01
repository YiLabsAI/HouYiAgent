"""SubAgentManager: spawn, join, and manage sub-agent lifecycles."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from houyi.application.runtime.context_strategy import ContextStrategy
from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.application.runtime.runner import AgentResult, AgentRunner
from houyi.domain.agent.spec import AgentSpec, SubAgentConfig

logger = logging.getLogger(__name__)


class SubAgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubAgentHandle(BaseModel):
    """Opaque handle returned by ``spawn``, used to ``join`` later."""

    handle_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    agent_id: str = ""
    role: str = ""
    task: str = ""
    status: SubAgentStatus = SubAgentStatus.PENDING
    created_at: float = Field(default_factory=time.time)

    model_config = {"arbitrary_types_allowed": True}


class SubAgentResult(BaseModel):
    """Result collected when joining a sub-agent handle."""

    agent_id: str = ""
    task: str = ""
    output: Any = None
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubAgentManager:
    """Manages sub-agent lifecycle: spawn, join, parallel, terminate.

    Sub-agents are executed via ``AgentRunner`` instances.  Each spawn
    creates an isolated runner; the manager tracks handles and collects
    results on join.
    """

    def __init__(
        self,
        *,
        llm_adapter: Any = None,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.event_emitter = event_emitter
        self._handles: dict[str, SubAgentHandle] = {}
        self._tasks: dict[str, asyncio.Task[AgentResult]] = {}
        self._runners: dict[str, AgentRunner] = {}

    async def spawn(
        self,
        spec: AgentSpec | SubAgentConfig,
        task: str,
        *,
        tools: list[Any] | None = None,
        max_turns: int = 20,
        context_strategy: ContextStrategy | None = None,
    ) -> SubAgentHandle:
        """Spawn a new sub-agent and start its tool-loop asynchronously."""
        if isinstance(spec, SubAgentConfig):
            agent_spec = AgentSpec(
                role=spec.role,
                skills=spec.skills,
                system_prompt=spec.system_prompt,
                max_turns=spec.max_turns,
            )
            max_turns = spec.max_turns
        else:
            agent_spec = spec

        runner = AgentRunner(
            agent_spec,
            llm_adapter=self.llm_adapter,
            tools=tools or [],
            context_strategy=context_strategy,
            max_turns=max_turns,
            event_emitter=self.event_emitter,
        )

        handle = SubAgentHandle(
            agent_id=runner.agent_id,
            role=agent_spec.role,
            task=task,
            status=SubAgentStatus.RUNNING,
        )

        async_task = asyncio.create_task(runner.run(task))
        self._handles[handle.handle_id] = handle
        self._tasks[handle.handle_id] = async_task
        self._runners[handle.handle_id] = runner

        if self.event_emitter:
            await self.event_emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.SUB_AGENT_SPAWNED,
                    agent_id=runner.agent_id,
                    agent_name=agent_spec.role,
                    data={"task": task},
                )
            )

        return handle

    async def spawn_parallel(
        self,
        agents: list[tuple[AgentSpec | SubAgentConfig, str]],
        *,
        max_concurrent: int = 5,
        tools: list[Any] | None = None,
    ) -> list[SubAgentHandle]:
        """Spawn multiple sub-agents concurrently, respecting *max_concurrent*."""
        semaphore = asyncio.Semaphore(max_concurrent)
        handles: list[SubAgentHandle] = []

        async def _spawn_one(spec: AgentSpec | SubAgentConfig, task: str) -> SubAgentHandle:
            async with semaphore:
                return await self.spawn(spec, task, tools=tools)

        coros = [_spawn_one(s, t) for s, t in agents]
        handles = await asyncio.gather(*coros)
        return list(handles)

    async def join(
        self,
        handle: SubAgentHandle,
        *,
        timeout: float | None = None,
    ) -> SubAgentResult:
        """Wait for a sub-agent to finish and return its result."""
        async_task = self._tasks.get(handle.handle_id)
        if async_task is None:
            return SubAgentResult(
                agent_id=handle.agent_id,
                task=handle.task,
                success=False,
                error="Handle not found",
            )

        try:
            if timeout is not None:
                result = await asyncio.wait_for(asyncio.shield(async_task), timeout=timeout)
            else:
                result = await async_task
        except TimeoutError:
            handle.status = SubAgentStatus.FAILED
            return SubAgentResult(
                agent_id=handle.agent_id,
                task=handle.task,
                success=False,
                error="Timeout",
            )
        except Exception as exc:
            handle.status = SubAgentStatus.FAILED
            return SubAgentResult(
                agent_id=handle.agent_id,
                task=handle.task,
                success=False,
                error=str(exc),
            )

        handle.status = SubAgentStatus.COMPLETED if result.success else SubAgentStatus.FAILED

        if self.event_emitter:
            await self.event_emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.SUB_AGENT_COMPLETED,
                    agent_id=handle.agent_id,
                    agent_name=handle.role,
                    data={"success": result.success, "output": str(result.output)[:200]},
                )
            )

        return SubAgentResult(
            agent_id=handle.agent_id,
            task=handle.task,
            output=result.output,
            success=result.success,
            error=result.error,
            duration_ms=result.duration_ms,
        )

    async def join_all(
        self,
        handles: list[SubAgentHandle],
        *,
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> list[SubAgentResult]:
        """Join all handles concurrently."""
        coros = [self.join(h, timeout=timeout) for h in handles]
        raw = await asyncio.gather(*coros, return_exceptions=return_exceptions)
        results: list[SubAgentResult] = []
        for r in raw:
            if isinstance(r, BaseException):
                results.append(SubAgentResult(success=False, error=str(r)))
            else:
                results.append(r)
        return results

    async def terminate(self, handle: SubAgentHandle) -> None:
        """Cancel a running sub-agent."""
        task = self._tasks.get(handle.handle_id)
        if task and not task.done():
            task.cancel()
            handle.status = SubAgentStatus.CANCELLED
