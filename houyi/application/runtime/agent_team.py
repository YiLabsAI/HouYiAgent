"""AgentTeamManager: spawn, join, and manage team agent lifecycles."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from houyi.application.context.context_strategy import ContextStrategy
from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.application.runtime.runner import AgentResult, AgentRunner
from houyi.domain.agent.spec import AgentSpec, AgentTeamConfig

logger = logging.getLogger(__name__)


class TeamAgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TeamAgentHandle(BaseModel):
    """Opaque handle returned by spawn, used to join later."""

    handle_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    agent_id: str = ""
    role: str = ""
    task: str = ""
    status: TeamAgentStatus = TeamAgentStatus.PENDING
    created_at: float = Field(default_factory=time.time)

    model_config = {"arbitrary_types_allowed": True}


class TeamAgentResult(BaseModel):
    """Result collected when joining a team agent handle."""

    agent_id: str = ""
    task: str = ""
    output: Any = None
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTeamManager:
    """Manages team agent lifecycle: spawn, join, parallel, terminate.

    Each spawn creates an isolated AgentRunner; the manager tracks
    handles and collects results on join.  Works for both DELEGATE
    (supervisor-worker) and AUTONOMOUS (peer) topologies.
    """

    def __init__(
        self,
        *,
        llm_adapter: Any = None,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.event_emitter = event_emitter
        self._handles: dict[str, TeamAgentHandle] = {}
        self._tasks: dict[str, asyncio.Task[AgentResult]] = {}
        self._runners: dict[str, AgentRunner] = {}

    def _resolve_llm(self, model_override: str | None) -> Any:
        """Return an LLM adapter, optionally overriding the model.

        When *model_override* is set (from AgentTeamConfig.model), the
        parent adapter is cloned with the new model name — avoiding the cost
        of constructing a whole new adapter from scratch.
        """
        if not model_override or self.llm_adapter is None:
            return self.llm_adapter
        adapter = self.llm_adapter
        if hasattr(adapter, "model"):
            try:
                import copy

                cloned = copy.copy(adapter)
                cloned.model = model_override
                cloned.default_model = model_override
                return cloned
            except Exception:
                logger.warning("Cannot clone adapter for model %s, using parent", model_override)
        return adapter

    async def spawn(
        self,
        spec: AgentSpec | AgentTeamConfig | Any,
        task: str,
        *,
        tools: list[Any] | None = None,
        max_turns: int = 20,
        context_strategy: ContextStrategy | None = None,
    ) -> TeamAgentHandle:
        """Spawn a new team agent and start its tool-loop asynchronously.

        *spec* may be an AgentSpec, AgentTeamConfig, or an Agent
        instance.  When an Agent is passed, its internal spec, tools, and
        LLM adapter are used directly.
        """
        from houyi.application.runtime.agent import Agent

        if isinstance(spec, Agent):
            agent_spec = spec.spec
            tools = tools or spec._tools
            max_turns = agent_spec.max_turns
            llm = spec._llm_adapter or self.llm_adapter
        elif isinstance(spec, AgentTeamConfig):
            agent_spec = AgentSpec(
                role=spec.role,
                skills=spec.skills,
                system_prompt=spec.system_prompt,
                max_turns=spec.max_turns,
            )
            max_turns = spec.max_turns
            llm = self._resolve_llm(spec.model)
        else:
            agent_spec = spec
            llm = self.llm_adapter

        runner = AgentRunner(
            agent_spec,
            llm_adapter=llm,
            tools=tools or [],
            context_strategy=context_strategy,
            max_turns=max_turns,
            event_emitter=self.event_emitter,
        )

        handle = TeamAgentHandle(
            agent_id=runner.agent_id,
            role=agent_spec.role,
            task=task,
            status=TeamAgentStatus.RUNNING,
        )

        async_task = asyncio.create_task(runner.run(task))
        self._handles[handle.handle_id] = handle
        self._tasks[handle.handle_id] = async_task
        self._runners[handle.handle_id] = runner

        if self.event_emitter:
            await self.event_emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.TEAM_AGENT_SPAWNED,
                    agent_id=runner.agent_id,
                    agent_name=agent_spec.role,
                    data={"task": task},
                )
            )

        return handle

    async def spawn_parallel(
        self,
        agents: list[tuple[AgentSpec | AgentTeamConfig | Any, str]],
        *,
        max_concurrent: int = 5,
        tools: list[Any] | None = None,
    ) -> list[TeamAgentHandle]:
        """Spawn multiple team agents concurrently, respecting *max_concurrent*."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _spawn_one(spec: AgentSpec | AgentTeamConfig | Any, task: str) -> TeamAgentHandle:
            async with semaphore:
                return await self.spawn(spec, task, tools=tools)

        coros = [_spawn_one(s, t) for s, t in agents]
        result = await asyncio.gather(*coros)
        return list(result)

    async def join(
        self,
        handle: TeamAgentHandle,
        *,
        timeout: float | None = None,
    ) -> TeamAgentResult:
        """Wait for a team agent to finish and return its result."""
        async_task = self._tasks.get(handle.handle_id)
        if async_task is None:
            return TeamAgentResult(
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
            handle.status = TeamAgentStatus.FAILED
            return TeamAgentResult(
                agent_id=handle.agent_id,
                task=handle.task,
                success=False,
                error="Timeout",
            )
        except Exception as exc:
            handle.status = TeamAgentStatus.FAILED
            return TeamAgentResult(
                agent_id=handle.agent_id,
                task=handle.task,
                success=False,
                error=str(exc),
            )

        handle.status = TeamAgentStatus.COMPLETED if result.success else TeamAgentStatus.FAILED

        if self.event_emitter:
            await self.event_emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.TEAM_AGENT_COMPLETED,
                    agent_id=handle.agent_id,
                    agent_name=handle.role,
                    data={"success": result.success, "output": str(result.output)[:200]},
                )
            )

        return TeamAgentResult(
            agent_id=handle.agent_id,
            task=handle.task,
            output=result.output,
            success=result.success,
            error=result.error,
            duration_ms=result.duration_ms,
        )

    async def join_all(
        self,
        handles: list[TeamAgentHandle],
        *,
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> list[TeamAgentResult]:
        """Join all handles concurrently."""
        coros = [self.join(h, timeout=timeout) for h in handles]
        raw = await asyncio.gather(*coros, return_exceptions=return_exceptions)
        results: list[TeamAgentResult] = []
        for r in raw:
            if isinstance(r, BaseException):
                results.append(TeamAgentResult(success=False, error=str(r)))
            else:
                results.append(r)
        return results

    async def terminate(self, handle: TeamAgentHandle) -> None:
        """Cancel a running team agent."""
        task = self._tasks.get(handle.handle_id)
        if task and not task.done():
            task.cancel()
            handle.status = TeamAgentStatus.CANCELLED
