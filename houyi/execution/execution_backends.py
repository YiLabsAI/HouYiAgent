"""Execution backend adapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from houyi.protocol.ir import ExecutionIR, PlanIR


class ExecutionBackend(Protocol):
    async def start(
        self,
        session_id: str,
        execution: ExecutionIR,
        plan: PlanIR,
        runner: Any,
    ) -> asyncio.Task: ...


@dataclass(slots=True)
class LocalBackend:
    async def start(
        self,
        session_id: str,
        execution: ExecutionIR,
        plan: PlanIR,
        runner: Any,
    ) -> asyncio.Task:
        return asyncio.create_task(runner(session_id, execution, plan))


@dataclass(slots=True)
class DistributedBackend:
    async def start(
        self,
        session_id: str,
        execution: ExecutionIR,
        plan: PlanIR,
        runner: Any,
    ) -> asyncio.Task:
        raise NotImplementedError(
            "Distributed backend not configured. Install/enable the distributed executor adapter."
        )


@dataclass(slots=True)
class SandboxBackend:
    async def start(
        self,
        session_id: str,
        execution: ExecutionIR,
        plan: PlanIR,
        runner: Any,
    ) -> asyncio.Task:
        raise NotImplementedError(
            "Sandbox backend not configured. Install/enable the sandbox executor adapter."
        )
