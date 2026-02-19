"""Lifecycle hook protocol for execution engine."""

from __future__ import annotations

from typing import Any, Protocol

from .context import ExecutionContext


class LifecycleHook(Protocol):
    async def on_execution_start(self, context: ExecutionContext) -> None: ...

    async def on_execution_end(self, context: ExecutionContext) -> None: ...

    async def before_node(
        self,
        context: ExecutionContext,
        node_id: str,
        node_exec: Any,
    ) -> None: ...

    async def after_node(
        self,
        context: ExecutionContext,
        node_id: str,
        node_exec: Any,
    ) -> None: ...
