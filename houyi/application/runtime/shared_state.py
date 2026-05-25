"""Pluggable shared-state backend for orchestrator collaboration."""

from __future__ import annotations

import asyncio
import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StateChange(BaseModel):
    """Single state mutation record."""

    state_id: str
    key: str
    old_value: Any = None
    new_value: Any = None


class OrchestratorState(BaseModel):
    """Mutable collaboration state shared among orchestrated agents.

    List fields use *append* semantics on write; scalar fields are overwritten.
    """

    state_id: str = ""
    task: str = ""
    status: str = "pending"
    agent_results: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SharedStateBackend(ABC):
    """Pluggable backend for orchestrator shared state.

    Default: in-memory (dict + asyncio.Lock) for same-process agents.
    Future: Redis / SQLite for cross-process scenarios.
    """

    @abstractmethod
    async def read(self, state_id: str) -> OrchestratorState:
        """Read the current state snapshot."""

    @abstractmethod
    async def write(self, state_id: str, updates: dict[str, Any]) -> None:
        """Atomic partial update using reducer semantics.

        - list fields: *append* items from the update list.
        - Scalar / dict fields: *overwrite* with the update value.
        """

    @abstractmethod
    def watch(self, state_id: str) -> AsyncIterator[StateChange]:
        """Subscribe to state changes (for Autonomous mode monitoring).

        Implementations should be async generators (async def with yield).
        The return type is AsyncIterator so that mypy treats the abstract
        signature and the concrete async-generator implementation consistently.
        """


class InMemoryStateBackend(SharedStateBackend):
    """Default in-process backend backed by dict + asyncio.Lock."""

    def __init__(self) -> None:
        self._store: dict[str, OrchestratorState] = {}
        self._lock = asyncio.Lock()
        self._watchers: dict[str, list[asyncio.Queue[StateChange]]] = {}

    async def read(self, state_id: str) -> OrchestratorState:
        async with self._lock:
            state = self._store.get(state_id)
            if state is None:
                state = OrchestratorState(state_id=state_id)
                self._store[state_id] = state
            return copy.deepcopy(state)

    async def write(self, state_id: str, updates: dict[str, Any]) -> None:
        async with self._lock:
            state = self._store.get(state_id)
            if state is None:
                state = OrchestratorState(state_id=state_id)
                self._store[state_id] = state

            changes: list[StateChange] = []
            for key, value in updates.items():
                if not hasattr(state, key):
                    continue
                old = getattr(state, key)
                if isinstance(old, list) and isinstance(value, list):
                    old.extend(value)
                    new = old
                else:
                    setattr(state, key, value)
                    new = value
                changes.append(
                    StateChange(state_id=state_id, key=key, old_value=old, new_value=new)
                )

        for ch in changes:
            await self._notify(state_id, ch)

    async def watch(self, state_id: str) -> AsyncIterator[StateChange]:
        queue: asyncio.Queue[StateChange] = asyncio.Queue()
        self._watchers.setdefault(state_id, []).append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._watchers.get(state_id, []).remove(queue) if queue in self._watchers.get(
                state_id, []
            ) else None

    async def delete(self, state_id: str) -> None:
        """Remove a state entry (useful in tests)."""
        async with self._lock:
            self._store.pop(state_id, None)

    async def _notify(self, state_id: str, change: StateChange) -> None:
        for q in self._watchers.get(state_id, []):
            try:
                q.put_nowait(change)
            except asyncio.QueueFull:
                logger.warning("Watcher queue full for state %s", state_id)
