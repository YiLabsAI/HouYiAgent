"""Thread-safe agent type registry."""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, Field

from houyi.domain.agent.spec import AgentSpec


class AgentTypeConfig(BaseModel):
    """Catalog entry describing an available agent type."""

    agent_type: str = Field(..., description="Unique type identifier (e.g. 'deep_researcher')")
    name: str = Field(..., description="Human-readable name")
    description: str = ""
    icon: str = ""
    status: str = "active"
    default_spec: AgentSpec | None = None
    supported_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRegistry:
    """Thread-safe registry of agent type configs.

    Provides ``register`` / ``get`` / ``list_available`` for the Agent Hub
    and orchestrator to discover available agent types.
    """

    def __init__(self) -> None:
        self._store: dict[str, AgentTypeConfig] = {}
        self._lock = threading.Lock()

    def register(self, agent_type: str, config: AgentTypeConfig) -> None:
        """Register an agent type. Overwrites if already exists."""
        with self._lock:
            self._store[agent_type] = config

    def get(self, agent_type: str) -> AgentTypeConfig | None:
        """Look up a registered agent type, or ``None``."""
        with self._lock:
            return self._store.get(agent_type)

    def list_available(self) -> list[AgentTypeConfig]:
        """Return all registered agent type configs."""
        with self._lock:
            return list(self._store.values())

    def unregister(self, agent_type: str) -> None:
        """Remove an agent type from the registry."""
        with self._lock:
            self._store.pop(agent_type, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, agent_type: str) -> bool:
        with self._lock:
            return agent_type in self._store
