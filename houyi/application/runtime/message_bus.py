"""Agent-to-agent message bus with P2P, Pub/Sub, and Broadcast patterns.

Implements the A2A Pub/Sub draft protocol for inter-agent communication.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentMessageType(str, Enum):
    """Well-known message types for agent communication."""

    TASK_DELEGATE = "task.delegate"
    TASK_RESULT = "task.result"
    TASK_PROGRESS = "task.progress"
    FINDING_PUBLISHED = "finding.published"
    SOURCE_DISCOVERED = "source.discovered"
    QUESTION_COVERED = "question.covered"
    CONFLICT_DETECTED = "conflict.detected"
    STRATEGY_CHANGE = "orchestrator.strategy_change"
    STOP_SEARCH = "orchestrator.stop_search"
    BUDGET_WARNING = "orchestrator.budget_warning"


class AgentMessage(BaseModel):
    """Message exchanged between agents."""

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender_id: str = ""
    message_type: AgentMessageType = AgentMessageType.TASK_DELEGATE
    topic: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    reply_to: str | None = None


_SENTINEL = object()


class AgentMessageBus:
    """In-process message bus for agent-to-agent communication.

    Provides three communication patterns aligned with A2A Pub/Sub draft:

    1. **Point-to-Point**: ``send`` / ``receive`` — for Delegate mode.
    2. **Pub/Sub**: ``publish`` / ``subscribe`` — for Autonomous mode.
    3. **Broadcast**: ``broadcast`` — for global orchestrator directives.

    Current implementation uses ``asyncio.Queue`` for in-process delivery.
    For distributed deployments, the queue backend can be replaced with a
    messaging system (e.g. Redis Streams, NATS, Kafka) by implementing a
    ``MessageBusBackend`` protocol — the public API surface stays identical.
    """

    def __init__(self) -> None:
        self._agent_queues: dict[str, asyncio.Queue[AgentMessage]] = {}
        self._topic_subscribers: dict[str, dict[str, asyncio.Queue[AgentMessage]]] = defaultdict(
            dict
        )
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────

    def register_agent(self, agent_id: str) -> None:
        """Register an agent so it can receive P2P messages."""
        if agent_id not in self._agent_queues:
            self._agent_queues[agent_id] = asyncio.Queue()

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent and clean up all subscriptions."""
        self._agent_queues.pop(agent_id, None)
        for subs in self._topic_subscribers.values():
            subs.pop(agent_id, None)

    @property
    def registered_agents(self) -> list[str]:
        return list(self._agent_queues.keys())

    # ── Point-to-Point ─────────────────────────────────────────

    async def send(self, target_agent_id: str, message: AgentMessage) -> None:
        """Send a message to a specific agent."""
        q = self._agent_queues.get(target_agent_id)
        if q is None:
            raise ValueError(f"Agent {target_agent_id!r} is not registered")
        await q.put(message)

    async def receive(
        self,
        agent_id: str,
        *,
        timeout: float | None = None,
    ) -> AgentMessage:
        """Receive the next P2P message for *agent_id*.

        Raises ``asyncio.TimeoutError`` if *timeout* elapses.
        """
        q = self._agent_queues.get(agent_id)
        if q is None:
            raise ValueError(f"Agent {agent_id!r} is not registered")
        if timeout is not None:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        return await q.get()

    # ── Pub/Sub ────────────────────────────────────────────────

    async def publish(self, topic: str, message: AgentMessage) -> None:
        """Publish *message* to all subscribers of *topic*."""
        msg = message.model_copy(update={"topic": topic})
        subs = self._topic_subscribers.get(topic, {})
        for q in subs.values():
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full on topic %s", topic)

    async def subscribe(self, topic: str, agent_id: str) -> AsyncIterator[AgentMessage]:
        """Subscribe to *topic* and yield messages as they arrive."""
        q: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._topic_subscribers[topic][agent_id] = q
        try:
            while True:
                yield await q.get()
        finally:
            self._topic_subscribers.get(topic, {}).pop(agent_id, None)

    async def unsubscribe(self, topic: str, agent_id: str) -> None:
        """Remove *agent_id* from *topic* subscribers."""
        self._topic_subscribers.get(topic, {}).pop(agent_id, None)

    # ── Broadcast ──────────────────────────────────────────────

    async def broadcast(
        self,
        message: AgentMessage,
        *,
        exclude: set[str] | None = None,
    ) -> None:
        """Send *message* to **all** registered agents (minus *exclude*)."""
        exclude = exclude or set()
        for aid, q in self._agent_queues.items():
            if aid in exclude:
                continue
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("Broadcast queue full for agent %s", aid)
