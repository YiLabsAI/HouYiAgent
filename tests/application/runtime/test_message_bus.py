"""Tests for AgentMessageBus: P2P, Pub/Sub, Broadcast."""

from __future__ import annotations

import asyncio

import pytest

from houyi.application.runtime.message_bus import AgentMessage, AgentMessageBus


class TestP2P:
    @pytest.mark.asyncio
    async def test_send_receive(self):
        bus = AgentMessageBus()
        bus.register_agent("a1")
        msg = AgentMessage(sender_id="a0", payload={"q": "hello"})
        await bus.send("a1", msg)
        received = await bus.receive("a1", timeout=1.0)
        assert received.payload["q"] == "hello"

    @pytest.mark.asyncio
    async def test_send_unregistered(self):
        bus = AgentMessageBus()
        with pytest.raises(ValueError, match="not registered"):
            await bus.send("ghost", AgentMessage())

    @pytest.mark.asyncio
    async def test_receive_timeout(self):
        bus = AgentMessageBus()
        bus.register_agent("a1")
        with pytest.raises(TimeoutError):
            await bus.receive("a1", timeout=0.01)


class TestPubSub:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = AgentMessageBus()
        collected: list[AgentMessage] = []

        async def consume():
            async for msg in bus.subscribe("findings", "a1"):
                collected.append(msg)
                break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        await bus.publish("findings", AgentMessage(sender_id="a2", payload={"data": 1}))
        await asyncio.wait_for(task, timeout=1.0)
        assert len(collected) == 1
        assert collected[0].topic == "findings"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = AgentMessageBus()
        bus._topic_subscribers["t1"]["a1"] = asyncio.Queue()
        await bus.unsubscribe("t1", "a1")
        assert "a1" not in bus._topic_subscribers.get("t1", {})


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_all(self):
        bus = AgentMessageBus()
        bus.register_agent("a1")
        bus.register_agent("a2")
        await bus.broadcast(AgentMessage(sender_id="orch"))
        m1 = await bus.receive("a1", timeout=0.1)
        m2 = await bus.receive("a2", timeout=0.1)
        assert m1.sender_id == "orch"
        assert m2.sender_id == "orch"

    @pytest.mark.asyncio
    async def test_broadcast_exclude(self):
        bus = AgentMessageBus()
        bus.register_agent("a1")
        bus.register_agent("a2")
        await bus.broadcast(AgentMessage(sender_id="orch"), exclude={"a1"})
        m2 = await bus.receive("a2", timeout=0.1)
        assert m2.sender_id == "orch"
        with pytest.raises(TimeoutError):
            await bus.receive("a1", timeout=0.01)


class TestLifecycle:
    def test_register_unregister(self):
        bus = AgentMessageBus()
        bus.register_agent("a1")
        assert "a1" in bus.registered_agents
        bus.unregister_agent("a1")
        assert "a1" not in bus.registered_agents

    def test_double_register(self):
        bus = AgentMessageBus()
        bus.register_agent("a1")
        bus.register_agent("a1")
        assert bus.registered_agents.count("a1") == 1
