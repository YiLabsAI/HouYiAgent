"""Unit tests for Research SSE event stream."""

from __future__ import annotations

import asyncio
import itertools
import json

from houyi_studio.server.research.sse import ResearchSSEEnvelope, research_sse_stream

from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter


class TestSSEEnvelope:
    def test_envelope_fields(self):
        env = ResearchSSEEnvelope(
            event_type="research.plan_generated",
            session_id="s1",
            sequence=1,
            payload={"plan": {}},
        )
        assert env.event_type == "research.plan_generated"
        assert env.session_id == "s1"
        assert env.sequence == 1
        assert env.payload_version == 1
        assert not env.replayed

    def test_to_dict(self):
        env = ResearchSSEEnvelope("t", "s", 1, {})
        d = env.to_dict()
        assert "event_id" in d
        assert "timestamp" in d

    def test_to_sse_format(self):
        env = ResearchSSEEnvelope("research.test", "s1", 1, {"a": 1})
        sse = env.to_sse()
        assert sse.startswith("id: ")
        assert "event: research.test" in sse
        assert "data: " in sse
        assert sse.endswith("\n\n")

    def test_sequence_monotonic(self):
        envelopes = [ResearchSSEEnvelope("research.step", "s1", i, {}) for i in range(1, 5)]
        sequences = [e.sequence for e in envelopes]
        for a, b in itertools.pairwise(sequences):
            assert a < b


class TestSSEStream:
    async def test_connect_message(self):
        emitter = EventEmitter()

        async def _emit_complete():
            await asyncio.sleep(0.01)
            await emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.PROGRESS,
                    agent_id="s1",
                    data={"research_event": "research.completed", "sequence": 1},
                )
            )

        asyncio.create_task(_emit_complete())
        chunks = []
        async for chunk in research_sse_stream(emitter, "s1"):
            chunks.append(chunk)
        assert any("connected" in c for c in chunks)
        assert any("research.completed" in c for c in chunks)

    async def test_event_ordering(self):
        emitter = EventEmitter()

        events_to_emit = [
            ("research.plan_generated", 1),
            ("research.step_started", 2),
            ("research.step_completed", 3),
            ("research.completed", 4),
        ]

        async def _emit_events():
            await asyncio.sleep(0.01)
            for evt, seq in events_to_emit:
                await emitter.emit(
                    AgentEvent(
                        event_type=AgentEventType.PROGRESS,
                        agent_id="s1",
                        data={"research_event": evt, "sequence": seq},
                    )
                )

        asyncio.create_task(_emit_events())
        chunks = []
        async for chunk in research_sse_stream(emitter, "s1"):
            chunks.append(chunk)

        event_chunks = [c for c in chunks if "event: research." in c]
        assert len(event_chunks) == 4

    async def test_terminates_on_failed(self):
        emitter = EventEmitter()

        async def _emit():
            await asyncio.sleep(0.01)
            await emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.PROGRESS,
                    agent_id="s1",
                    data={"research_event": "research.failed", "sequence": 1, "error": "boom"},
                )
            )

        asyncio.create_task(_emit())
        chunks = []
        async for chunk in research_sse_stream(emitter, "s1"):
            chunks.append(chunk)
        assert any("research.failed" in c for c in chunks)

    async def test_replay_buffered_events(self):
        emitter = EventEmitter()
        buffer = [
            ResearchSSEEnvelope("research.plan_generated", "s1", 1, {}),
            ResearchSSEEnvelope("research.step_started", "s1", 2, {}),
        ]
        target_event_id = buffer[0].event_id

        async def _emit():
            await asyncio.sleep(0.01)
            await emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.PROGRESS,
                    agent_id="s1",
                    data={"research_event": "research.completed", "sequence": 3},
                )
            )

        asyncio.create_task(_emit())
        chunks = []
        async for chunk in research_sse_stream(
            emitter,
            "s1",
            last_event_id=target_event_id,
            event_buffer=buffer,
        ):
            chunks.append(chunk)
        replayed = [c for c in chunks if "research.step_started" in c]
        assert len(replayed) >= 1

    async def test_replay_empty_buffer(self):
        emitter = EventEmitter()

        async def _emit():
            await asyncio.sleep(0.01)
            await emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.PROGRESS,
                    agent_id="s1",
                    data={"research_event": "research.completed", "sequence": 1},
                )
            )

        asyncio.create_task(_emit())
        chunks = []
        async for chunk in research_sse_stream(
            emitter, "s1", last_event_id="missing", event_buffer=[]
        ):
            chunks.append(chunk)
        replayed = [c for c in chunks if '"replayed": true' in c]
        assert len(replayed) == 0

    async def test_cancelled_terminates(self):
        emitter = EventEmitter()

        async def _emit():
            await asyncio.sleep(0.01)
            await emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.PROGRESS,
                    agent_id="s1",
                    data={
                        "research_event": "research.cancelled",
                        "sequence": 1,
                        "reason": "user",
                    },
                )
            )

        asyncio.create_task(_emit())
        chunks = []
        async for chunk in research_sse_stream(emitter, "s1"):
            chunks.append(chunk)
        assert any("research.cancelled" in c for c in chunks)

    async def test_stream_seq_monotonic(self):
        emitter = EventEmitter()

        async def _emit():
            await asyncio.sleep(0.01)
            for seq in [1, 2, 3]:
                await emitter.emit(
                    AgentEvent(
                        event_type=AgentEventType.PROGRESS,
                        agent_id="s1",
                        data={"research_event": "research.step_started", "sequence": seq},
                    )
                )
            await emitter.emit(
                AgentEvent(
                    event_type=AgentEventType.PROGRESS,
                    agent_id="s1",
                    data={"research_event": "research.completed", "sequence": 4},
                )
            )

        asyncio.create_task(_emit())
        sequences = []
        async for chunk in research_sse_stream(emitter, "s1"):
            for line in chunk.split("\n"):
                if line.startswith("data: "):
                    d = json.loads(line[6:])
                    if d.get("sequence", 0) > 0:
                        sequences.append(d["sequence"])
        for a, b in itertools.pairwise(sequences):
            assert a < b
