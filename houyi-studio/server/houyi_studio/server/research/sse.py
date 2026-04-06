"""Research SSE event stream.

Structured envelope format for real-time research progress delivery.
Supports ``Last-Event-ID`` reconnection with replay.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from houyi.application.runtime.events import AgentEvent, EventEmitter

logger = logging.getLogger(__name__)


class ResearchSSEEnvelope:
    """Structured SSE envelope per spec §4.3.

    Fields: event_id, event_type, run_id, sequence, timestamp,
    payload_version, payload.
    """

    __slots__ = (
        "event_id",
        "event_type",
        "payload",
        "payload_version",
        "replayed",
        "run_id",
        "sequence",
        "timestamp",
    )

    def __init__(
        self,
        event_type: str,
        run_id: str,
        sequence: int,
        payload: dict[str, Any],
        *,
        replayed: bool = False,
    ) -> None:
        self.event_id = uuid.uuid4().hex[:16]
        self.event_type = event_type
        self.run_id = run_id
        self.sequence = sequence
        self.timestamp = time.time()
        self.payload_version = 1
        self.payload = payload
        self.replayed = replayed

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "payload_version": self.payload_version,
            "payload": self.payload,
            "replayed": self.replayed,
        }

    def to_sse(self) -> str:
        data = json.dumps(self.to_dict(), default=str)
        return f"id: {self.event_id}\nevent: {self.event_type}\ndata: {data}\n\n"


async def research_sse_stream(
    emitter: EventEmitter,
    run_id: str,
    *,
    last_event_id: str | None = None,
    event_buffer: list[ResearchSSEEnvelope] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted events from the research run emitter.

    Supports reconnection via ``last_event_id``: replays buffered events
    that were emitted after the given event ID.
    """
    buffer: list[ResearchSSEEnvelope] = event_buffer if event_buffer is not None else []
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def _handler(event: AgentEvent) -> None:
        await queue.put(event)

    emitter.on_any(_handler)

    try:
        if last_event_id and buffer:
            replaying = False
            for env in buffer:
                if env.event_id == last_event_id:
                    replaying = True
                    continue
                if replaying:
                    env.replayed = True
                    yield env.to_sse()
        elif not last_event_id and buffer:
            for env in buffer:
                env.replayed = True
                yield env.to_sse()

        yield f": connected run={run_id}\n\n"

        heartbeat_interval = 15.0
        last_heartbeat = time.monotonic()

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
            except TimeoutError:
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    hb = ResearchSSEEnvelope(
                        event_type="research.heartbeat",
                        run_id=run_id,
                        sequence=0,
                        payload={"status": "alive", "elapsed_seconds": round(now, 2)},
                    )
                    buffer.append(hb)
                    yield hb.to_sse()
                    last_heartbeat = now
                continue

            research_event = event.data.get("research_event", "unknown")
            sequence = event.data.get("sequence", 0)
            payload = {
                k: v for k, v in event.data.items() if k not in ("research_event", "sequence")
            }

            envelope = ResearchSSEEnvelope(
                event_type=research_event,
                run_id=run_id,
                sequence=sequence,
                payload=payload,
            )
            if event_buffer is None:
                buffer.append(envelope)
            yield envelope.to_sse()

            if research_event in ("research.completed", "research.failed", "research.cancelled"):
                break

    finally:
        emitter.off_any(_handler)
