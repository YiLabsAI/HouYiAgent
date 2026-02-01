from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from houyi.protocol.ir.tooling_ir import LLMToolCallOutputIR


class _DummyStore:
    def get(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def save(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def init_execution(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def add(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _DummyTask:
    def done(self) -> bool:
        return True

    def cancel(self) -> None:
        return None


class _RecordingObservationService:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)


@dataclass
class _NodePayload:
    event_id: str
    session_id: str
    execution_id: str
    node_id: str
    status: Any
    inputs: Any
    outputs: Any
    error: str | None


@dataclass
class _RestoreResult:
    event_id: str
    session_id: str
    checkpoint_id: str
    execution_id: str | None
    replay_mode: str | None
    success: bool
    message: str | None


@dataclass
class _Outcome:
    execution_status: Any
    node_statuses: list[Any]
    result: Any


@pytest.mark.asyncio
async def test_restore_checkpoint_emits_node_status_with_dict_outputs() -> None:
    """Regression: checkpoint restore must serialize Pydantic outputs into dict before emitting."""

    from houyi_studio.server.checkpoint_service import CheckpointService

    observation_service = _RecordingObservationService()

    service = CheckpointService(
        checkpoint_store=_DummyStore(),
        execution_store=_DummyStore(),
        plan_store=_DummyStore(),
        observation_service=observation_service,
        execution_tasks={},
        llm_call_logs={},
    )

    # Force restore_checkpoint to produce a node payload whose outputs is a Pydantic model.
    payload = _NodePayload(
        event_id="evt_node",
        session_id="session_1",
        execution_id="exec_1",
        node_id="node_1",
        status="completed",
        inputs={"x": 1},
        outputs=LLMToolCallOutputIR(content="ok"),
        error=None,
    )
    outcome = _Outcome(
        execution_status=None,
        node_statuses=[payload],
        result=_RestoreResult(
            event_id="evt_restore",
            session_id="session_1",
            checkpoint_id="cp_1",
            execution_id="exec_1",
            replay_mode="deterministic",
            success=True,
            message=None,
        ),
    )

    async def _fake_restore_checkpoint(*_args: Any, **_kwargs: Any) -> Any:
        return outcome

    service._manager.restore_checkpoint = _fake_restore_checkpoint  # type: ignore[attr-defined]

    await service.restore_checkpoint(
        session_id="session_1",
        checkpoint_id="cp_1",
        replay_mode="deterministic",
        execution_id="exec_1",
    )

    def _event_type_value(ev: Any) -> str | None:
        event_type = getattr(ev, "event_type", None)
        if event_type is None:
            return None
        value = getattr(event_type, "value", None)
        if isinstance(value, str):
            return value
        if isinstance(event_type, str):
            return event_type
        return None

    node_events = [
        ev for ev in observation_service.events if _event_type_value(ev) == "node_status"
    ]
    assert node_events, "Expected NodeStatusEvent to be emitted"
    assert isinstance(node_events[0].outputs, dict)
