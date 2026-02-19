"""Unit tests for ExecutionCommandHandler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from houyi_studio.server.execution.command_handler import ExecutionCommandHandler
from houyi_studio.server.gateway.commands import (
    AbortCommand,
    PauseCommand,
    RestoreCheckpointCommand,
    ResumeCommand,
    SetLogLevelCommand,
)
from houyi_studio.server.gateway.events import LogLevelEvent


class FakeEngine:
    def __init__(self) -> None:
        self.pause_execution = AsyncMock()
        self.resume_execution = AsyncMock()
        self.abort_execution = AsyncMock()
        self.retry_node = AsyncMock()
        self.start_execution = AsyncMock()
        self.restore_checkpoint = AsyncMock()
        self.plan_service = MagicMock()


def _make_handler(
    events: list[object],
    engine: FakeEngine,
    *,
    patch_return: bool = False,
) -> ExecutionCommandHandler:
    async def send_event(_session_id: str, event: object) -> None:
        events.append(event)

    return ExecutionCommandHandler(
        send_event=send_event,
        get_execution_engine=lambda: engine,
        apply_plan_patches=lambda _plan, _patches: patch_return,
    )


def test_can_handle_typed_commands() -> None:
    cmd = PauseCommand(
        command_type="pause",
        command_id="cmd_1",
        session_id="s1",
        execution_id="exec_1",
    )
    assert ExecutionCommandHandler.can_handle(cmd)
    assert not ExecutionCommandHandler.can_handle({"command_type": "list_workflows"})


@pytest.mark.asyncio
async def test_pause_delegates() -> None:
    events: list[object] = []
    engine = FakeEngine()
    handler = _make_handler(events, engine)
    cmd = PauseCommand(
        command_type="pause",
        command_id="cmd_1",
        session_id="s1",
        execution_id="exec_1",
    )
    await handler.handle(cmd, "s1")
    engine.pause_execution.assert_awaited_once_with("exec_1")


@pytest.mark.asyncio
async def test_resume_delegates() -> None:
    events: list[object] = []
    engine = FakeEngine()
    handler = _make_handler(events, engine)
    cmd = ResumeCommand(
        command_type="resume",
        command_id="cmd_2",
        session_id="s1",
        execution_id="exec_1",
    )
    await handler.handle(cmd, "s1")
    engine.resume_execution.assert_awaited_once_with("exec_1")


@pytest.mark.asyncio
async def test_abort_delegates() -> None:
    events: list[object] = []
    engine = FakeEngine()
    handler = _make_handler(events, engine)
    cmd = AbortCommand(
        command_type="abort",
        command_id="cmd_3",
        session_id="s1",
        execution_id="exec_1",
    )
    await handler.handle(cmd, "s1")
    engine.abort_execution.assert_awaited_once_with("exec_1")


@pytest.mark.asyncio
async def test_set_log_level_sends_event() -> None:
    events: list[object] = []
    engine = FakeEngine()
    handler = _make_handler(events, engine)
    cmd = SetLogLevelCommand(
        command_type="set_log_level",
        command_id="cmd_4",
        session_id="s1",
        level="DEBUG",
    )
    await handler.handle(cmd, "s1")
    assert len(events) == 1
    assert isinstance(events[0], LogLevelEvent)
    assert events[0].requested_level == "DEBUG"


@pytest.mark.asyncio
async def test_restore_checkpoint_delegates() -> None:
    events: list[object] = []
    engine = FakeEngine()
    handler = _make_handler(events, engine)
    cmd = RestoreCheckpointCommand(
        command_type="restore_checkpoint",
        command_id="cmd_5",
        session_id="s1",
        execution_id="exec_1",
        checkpoint_id="cp_1",
        replay_mode="deterministic",
    )
    await handler.handle(cmd, "s1")
    engine.restore_checkpoint.assert_awaited_once()
