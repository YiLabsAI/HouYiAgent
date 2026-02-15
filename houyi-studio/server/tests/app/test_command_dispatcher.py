from __future__ import annotations

import pytest
from houyi_studio.server.command_dispatcher import CommandDispatcher, get_command_type_and_id
from houyi_studio.server.commands import ListSkillsCommand


def test_get_command_type_and_id_from_dict() -> None:
    command_type, command_id = get_command_type_and_id(
        {"command_type": "list_skills", "command_id": "cmd_1"}
    )
    assert command_type == "list_skills"
    assert command_id == "cmd_1"


def test_get_command_type_and_id_from_typed_command() -> None:
    command = ListSkillsCommand(command_id="cmd_2", session_id="s1")
    command_type, command_id = get_command_type_and_id(command)
    assert command_type == "list_skills"
    assert command_id == "cmd_2"


@pytest.mark.asyncio
async def test_dispatch_returns_false_when_no_handler() -> None:
    dispatcher = CommandDispatcher()
    handled = await dispatcher.dispatch({"command_type": "x"}, "s1")
    assert handled is False


@pytest.mark.asyncio
async def test_dispatch_invokes_registered_handler() -> None:
    dispatcher = CommandDispatcher()
    state: dict[str, str] = {}

    async def handler(command, session_id: str) -> None:
        state["session_id"] = session_id
        state["command_type"] = command["command_type"]  # type: ignore[index]

    dispatcher.register("list_workflows", handler)
    handled = await dispatcher.dispatch({"command_type": "list_workflows"}, "s42")

    assert handled is True
    assert state == {"session_id": "s42", "command_type": "list_workflows"}
