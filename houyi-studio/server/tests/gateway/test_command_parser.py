from __future__ import annotations

from houyi_studio.server.gateway.command_parser import CommandParser
from houyi_studio.server.gateway.commands import ListSkillsCommand, StartExecutionCommand


def test_parse_typed_command_start_execution() -> None:
    parser = CommandParser()
    payload = {
        "command_type": "start_execution",
        "command_id": "cmd_1",
        "session_id": "s1",
        "plan_id": "plan_1",
        "inputs": {},
    }
    parsed = parser.parse(payload)
    assert isinstance(parsed, StartExecutionCommand)
    assert parsed.plan_id == "plan_1"


def test_parse_typed_command_skill_list() -> None:
    parser = CommandParser()
    payload = {
        "command_type": "list_skills",
        "command_id": "cmd_2",
        "session_id": "s1",
    }
    parsed = parser.parse(payload)
    assert isinstance(parsed, ListSkillsCommand)


def test_parse_passthrough_command() -> None:
    parser = CommandParser()
    payload = {
        "command_type": "list_workflows",
        "command_id": "cmd_3",
        "session_id": "s1",
    }
    parsed = parser.parse(payload)
    assert parsed == payload


def test_parse_unknown_command_returns_none() -> None:
    parser = CommandParser()
    parsed = parser.parse({"command_type": "unknown_cmd"})
    assert parsed is None


def test_parse_invalid_payload_returns_none() -> None:
    parser = CommandParser()
    parsed = parser.parse({"command_id": "cmd_4", "session_id": "s1"})
    assert parsed is None
