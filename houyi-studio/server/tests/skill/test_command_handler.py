from __future__ import annotations

from typing import Any

import pytest
from houyi_studio.server.gateway.commands import (
    ConfigureSkillCommand,
    DryRunSkillCommand,
    GetSkillDetailCommand,
    ListSkillsCommand,
)
from houyi_studio.server.gateway.events import (
    DryRunResultEvent,
    SkillConfiguredEvent,
    SkillDetailEvent,
    SkillErrorEvent,
    SkillListEvent,
)
from houyi_studio.server.skill.command_handler import SkillCommandHandler


class FakeSkillService:
    def __init__(self) -> None:
        self._detail: dict[str, Any] | None = None

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "planner",
                "display_name": "Planner",
                "description": "Planning helper",
                "tools": ["plan"],
                "policy_action": "allow",
                "side_effect": "none",
                "certification": "silver",
                "source": "builtin",
            }
        ]

    def get_skill_detail(self, skill_name: str) -> dict[str, Any] | None:
        return self._detail if self._detail and self._detail.get("name") == skill_name else None

    def get_skill_metrics(self, skill_name: str) -> dict[str, Any] | None:
        return None

    def load_skill(
        self,
        source: str,
        install_strategy: str | None = None,
    ) -> tuple[bool, str, str | None]:
        return True, "planner", None

    def unload_skill(self, skill_name: str) -> tuple[bool, str | None]:
        return True, None

    def configure_skill(
        self, skill_name: str, policy_action: str | None = None, auto_invoke: bool | None = None
    ) -> tuple[bool, str | None]:
        if skill_name == "missing":
            return False, "Skill not found"
        return True, None

    async def dry_run(
        self,
        skill_name: str,
        tool_name: str,
        input_data: dict[str, Any],
        live: bool = False,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> dict[str, Any]:
        return {
            "valid": True,
            "schema_errors": [],
            "policy_result": "allow",
            "capability_gaps": [],
            "estimated_side_effects": [],
        }

    def respond_to_consent(self, request_id: str, granted: bool, remember: bool) -> bool:
        return True


@pytest.mark.asyncio
async def test_list_skills_sends_skill_list_event() -> None:
    events: list[object] = []
    service = FakeSkillService()

    async def send_event(session_id: str, event: object) -> None:
        events.append(event)

    handler = SkillCommandHandler(send_event=send_event, skill_service_getter=lambda: service)
    command = ListSkillsCommand(command_id="cmd_1", session_id="s1")

    await handler.handle(command, "s1")

    assert len(events) == 1
    assert isinstance(events[0], SkillListEvent)
    assert events[0].skills[0].name == "planner"
    assert events[0].skills[0].source == "builtin"


@pytest.mark.asyncio
async def test_get_skill_detail_not_found_sends_error_event() -> None:
    events: list[object] = []
    service = FakeSkillService()

    async def send_event(session_id: str, event: object) -> None:
        events.append(event)

    handler = SkillCommandHandler(send_event=send_event, skill_service_getter=lambda: service)
    command = GetSkillDetailCommand(command_id="cmd_2", session_id="s1", skill_name="missing")

    await handler.handle(command, "s1")

    assert len(events) == 1
    assert isinstance(events[0], SkillErrorEvent)
    assert events[0].error_code == "skill_not_found"


@pytest.mark.asyncio
async def test_get_skill_detail_defaults_version_when_none() -> None:
    events: list[object] = []
    service = FakeSkillService()
    service._detail = {
        "name": "planner",
        "display_name": "Planner",
        "version": None,
        "tools": [],
        "permissions": [],
        "policy": {},
        "hooks": [],
        "certification": "silver",
        "side_effect": "none",
        "source": "community",
    }

    async def send_event(session_id: str, event: object) -> None:
        events.append(event)

    handler = SkillCommandHandler(send_event=send_event, skill_service_getter=lambda: service)
    command = GetSkillDetailCommand(command_id="cmd_3", session_id="s1", skill_name="planner")

    await handler.handle(command, "s1")

    assert len(events) == 1
    assert isinstance(events[0], SkillDetailEvent)
    assert events[0].skill.version is None
    assert events[0].skill.source == "community"


@pytest.mark.asyncio
async def test_configure_skill_sends_configured_event() -> None:
    events: list[object] = []
    service = FakeSkillService()

    async def send_event(session_id: str, event: object) -> None:
        events.append(event)

    handler = SkillCommandHandler(send_event=send_event, skill_service_getter=lambda: service)
    command = ConfigureSkillCommand(
        command_id="cmd_4",
        session_id="s1",
        skill_name="planner",
        policy_action="allow_with_consent",
        auto_invoke=False,
    )

    await handler.handle(command, "s1")

    assert len(events) == 1
    assert isinstance(events[0], SkillConfiguredEvent)
    assert events[0].policy_action == "allow_with_consent"
    assert events[0].auto_invoke is False


@pytest.mark.asyncio
async def test_dry_run_sends_result_event() -> None:
    events: list[object] = []
    service = FakeSkillService()

    async def send_event(session_id: str, event: object) -> None:
        events.append(event)

    handler = SkillCommandHandler(send_event=send_event, skill_service_getter=lambda: service)
    command = DryRunSkillCommand(
        command_id="cmd_5",
        session_id="s1",
        skill_name="planner",
        tool_name="plan",
        input={},
    )

    await handler.handle(command, "s1")

    assert len(events) == 1
    assert isinstance(events[0], DryRunResultEvent)
    assert events[0].result.valid is True
