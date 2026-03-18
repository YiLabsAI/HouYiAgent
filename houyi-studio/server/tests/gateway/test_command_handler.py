from __future__ import annotations

from typing import Any

import pytest
from houyi_studio.server.gateway.command_handler import CommandHandler
from houyi_studio.server.gateway.events import KnowledgeLibraryCreatedEvent, WorkflowListEvent


class FakeWorkflowService:
    def __init__(self) -> None:
        self.saved: tuple[str, object] | None = None

    def save_workflow(self, workflow_name: str, plan: object) -> bool:
        self.saved = (workflow_name, plan)
        return True

    def list_workflows(self) -> list[dict[str, Any]]:
        return [{"name": "wf1", "saved_at": "2026-02-01T00:00:00Z"}]


class FakePlanService:
    def __init__(self, plan: object) -> None:
        self._plan = plan

    def get_current_plan(self, _session_id: str) -> object:
        return self._plan


class FakeEngine:
    def __init__(self, plan: object) -> None:
        self.workflow_service = FakeWorkflowService()
        self.plan_service = FakePlanService(plan)
        self.plans = {"s1": plan}


class FakeKnowledgeService:
    def create_library(
        self,
        *,
        name: str,
        description: str,
        mode: str,
        knowledge_dir: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "library_id": "kb1",
            "name": name,
            "description": description,
            "mode": mode,
            "knowledge_dir": knowledge_dir,
            "metadata": metadata,
        }

    def list_libraries(self) -> list[dict[str, Any]]:
        return []


class FakePlan:
    def __init__(self) -> None:
        self.nodes = [1]
        self.edges = [1]


def _make_handler(
    events: list[object], engine: FakeEngine, knowledge: FakeKnowledgeService
) -> CommandHandler:
    async def send_event(_session_id: str, event: object) -> None:
        events.append(event)

    return CommandHandler(
        send_event=send_event,
        get_execution_engine=lambda: engine,
        sanitize_plan_payload=lambda payload: payload,
        knowledge_service_getter=lambda: knowledge,
    )


@pytest.mark.asyncio
async def test_save_workflow_plan() -> None:
    plan = FakePlan()
    events: list[object] = []
    engine = FakeEngine(plan)
    handler = _make_handler(events, engine, FakeKnowledgeService())

    await handler.handle({"command_type": "save_workflow", "workflow_name": "wf_save"}, "s1")

    assert engine.workflow_service.saved == ("wf_save", plan)
    assert events == []


@pytest.mark.asyncio
async def test_list_workflows_event() -> None:
    events: list[object] = []
    engine = FakeEngine({"nodes": [], "edges": []})
    handler = _make_handler(events, engine, FakeKnowledgeService())

    await handler.handle({"command_type": "list_workflows"}, "s1")

    assert len(events) == 1
    assert isinstance(events[0], WorkflowListEvent)
    assert events[0].workflows[0]["name"] == "wf1"


@pytest.mark.asyncio
async def test_create_library_event() -> None:
    events: list[object] = []
    engine = FakeEngine({"nodes": [], "edges": []})
    handler = _make_handler(events, engine, FakeKnowledgeService())

    await handler.handle(
        {
            "command_type": "create_knowledge_library",
            "name": "KB Name",
            "description": "desc",
            "mode": "auto",
            "knowledge_dir": "./knowledge",
        },
        "s1",
    )

    assert len(events) == 1
    assert isinstance(events[0], KnowledgeLibraryCreatedEvent)
    assert events[0].library["name"] == "KB Name"


def test_can_handle_types() -> None:
    assert CommandHandler.can_handle({"command_type": "list_workflows"})
    assert not CommandHandler.can_handle({"command_type": "unknown_cmd"})
