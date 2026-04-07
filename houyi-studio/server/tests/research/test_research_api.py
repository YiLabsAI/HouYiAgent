from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from houyi_studio.server.research.api import agents_router, router
from houyi_studio.server.research.service import ResearchService

from houyi.adapters.llm.base import LLMAdapter, LLMResponse, StreamChunk
from houyi.application.research.types import ResearchStatus
from houyi.skills.web_search.service import WebSearchService
from houyi.skills.web_search.types import (
    WebSearchMetadata,
    WebSearchResponse,
    WebSearchResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PLAN_JSON = json.dumps(
    {
        "sub_questions": [
            {"question": "Q1", "priority": 5, "search_strategy": "web", "expected_sources": 3}
        ],
        "outline": [{"title": "S1", "objective": "O1", "related_question_ids": [0]}],
        "estimated_duration_min": 5,
    }
)


class _MockLLM(LLMAdapter):
    def __init__(self) -> None:
        self._idx = 0

    async def chat(self, messages: list, **kwargs: Any) -> LLMResponse:
        self._idx += 1
        return LLMResponse(content=_PLAN_JSON, finish_reason="stop", model="mock")

    async def stream_chat(self, messages: list, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk()


def _mock_ws() -> WebSearchService:
    svc = AsyncMock(spec=WebSearchService)
    svc.search = AsyncMock(
        return_value=WebSearchResponse(
            query="q",
            provider="mock",
            results=[WebSearchResult(title="R1", url="https://a.com", snippet="S1")],
            metadata=WebSearchMetadata(
                cached=False, cache_hit=False, latency_ms=5, provider="mock"
            ),
        )
    )
    return svc


@pytest.fixture()
def client(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.include_router(agents_router)
    svc = ResearchService(_MockLLM(), _mock_ws(), data_dir=tmp_path)
    app.state.research_service = svc
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateEndpoint:
    def test_create_201(self, client):
        r = client.post("/api/research/runs", json={"query": "AI frameworks"})
        assert r.status_code == 201
        data = r.json()
        assert "run_id" in data
        assert data["plan"] is not None

    def test_create_idempotent(self, client):
        body = {"query": "Q1", "idempotency_key": "k1"}
        r1 = client.post("/api/research/runs", json=body)
        r2 = client.post("/api/research/runs", json=body)
        assert r1.json()["run_id"] == r2.json()["run_id"]

    def test_create_minimal_body(self, client):
        r = client.post("/api/research/runs", json={"query": "x"})
        assert r.status_code == 201
        assert r.json()["run_id"]


class TestListEndpoint:
    def test_list_empty(self, client):
        r = client.get("/api/research/runs")
        assert r.status_code == 200
        assert r.json()["runs"] == []

    def test_list_after_create(self, client):
        client.post("/api/research/runs", json={"query": "test"})
        r = client.get("/api/research/runs")
        assert len(r.json()["runs"]) == 1

    def test_list_default_mode_delegate(self, client, monkeypatch):
        monkeypatch.delenv("HOUYI_RESEARCH_ORCHESTRATION_MODE", raising=False)
        client.post("/api/research/runs", json={"query": "mode test"})
        r = client.get("/api/research/runs")
        assert r.status_code == 200
        assert r.json()["runs"][0]["orchestration_mode"] == "delegate"

    def test_list_keeps_explicit_mode(self, client):
        client.post(
            "/api/research/runs",
            json={
                "query": "mode direct",
                "settings": {"depth": "quick", "orchestration_mode": "direct"},
            },
        )
        r = client.get("/api/research/runs")
        assert r.status_code == 200
        assert r.json()["runs"][0]["orchestration_mode"] == "direct"

    def test_list_pagination_limit(self, client):
        for q in ["q1", "q2", "q3"]:
            client.post("/api/research/runs", json={"query": q})
        r = client.get("/api/research/runs?offset=0&limit=2")
        assert r.status_code == 200
        assert len(r.json()["runs"]) == 2


class TestGetRun:
    def test_get_existing(self, client):
        r1 = client.post("/api/research/runs", json={"query": "test"})
        run_id = r1.json()["run_id"]
        r2 = client.get(f"/api/research/runs/{run_id}")
        assert r2.status_code == 200
        assert r2.json()["run_id"] == run_id

    def test_get_nonexistent(self, client):
        r = client.get("/api/research/runs/nonexist")
        assert r.status_code == 404


class TestEditPlan:
    def test_edit_bumps_version(self, client):
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        r2 = client.put(
            f"/api/research/runs/{run_id}/plan",
            json={"edits": [{"op": "add", "target_question": "New Q?"}]},
        )
        assert r2.status_code == 200
        assert r2.json()["plan"]["version"] == 2

    def test_edit_version_conflict(self, client):
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        r2 = client.put(
            f"/api/research/runs/{run_id}/plan",
            json={"edits": [], "client_plan_version": 99},
        )
        assert r2.status_code == 409

    def test_edit_nonexistent_404(self, client):
        r = client.put("/api/research/runs/bad/plan", json={"edits": []})
        assert r.status_code == 404


class TestExecuteEndpoint:
    def test_execute_returns_202(self, client):
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        svc = client.app.state.research_service
        svc.launch_run = AsyncMock()
        r2 = client.post(f"/api/research/runs/{run_id}/start", json={})
        assert r2.status_code == 202
        assert r2.json()["status"] == "executing"

    def test_execute_resume_running(self, client):
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        svc = client.app.state.research_service
        svc.get_run(run_id)._status = ResearchStatus.EXECUTING
        r2 = client.post(
            f"/api/research/runs/{run_id}/start",
            json={"resume_if_running": True},
        )
        assert r2.status_code == 202
        assert r2.json()["run_id"] == run_id

    def test_execute_conflict_409(self, client):
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        svc = client.app.state.research_service
        svc.get_run(run_id)._status = ResearchStatus.EXECUTING
        r2 = client.post(f"/api/research/runs/{run_id}/start", json={})
        assert r2.status_code == 409


class TestCancelEndpoint:
    def test_cancel(self, client):
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        r2 = client.post(f"/api/research/runs/{run_id}/cancel", json={"reason": "done"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "cancelled"

    def test_cancel_nonexistent_404(self, client):
        r = client.post("/api/research/runs/bad/cancel", json={})
        assert r.status_code == 404


class TestDeleteEndpoint:
    def test_delete_cancelled(self, client):
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        client.post(f"/api/research/runs/{run_id}/cancel", json={})
        r2 = client.delete(f"/api/research/runs/{run_id}")
        assert r2.status_code == 204

    def test_delete_plan_ready(self, client):
        """Non-executing runs can be deleted."""
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        r2 = client.delete(f"/api/research/runs/{run_id}")
        assert r2.status_code == 204


class TestReportEndpoint:
    def test_report_not_ready(self, client):
        r = client.post("/api/research/runs", json={"query": "test"})
        run_id = r.json()["run_id"]
        r2 = client.get(f"/api/research/runs/{run_id}/report")
        assert r2.status_code == 409


class TestAgentTypes:
    def test_agent_types(self, client):
        r = client.get("/api/agents/types")
        assert r.status_code == 200
        types = r.json()["types"]
        assert len(types) >= 4
        names = [t["name"] for t in types]
        assert "Deep Research" in names
        assert "Code Analyst" in names
        assert "Personal Office" in names
        assert "Data Analysis" in names
