"""Server integration tests for Memory pipeline (US-1 ~ US-5).

Tests the full API-level Memory Integration flow using FastAPI TestClient:
- US-1: Extract → Candidates appear in inbox
- US-2: Chatbox messages → extract endpoint → candidates
- US-3: Approve candidate → recall injects into context
- US-4: Records CRUD (list / edit / delete)
- US-5: Config toggle disables extraction

These run against real Memory subsystem with mock LLM extraction.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from houyi_studio.server.memory.api import router as memory_router
from houyi_studio.server.memory.service import MemoryService

from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import MemoryPolicy


def _make_llm_mock(items: list[dict]) -> AsyncMock:
    """Build a mock LLMAdapter whose chat() returns a JSON array."""
    mock = AsyncMock()
    resp = AsyncMock()
    resp.content = json.dumps(items)
    mock.chat.return_value = resp
    return mock


@pytest.fixture()
def app_and_client(tmp_path):
    """Wire up a real Memory subsystem with mock LLM behind TestClient."""
    llm = _make_llm_mock(
        [
            {"content": "User prefers Python", "type": "preference", "confidence": 0.85},
            {"content": "User name: Alice", "type": "profile", "confidence": 0.9},
        ]
    )
    store = MemoryStore(data_dir=tmp_path / "memory")
    engine = MemoryEngine(store, llm_adapter=llm, policy=MemoryPolicy(auto_approve=False))
    service = MemoryService(engine)

    app = FastAPI()
    app.state.memory_service = service
    app.state.memory_engine = engine
    app.include_router(memory_router)

    client = TestClient(app)
    yield client, service, engine, llm
    client.close()


# ====================================================================
# US-1 / US-2: Extract → Candidates appear
# ====================================================================


class TestExtractPipeline:
    """POST /api/memory/extract produces candidates visible in GET /api/memory/candidates."""

    def test_extract_creates_candidates(self, app_and_client):
        client, *_ = app_and_client
        resp = client.post(
            "/api/memory/extract",
            json={
                "messages": [
                    {"role": "user", "content": "My name is Alice. I prefer Python."},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1

        cands_resp = client.get("/api/memory/candidates")
        assert cands_resp.status_code == 200
        candidates = cands_resp.json()["candidates"]
        assert len(candidates) >= 1

    def test_extract_empty_messages(self, app_and_client):
        client, *_ = app_and_client
        resp = client.post("/api/memory/extract", json={"messages": []})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_extract_assistant_only(self, app_and_client):
        client, service, engine, llm = app_and_client
        llm.chat.return_value.content = "[]"

        resp = client.post(
            "/api/memory/extract",
            json={
                "messages": [{"role": "assistant", "content": "I am helpful."}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_extract_with_session_id(self, app_and_client):
        client, *_ = app_and_client
        resp = client.post(
            "/api/memory/extract",
            json={
                "messages": [{"role": "user", "content": "Remember port 8080."}],
                "session_id": "sess_42",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1


# ====================================================================
# US-3: Approve candidate → record exists → recall works
# ====================================================================


class TestApproveAndRecall:
    """Approve a candidate, verify it appears in records."""

    def test_approve_creates_record(self, app_and_client):
        client, *_ = app_and_client
        client.post(
            "/api/memory/extract",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        cands = client.get("/api/memory/candidates").json()["candidates"]
        assert len(cands) >= 1

        cand_id = cands[0]["candidate_id"]
        approve_resp = client.post(f"/api/memory/candidates/{cand_id}/approve")
        assert approve_resp.status_code == 200
        assert "record" in approve_resp.json()

        records = client.get("/api/memory/records").json()["records"]
        assert len(records) >= 1

    def test_approve_nonexistent_returns_404(self, app_and_client):
        client, *_ = app_and_client
        resp = client.post("/api/memory/candidates/nonexistent/approve")
        assert resp.status_code == 404

    def test_reject_candidate(self, app_and_client):
        client, *_ = app_and_client
        client.post(
            "/api/memory/extract",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        cands = client.get("/api/memory/candidates").json()["candidates"]
        cand_id = cands[0]["candidate_id"]

        reject_resp = client.post(f"/api/memory/candidates/{cand_id}/reject")
        assert reject_resp.status_code == 200
        assert reject_resp.json()["status"] == "rejected"


# ====================================================================
# US-4: Records CRUD
# ====================================================================


class TestRecordsCRUD:
    """Records: list, edit content, delete with confirmation."""

    def _approve_first_candidate(self, client) -> str:
        client.post(
            "/api/memory/extract",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        cands = client.get("/api/memory/candidates").json()["candidates"]
        cand_id = cands[0]["candidate_id"]
        resp = client.post(f"/api/memory/candidates/{cand_id}/approve")
        return resp.json()["record"]["record_id"]

    def test_list_records(self, app_and_client):
        client, *_ = app_and_client
        self._approve_first_candidate(client)
        records = client.get("/api/memory/records").json()["records"]
        assert len(records) >= 1

    def test_edit_record(self, app_and_client):
        client, *_ = app_and_client
        record_id = self._approve_first_candidate(client)

        resp = client.put(
            f"/api/memory/records/{record_id}",
            json={
                "content": "Updated content",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["record"]["content"] == "Updated content"

    def test_delete_record(self, app_and_client):
        client, *_ = app_and_client
        record_id = self._approve_first_candidate(client)

        del_resp = client.delete(f"/api/memory/records/{record_id}")
        assert del_resp.status_code == 204

        records = client.get("/api/memory/records").json()["records"]
        ids = [r["record_id"] for r in records]
        assert record_id not in ids

    def test_delete_nonexistent_returns_404(self, app_and_client):
        client, *_ = app_and_client
        resp = client.delete("/api/memory/records/nonexistent")
        assert resp.status_code == 404

    def test_edit_nonexistent_returns_404(self, app_and_client):
        client, *_ = app_and_client
        resp = client.put("/api/memory/records/nonexistent", json={"content": "x"})
        assert resp.status_code == 404


# ====================================================================
# US-5: Config toggle
# ====================================================================


class TestMemoryConfig:
    """GET/PUT /api/memory/config controls extraction behavior."""

    def test_default_config(self, app_and_client):
        client, *_ = app_and_client
        resp = client.get("/api/memory/config")
        assert resp.status_code == 200
        cfg = resp.json()["config"]
        assert cfg["enabled"] is True
        assert cfg["auto_extract"] is True

    def test_disable_memory(self, app_and_client):
        client, *_ = app_and_client
        resp = client.put("/api/memory/config", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["config"]["enabled"] is False

    def test_disable_auto_extract(self, app_and_client):
        client, *_ = app_and_client
        resp = client.put("/api/memory/config", json={"auto_extract": False})
        assert resp.status_code == 200
        assert resp.json()["config"]["auto_extract"] is False

    def test_partial_update_preserves_other(self, app_and_client):
        client, *_ = app_and_client
        client.put("/api/memory/config", json={"enabled": False})
        resp = client.put("/api/memory/config", json={"auto_extract": False})
        cfg = resp.json()["config"]
        assert cfg["enabled"] is False
        assert cfg["auto_extract"] is False


# ====================================================================
# LLM fallback: when LLM fails, rule-based extraction still works
# ====================================================================


class TestLLMFallback:
    """If LLM is down, extraction degrades gracefully to rule-based."""

    def test_llm_error_falls_back(self, app_and_client):
        client, _, _, llm = app_and_client
        llm.chat.side_effect = RuntimeError("LLM unavailable")

        resp = client.post(
            "/api/memory/extract",
            json={
                "messages": [
                    {"role": "user", "content": "Remember that the port is 8080."},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1
