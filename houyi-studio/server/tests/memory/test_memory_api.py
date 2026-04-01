"""Unit tests for Memory API endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from houyi_studio.server.memory.api import router
from houyi_studio.server.memory.service import MemoryService

from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import MemoryCandidate


@pytest.fixture()
def client(tmp_path):
    app = FastAPI()
    app.include_router(router)
    store = MemoryStore(data_dir=str(tmp_path))
    engine = MemoryEngine(store)
    svc = MemoryService(engine)
    c = MemoryCandidate(content="test candidate", confidence=0.8, suggested_tags=["demo"])
    svc.add_candidates([c])
    app.state.memory_service = svc
    app.state.memory_engine = engine
    app.state._test_candidate = c
    return TestClient(app)


class TestCandidateEndpoints:
    def test_list_candidates(self, client):
        r = client.get("/api/memory/candidates")
        assert r.status_code == 200
        assert len(r.json()["candidates"]) >= 1

    def test_list_filter_status(self, client):
        r = client.get("/api/memory/candidates?status=pending")
        assert r.status_code == 200
        assert len(r.json()["candidates"]) >= 1

    def test_update_candidate(self, client):
        cid = client.app.state._test_candidate.candidate_id
        r = client.put(f"/api/memory/candidates/{cid}", json={"content": "edited"})
        assert r.status_code == 200
        assert r.json()["candidate"]["content"] == "edited"

    def test_update_nonexistent(self, client):
        r = client.put("/api/memory/candidates/missing", json={"content": "x"})
        assert r.status_code == 404

    def test_approve_candidate(self, client):
        cid = client.app.state._test_candidate.candidate_id
        r = client.post(f"/api/memory/candidates/{cid}/approve")
        assert r.status_code == 200
        assert "record" in r.json()

    def test_approve_nonexistent(self, client):
        r = client.post("/api/memory/candidates/nonexistent/approve")
        assert r.status_code == 404

    def test_reject_candidate(self, client):
        c2 = MemoryCandidate(content="reject me")
        client.app.state.memory_service.add_candidates([c2])
        r = client.post(f"/api/memory/candidates/{c2.candidate_id}/reject")
        assert r.status_code == 200
        assert r.json()["status"] == "rejected"


class TestRecordEndpoints:
    def test_list_records_empty(self, client):
        r = client.get("/api/memory/records")
        assert r.status_code == 200

    def test_update_nonexistent(self, client):
        r = client.put("/api/memory/records/missing", json={"content": "x"})
        assert r.status_code == 404

    def test_delete_nonexistent(self, client):
        r = client.delete("/api/memory/records/missing")
        assert r.status_code == 404

    def test_list_records_scope(self, client):
        cid = client.app.state._test_candidate.candidate_id
        client.post(f"/api/memory/candidates/{cid}/approve")
        r = client.get("/api/memory/records?scope=user")
        assert r.status_code == 200
        assert len(r.json()["records"]) >= 1

    def test_approve_then_in_list(self, client):
        cid = client.app.state._test_candidate.candidate_id
        r1 = client.post(f"/api/memory/candidates/{cid}/approve")
        assert r1.status_code == 200
        r2 = client.get("/api/memory/records")
        records = r2.json()["records"]
        assert any(rec["content"] == "test candidate" for rec in records)

    def test_recall_history(self, client):
        r = client.get("/api/memory/records/any/recalls")
        assert r.status_code == 200
        assert r.json()["recalls"] == []


class TestExtractEndpoint:
    def test_extract_returns_candidates(self, client):
        messages = [
            {"role": "user", "content": "I prefer Python for data science"},
            {"role": "assistant", "content": "Good choice!"},
        ]
        r = client.post("/api/memory/extract", json={"messages": messages})
        assert r.status_code == 200
        assert "candidates" in r.json()
        assert "count" in r.json()

    def test_extract_empty_messages(self, client):
        r = client.post("/api/memory/extract", json={"messages": []})
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_extract_with_session_id(self, client):
        messages = [{"role": "user", "content": "I like TypeScript too"}]
        r = client.post(
            "/api/memory/extract",
            json={"messages": messages, "session_id": "chat-123"},
        )
        assert r.status_code == 200


class TestConfigEndpoint:
    def test_get_config(self, client):
        r = client.get("/api/memory/config")
        assert r.status_code == 200
        config = r.json()["config"]
        assert config["enabled"] is True
        assert config["auto_extract"] is True

    def test_update_config(self, client):
        r = client.put("/api/memory/config", json={"enabled": False})
        assert r.status_code == 200
        assert r.json()["config"]["enabled"] is False
        assert r.json()["config"]["auto_extract"] is True

    def test_update_auto_extract(self, client):
        r = client.put("/api/memory/config", json={"auto_extract": False})
        assert r.status_code == 200
        assert r.json()["config"]["auto_extract"] is False
