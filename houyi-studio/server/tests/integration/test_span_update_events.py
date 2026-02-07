from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_start_execution_emits_span_update_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: starting an execution must emit SpanUpdateEvent over WebSocket.

    This is the backend contract required by the Timeline waterfall UI.
    """

    # Force mock streaming path to avoid external network calls.
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "siliconflow")

    from houyi_studio.server.app import app

    session_id = "session_test_span_update"

    plan_payload = {
        "plan_id": "plan_span_update",
        "nodes": [
            {
                "node_id": "node_llm_1",
                "node_type": "llm",
                "position": {"x": 0, "y": 0},
                "config": {
                    "prompt": "hello",
                },
                "inputs": {
                    "prompt": "hello",
                },
                "outputs": {},
                "metadata": {},
            }
        ],
        "edges": [],
        "entry_node_id": "node_llm_1",
        "metadata": {},
    }

    start_command = {
        "command_type": "start_execution",
        "command_id": "cmd_1",
        "session_id": session_id,
        "plan_id": "plan_span_update",
        "inputs": {
            "plan": plan_payload,
            "run_settings": {
                "use_mock_llm": True,
            },
        },
    }

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/session/{session_id}") as ws:
            ws.send_json(start_command)

            deadline = time.time() + 10.0
            seen_span_update = False

            while time.time() < deadline:
                event = ws.receive_json()
                event_type = event.get("event_type")
                if isinstance(event_type, dict):
                    event_type = event_type.get("value")

                if event_type == "span_update":
                    assert event.get("execution_id"), "span_update must include execution_id"
                    assert event.get("span_id"), "span_update must include span_id"
                    assert event.get("trace_id"), "span_update must include trace_id"
                    seen_span_update = True
                    break

            assert seen_span_update, "Expected at least one span_update event after start_execution"
