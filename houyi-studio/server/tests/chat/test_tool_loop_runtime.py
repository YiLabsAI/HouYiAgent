import json

from houyi_studio.server.chat.tool_loop_runtime import (
    build_tool_trace_events,
    collect_persisted_tool_messages,
)
from houyi_studio.server.chat.types import MessageRole


def _event_payload(event: str) -> dict[str, object]:
    data_line = next(line for line in event.splitlines() if line.startswith("data: "))
    return {
        "event": next(
            line.removeprefix("event: ")
            for line in event.splitlines()
            if line.startswith("event: ")
        ),
        "data": json.loads(data_line.removeprefix("data: ")),
    }


class TestBuildToolTraceEvents:
    def test_events_emit_rounds(self):
        events = build_tool_trace_events(
            tool_trace=[
                {
                    "tool_call_id": "call-2",
                    "tool_name": "read_file",
                    "requested_tool_name": "houyi_read_file",
                    "round_index": 1,
                    "parallel_group_id": "grp-1",
                    "duration_ms": 12,
                    "args": {"path": "a.py"},
                    "result": {"raw": {"ok": True}},
                },
                {
                    "tool_call_id": "call-1",
                    "tool_name": "grep_search",
                    "requested_tool_name": "houyi_grep",
                    "round_index": 0,
                    "parallel_group_id": "grp-0",
                    "duration_ms": 8,
                    "args": {"query": "foo"},
                    "result": {"raw": {"matches": 3}},
                },
            ],
            assistant_message_id="msg-1",
            trace_id="trace-1",
        )

        payloads = [_event_payload(event) for event in events]
        assert [payload["event"] for payload in payloads] == [
            "agent.iteration",
            "agent.iteration",
            "tool_call.start",
            "tool_call.result",
            "tool_call.start",
            "tool_call.result",
        ]
        assert payloads[0]["data"]["round_index"] == 0
        assert payloads[1]["data"]["round_index"] == 1
        assert payloads[2]["data"]["tool_call_id"] == "call-2"
        assert payloads[2]["data"]["requested_tool_name"] == "houyi_read_file"
        assert payloads[5]["data"]["result"] == {"matches": 3}
        assert payloads[5]["data"]["requested_tool_name"] == "houyi_grep"

    def test_events_emit_error(self):
        events = build_tool_trace_events(
            tool_trace=[
                {
                    "tool_call_id": "call-1",
                    "tool_name": "read_url",
                    "round_index": 0,
                    "parallel_group_id": "grp-0",
                    "duration_ms": 18,
                    "args": {"url": "https://example.com"},
                    "result": {"raw": {"error": "timeout"}},
                }
            ],
            assistant_message_id="msg-1",
            trace_id="trace-1",
        )

        payloads = [_event_payload(event) for event in events]
        assert [payload["event"] for payload in payloads] == [
            "agent.iteration",
            "tool_call.start",
            "tool_call.error",
        ]
        assert payloads[2]["data"]["error"] == {"error": "timeout"}


class TestCollectPersistedToolMessages:
    def test_persist_keeps_calls(self):
        messages = collect_persisted_tool_messages(
            intermediate_messages=[
                {
                    "role": MessageRole.ASSISTANT.value,
                    "content": "calling tools",
                    "reasoning_content": "need file context",
                    "tool_calls": [{"id": "call-1", "type": "function"}],
                }
            ],
            tool_trace=None,
        )

        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].content == "calling tools"
        assert messages[0].reasoning_content == "need file context"
        assert messages[0].tool_calls == [{"id": "call-1", "type": "function"}]

    def test_persist_merges_meta(self):
        messages = collect_persisted_tool_messages(
            intermediate_messages=[
                {
                    "role": MessageRole.TOOL.value,
                    "name": "read_file",
                    "content": "file body",
                    "tool_call_id": "call-1",
                    "metadata": {"source": "runner", "round_index": 9},
                }
            ],
            tool_trace=[
                {
                    "tool_call_id": "call-1",
                    "round_index": 1,
                    "parallel_group_id": "grp-1",
                    "duration_ms": 22,
                }
            ],
        )

        assert len(messages) == 1
        assert messages[0].role == MessageRole.TOOL
        assert messages[0].name == "read_file"
        assert messages[0].metadata == {
            "source": "runner",
            "round_index": 9,
            "parallel_group_id": "grp-1",
            "duration_ms": 22,
        }

    def test_compress_search(self):
        messages = collect_persisted_tool_messages(
            intermediate_messages=[
                {
                    "role": MessageRole.TOOL.value,
                    "name": "web_search",
                    "content": json.dumps(
                        {
                            "data": {
                                "pattern": "skill.md",
                                "matches": [
                                    {"title": f"title-{idx}", "url": f"https://example.com/{idx}"}
                                    for idx in range(4)
                                ],
                            }
                        }
                    ),
                    "tool_call_id": "call-1",
                }
            ],
            model="gpt-4o-mini",
            tool_result_max_tokens=4096,
            per_tool_quota={"search": 2},
            tool_trace=None,
        )

        assert len(messages) == 1
        payload = json.loads(messages[0].content)
        assert payload["tool_category"] == "search"
        assert len(payload["results"]) == 2
        assert messages[0].metadata["tool_result_profile"]["compressed"] is True
        assert messages[0].metadata["tool_result_profile"]["tool_category"] == "search"

    def test_compress_read(self):
        messages = collect_persisted_tool_messages(
            intermediate_messages=[
                {
                    "role": MessageRole.TOOL.value,
                    "name": "read_url_content",
                    "content": json.dumps(
                        {
                            "data": {
                                "url": "https://example.com/page",
                                "content": "para1\npara2\npara3\npara4",
                            }
                        }
                    ),
                    "tool_call_id": "call-1",
                }
            ],
            model="gpt-4o-mini",
            tool_result_max_tokens=4096,
            per_tool_quota={"read": 2},
            tool_trace=None,
        )

        payload = json.loads(messages[0].content)
        assert payload["tool_category"] == "read"
        assert payload["source"] == "https://example.com/page"
        assert "para1" in payload["excerpt"]
        assert "para3" not in payload["excerpt"]

    def test_compress_exec(self):
        messages = collect_persisted_tool_messages(
            intermediate_messages=[
                {
                    "role": MessageRole.TOOL.value,
                    "name": "shell_exec",
                    "content": json.dumps(
                        {
                            "data": {
                                "stdout": "l1\nl2\nl3\nl4",
                                "stderr": "e1\ne2\ne3",
                                "exit_code": 1,
                            }
                        }
                    ),
                    "tool_call_id": "call-1",
                }
            ],
            model="gpt-4o-mini",
            tool_result_max_tokens=4096,
            per_tool_quota={"exec": 2},
            tool_trace=None,
        )

        payload = json.loads(messages[0].content)
        assert payload["tool_category"] == "exec"
        assert payload["exit_code"] == 1
        assert "l1" in payload["stdout"]
        assert "l3" not in payload["stdout"]

    def test_compress_fallback(self):
        messages = collect_persisted_tool_messages(
            intermediate_messages=[
                {
                    "role": MessageRole.TOOL.value,
                    "name": "read_file",
                    "content": json.dumps({"data": {"content": "x " * 400}}),
                    "tool_call_id": "call-1",
                }
            ],
            model="gpt-4o-mini",
            tool_result_max_tokens=20,
            per_tool_quota={"read": 100},
            tool_trace=None,
        )

        payload = json.loads(messages[0].content)
        assert payload["truncated"] is True
        assert messages[0].metadata["tool_result_profile"]["compressed"] is True
