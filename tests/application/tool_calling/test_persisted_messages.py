import json
from unittest.mock import patch

import pytest

from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.tool_calling.persisted_messages import (
    collect_persisted_tool_message_payloads,
)


@pytest.fixture(autouse=True, scope="module")
def _mock_tiktoken():
    """Mock tiktoken to avoid real encoding load, saving ~0.1s per TokenEstimator()."""
    with patch.object(TokenEstimator, "_try_load_encoding", return_value=None):
        yield


class TestCollectPersistedToolMessagePayloads:
    def test_payload_keeps_calls(self):
        messages = collect_persisted_tool_message_payloads(
            intermediate_messages=[
                {
                    "role": "assistant",
                    "content": "calling tools",
                    "reasoning_content": "need file context",
                    "tool_calls": [{"id": "call-1", "type": "function"}],
                }
            ],
            tool_trace=None,
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "calling tools"
        assert messages[0]["reasoning_content"] == "need file context"
        assert messages[0]["tool_calls"] == [{"id": "call-1", "type": "function"}]

    def test_payload_merges_meta(self):
        messages = collect_persisted_tool_message_payloads(
            intermediate_messages=[
                {
                    "role": "tool",
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
        assert messages[0]["role"] == "tool"
        assert messages[0]["name"] == "read_file"
        assert messages[0]["metadata"] == {
            "source": "runner",
            "round_index": 9,
            "parallel_group_id": "grp-1",
            "duration_ms": 22,
        }

    def test_payload_prefers_trace_raw(self):
        messages = collect_persisted_tool_message_payloads(
            intermediate_messages=[
                {
                    "role": "tool",
                    "name": "houyi_find_files",
                    "content": json.dumps(
                        {"_truncated": True, "_truncated_message": "...[truncated]..."}
                    ),
                    "tool_call_id": "call-1",
                    "metadata": {},
                }
            ],
            tool_trace=[
                {
                    "tool_call_id": "call-1",
                    "args": {"pattern": "skill.md", "root": "/repo"},
                    "result": {
                        "raw": {
                            "data": {"matches": ["a", "b"], "path": "/repo"},
                            "success": True,
                        }
                    },
                }
            ],
        )

        assert len(messages) == 1
        payload = json.loads(messages[0]["content"])
        assert payload["data"]["matches"] == ["a", "b"]
        assert messages[0]["metadata"]["tool_args"] == {"pattern": "skill.md", "root": "/repo"}

    def test_payload_compress_search(self):
        messages = collect_persisted_tool_message_payloads(
            intermediate_messages=[
                {
                    "role": "tool",
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
        payload = json.loads(messages[0]["content"])
        assert payload["tool_category"] == "search"
        assert len(payload["results"]) == 2
        assert messages[0]["metadata"]["tool_result_profile"]["compressed"] is True
        assert messages[0]["metadata"]["tool_result_profile"]["tool_category"] == "search"

    def test_payload_compress_read(self):
        messages = collect_persisted_tool_message_payloads(
            intermediate_messages=[
                {
                    "role": "tool",
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

        payload = json.loads(messages[0]["content"])
        assert payload["tool_category"] == "read"
        assert payload["source"] == "https://example.com/page"
        assert "para1" in payload["excerpt"]
        assert "para3" not in payload["excerpt"]

    def test_payload_compress_exec(self):
        messages = collect_persisted_tool_message_payloads(
            intermediate_messages=[
                {
                    "role": "tool",
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

        payload = json.loads(messages[0]["content"])
        assert payload["tool_category"] == "exec"
        assert payload["exit_code"] == 1
        assert "l1" in payload["stdout"]
        assert "l3" not in payload["stdout"]

    def test_payload_compress_fallback(self):
        messages = collect_persisted_tool_message_payloads(
            intermediate_messages=[
                {
                    "role": "tool",
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

        payload = json.loads(messages[0]["content"])
        assert payload["truncated"] is True
        assert messages[0]["metadata"]["tool_result_profile"]["compressed"] is True
