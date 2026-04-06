from types import SimpleNamespace

from houyi.application.context.compaction_summary import (
    build_compaction_summary,
    summarize_compaction_message,
    summarize_tool_calls,
    summarize_tool_payload,
)


class _Role:
    def __init__(self, value: str):
        self.value = value


def _message(role: str, **kwargs):
    return SimpleNamespace(role=_Role(role), **kwargs)


class TestBuildCompactionSummary:
    def test_summary_tool_loop(self):
        summary = build_compaction_summary(
            [
                _message(message_id="u1", role="user", content="find skill.md"),
                _message(
                    message_id="a1",
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "code_search",
                                "arguments": '{"query":"skill.md"}',
                            },
                        }
                    ],
                ),
                _message(
                    message_id="t1",
                    role="tool",
                    name="code_search",
                    tool_call_id="call_1",
                    content='{"data":{"matches":[],"pattern":"skill.md","root_path":"/tmp/repo","truncated":false},"meta":{"ok":true}}',
                ),
            ]
        )

        assert "assistant: [tool loop: code_search]" in summary
        assert "tool: code_search search 'skill.md' returned 0 match(es)" in summary
        assert "root_path" not in summary
        assert '"truncated"' not in summary


class TestSummarizeToolCalls:
    def test_none_returns_loop(self) -> None:
        assert summarize_tool_calls(None) == "[tool loop]"

    def test_empty_list(self) -> None:
        assert summarize_tool_calls([]) == "[tool loop]"

    def test_non_dict_items_skipped(self) -> None:
        result = summarize_tool_calls(["not_a_dict"])
        assert "1 call(s)" in result

    def test_missing_function_key(self) -> None:
        result = summarize_tool_calls([{"id": "c1"}])
        assert "1 call(s)" in result

    def test_two_unique_names(self) -> None:
        calls = [
            {"function": {"name": "search"}},
            {"function": {"name": "browse"}},
        ]
        result = summarize_tool_calls(calls)
        assert "search" in result
        assert "browse" in result

    def test_more_than_two_unique(self) -> None:
        calls = [
            {"function": {"name": "a"}},
            {"function": {"name": "b"}},
            {"function": {"name": "c"}},
        ]
        result = summarize_tool_calls(calls)
        assert "+1 more" in result


class TestSummarizeToolPayload:
    def test_empty_content(self) -> None:
        msg = {"role": "tool", "name": "grep", "content": "  "}
        result = summarize_tool_payload(msg)
        assert "empty result" in result

    def test_non_json_content(self) -> None:
        msg = {"role": "tool", "name": "run", "content": "plain text output"}
        result = summarize_tool_payload(msg)
        assert "run:" in result

    def test_list_payload(self) -> None:
        msg = {"role": "tool", "name": "ls", "content": "[1,2,3]"}
        result = summarize_tool_payload(msg)
        assert "3 item(s)" in result

    def test_dict_error_branch(self) -> None:
        msg = {"role": "tool", "name": "api", "content": '{"error": "not found"}'}
        result = summarize_tool_payload(msg)
        assert "error" in result

    def test_dict_useful_keys(self) -> None:
        msg = {"role": "tool", "name": "q", "content": '{"status": "ok", "value": 1}'}
        result = summarize_tool_payload(msg)
        assert "result keys" in result

    def test_dict_no_useful_keys(self) -> None:
        msg = {"role": "tool", "name": "q", "content": '{"data": "x"}'}
        result = summarize_tool_payload(msg)
        # data is a dict key but "data" string is not a dict, so falls through
        assert "structured result" in result or "q:" in result

    def test_data_with_count(self) -> None:
        msg = {"role": "tool", "name": "db", "content": '{"data": {"count": 42}}'}
        result = summarize_tool_payload(msg)
        assert "42 item(s)" in result

    def test_data_useful_keys(self) -> None:
        msg = {
            "role": "tool",
            "name": "fs",
            "content": '{"data": {"files": [], "dirs": []}}',
        }
        result = summarize_tool_payload(msg)
        assert "fields" in result


class TestSummarizeCompactionMsg:
    def test_role_with_content(self) -> None:
        msg = {"role": "user", "content": "hello world"}
        result = summarize_compaction_message(msg)
        assert result == "user: hello world"

    def test_empty_content(self) -> None:
        msg = {"role": "assistant", "content": ""}
        result = summarize_compaction_message(msg)
        assert "[empty]" in result

    def test_tool_calls_with_content(self) -> None:
        msg = {
            "role": "assistant",
            "content": "Let me search",
            "tool_calls": [{"function": {"name": "s"}}],
        }
        result = summarize_compaction_message(msg)
        assert "Let me search" in result
