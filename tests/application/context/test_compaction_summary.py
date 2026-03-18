from types import SimpleNamespace

from houyi.application.context.compaction_summary import build_compaction_summary


class _Role:
    def __init__(self, value: str):
        self.value = value


def _message(role: str, **kwargs):
    return SimpleNamespace(role=_Role(role), **kwargs)


class TestBuildCompactionSummary:
    def test_summary_tool_loop(self):
        summary = build_compaction_summary(
            [
                _message(message_id="u1", role="user", content="搜索文件 skill.md"),
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
