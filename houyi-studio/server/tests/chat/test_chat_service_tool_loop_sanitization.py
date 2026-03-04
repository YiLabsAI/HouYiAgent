from unittest.mock import MagicMock

from houyi_studio.server.chat.chat_service import (
    ChatService,
    _coerce_text_content,
    _looks_like_tool_intent,
    _sanitize_tool_loop_messages,
)
from houyi_studio.server.chat.types import SendMessageRequest


class TestCoerceTextContent:
    def test_keeps_plain_string(self):
        assert _coerce_text_content("hello") == "hello"

    def test_flattens_multimodal_text_parts(self):
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "search file skill.md"},
        ]
        assert _coerce_text_content(content) == "search file skill.md"

    def test_serializes_non_string_payload(self):
        content = {"query": "skill.md", "source": "tool"}
        coerced = _coerce_text_content(content)
        assert isinstance(coerced, str)
        assert "skill.md" in coerced


class TestSanitizeToolLoopMessages:
    def test_forces_content_to_string(self):
        messages = [
            {"role": "system", "content": {"policy": "strict"}},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]

        sanitized = _sanitize_tool_loop_messages(messages)

        assert all(isinstance(msg["content"], str) for msg in sanitized)
        assert sanitized[1]["content"] == "hello"
        assert sanitized[2]["content"] == "done"

    def test_normalizes_tool_call_arguments_to_json_string(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "houyi_grep",
                            "arguments": {"pattern": "skill.md", "path": "/tmp"},
                        },
                    }
                ],
            }
        ]

        sanitized = _sanitize_tool_loop_messages(messages)
        tool_call = sanitized[0]["tool_calls"][0]
        arguments = tool_call["function"]["arguments"]

        assert isinstance(arguments, str)
        assert "skill.md" in arguments

    def test_truncates_single_message_content(self):
        messages = [
            {
                "role": "tool",
                "content": "x" * 20_000,
            }
        ]

        sanitized = _sanitize_tool_loop_messages(messages)

        content = sanitized[0]["content"]
        assert len(content) <= 12_000 + len("\n...[truncated]...\n")
        assert "[truncated]" in content

    def test_caps_total_content_budget_by_dropping_oldest_non_system_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 120_000},
            {"role": "tool", "content": "b" * 120_000},
            {"role": "assistant", "content": "latest"},
        ]

        sanitized = _sanitize_tool_loop_messages(messages)

        # system should be preserved while old oversized non-system entries are pruned.
        roles = [m["role"] for m in sanitized]
        assert roles[0] == "system"
        assert roles[-1] == "assistant"
        total_chars = sum(len(str(m.get("content") or "")) for m in sanitized)
        assert total_chars <= 220_000


class TestToolIntentHeuristics:
    def test_detects_explicit_tool_keywords(self):
        assert _looks_like_tool_intent("请帮我 grep 一下 chat_service.py") is True

    def test_detects_file_path_like_queries(self):
        assert _looks_like_tool_intent("look into ./houyi-studio/server/chat_service.py") is True

    def test_skips_general_chitchat(self):
        assert _looks_like_tool_intent("xx 去哪里了") is False


class TestToolLoopGating:
    def test_disables_tool_loop_for_non_tool_query_by_heuristic(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="xx 去哪里了"),
            resolved_skills=["houyi_grep", "houyi_read_file"],
        )

        assert decision.mode == "disabled_by_gating"
        assert decision.reason == "heuristic_no_tool_intent"
        assert decision.enabled_skills == []

    def test_keeps_enabled_when_explicit_skills_requested(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="hi", enable_skills=["web_search"]),
            resolved_skills=["houyi_grep", "web_search"],
        )

        assert decision.mode == "enabled"
        assert decision.reason == "explicit_skill_request"
        assert decision.enabled_skills == ["houyi_grep", "web_search"]

    def test_honors_request_level_disable(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="find skill.md", enable_tool_calls=False),
            resolved_skills=["houyi_grep"],
        )

        assert decision.mode == "disabled_by_request"
        assert decision.reason == "request_disable"
        assert decision.enabled_skills == []

    def test_enables_tool_loop_by_default_with_aggressive_strategy(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="hello", tool_call_strategy="aggressive"),
            resolved_skills=["houyi_grep", "houyi_read_file"],
        )

        assert decision.mode == "enabled"
        assert decision.reason == "strategy_aggressive_default_on"
        assert decision.enabled_skills == ["houyi_grep", "houyi_read_file"]

    def test_requires_explicit_tool_request_with_conservative_strategy(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="find skill.md", tool_call_strategy="conservative"),
            resolved_skills=["houyi_grep", "houyi_read_file"],
        )

        assert decision.mode == "disabled_by_gating"
        assert decision.reason == "strategy_conservative_requires_explicit"
        assert decision.enabled_skills == []
