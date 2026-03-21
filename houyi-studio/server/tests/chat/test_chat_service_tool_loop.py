from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from houyi_studio.server.chat.chat_service import (
    ChatService,
    _coerce_text_content,
    _looks_like_repo_intent,
    _looks_like_tool_intent,
    _looks_like_web_intent,
    _sanitize_final_stream_messages,
    _sanitize_tool_loop_messages,
    _sanitize_tool_loop_structure,
)
from houyi_studio.server.chat.types import SendMessageRequest


class TestCoerceTextContent:
    def test_plain_string(self):
        assert _coerce_text_content("hello") == "hello"

    def test_multimodal_text(self):
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "search file skill.md"},
        ]
        assert _coerce_text_content(content) == "search file skill.md"

    def test_mapping_string(self):
        content = {"query": "skill.md", "source": "tool"}
        coerced = _coerce_text_content(content)
        assert isinstance(coerced, str)
        assert "skill.md" in coerced


class TestSanitizeFinalStreamMessages:
    def test_drops_empty_assistant_tool(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "houyi_grep", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "plan first",
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "houyi_grep",
                "content": '{"matches": 3}',
            },
            {"role": "assistant", "content": "final answer"},
        ]

        sanitized, stats = _sanitize_final_stream_messages(messages)

        assert sanitized == [
            {"role": "user", "content": '[tool:houyi_grep] {"matches": 3}'},
            {"role": "assistant", "content": "final answer"},
        ]
        assert stats["assistant_tool_call_carrier_count"] == 1
        assert stats["assistant_reasoning_removed_count"] == 1
        assert stats["assistant_reasoning_only_removed_count"] == 1
        assert stats["tool_result_projection_count"] == 1


class TestSanitizeToolLoopMessages:
    def test_content_strings(self):
        messages = [
            {"role": "system", "content": {"policy": "strict"}},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]

        sanitized = _sanitize_tool_loop_messages(messages)

        assert all(isinstance(msg["content"], str) for msg in sanitized)
        assert sanitized[1]["content"] == "hello"
        assert sanitized[2]["content"] == "done"

    def test_tool_args_json(self):
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
            },
            {
                "role": "tool",
                "content": '{"matches":["README.md"]}',
                "tool_call_id": "call-1",
                "name": "houyi_grep",
            },
        ]

        sanitized = _sanitize_tool_loop_messages(messages)
        tool_call = sanitized[0]["tool_calls"][0]
        arguments = tool_call["function"]["arguments"]

        assert isinstance(arguments, str)
        assert "skill.md" in arguments

    def test_truncate_single_message(self):
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

    def test_drop_old_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 120_000},
            {"role": "tool", "content": "b" * 120_000},
            {"role": "assistant", "content": "latest"},
        ]

        sanitized = _sanitize_tool_loop_messages(messages)

        roles = [m["role"] for m in sanitized]
        assert roles[0] == "system"
        assert roles[-1] == "assistant"
        total_chars = sum(len(str(m.get("content") or "")) for m in sanitized)
        assert total_chars <= 220_000

    def test_latest_tool_turn(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "demo_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "demo_b", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": '{"ok":true}',
                "tool_call_id": "call_1",
                "name": "demo_a",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "demo_c", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "x" * 220_000,
                "tool_call_id": "call_3",
                "name": "demo_c",
            },
        ]

        sanitized = _sanitize_tool_loop_structure(messages)

        assert len(sanitized) == 2
        assert sanitized[0]["role"] == "assistant"
        assert sanitized[0]["tool_calls"][0]["id"] == "call_3"
        assert sanitized[1]["role"] == "tool"
        assert sanitized[1]["tool_call_id"] == "call_3"

    def test_drops_incomplete_toolturn(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "demo_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "demo_b", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": '{"ok":true}',
                "tool_call_id": "call_1",
                "name": "demo_a",
            },
            {"role": "assistant", "content": "final answer"},
        ]

        sanitized = _sanitize_tool_loop_structure(messages)

        assert len(sanitized) == 1
        assert sanitized[0]["role"] == "assistant"
        assert sanitized[0]["content"] == "final answer"


class TestToolIntentHeuristics:
    def test_explicit_keywords(self):
        assert _looks_like_tool_intent("grep chat_service.py") is True

    def test_file_paths(self):
        assert _looks_like_tool_intent("look into ./houyi-studio/server/chat_service.py") is True

    def test_chitchat(self):
        assert _looks_like_tool_intent("where did messi go") is False

    def test_generic_lookup_phrase(self):
        assert _looks_like_tool_intent("look up recent RocketMQ news") is False


class TestRepoIntent:
    def test_github_url(self):
        assert _looks_like_repo_intent("https://github.com/snap-research/locomo") is True

    def test_local_file(self):
        assert _looks_like_repo_intent("grep skill.md in ./houyi-studio") is False


class TestWebIntent:
    def test_web_lookup_phrase(self):
        assert _looks_like_web_intent("look up recent RocketMQ news") is True

    def test_online_search(self):
        query = 'Query the local "skill.md" document and conduct an online search for Agent topic on InfoQ'
        assert _looks_like_web_intent(query) is True

    def test_repo_query_counts_as_web(self):
        assert _looks_like_web_intent("https://github.com/snap-research/locomo") is True

    def test_local_file_not_web(self):
        assert _looks_like_web_intent("grep skill.md in ./houyi-studio") is False

    def test_web_typo_phrase(self):
        typo_query = (
            "\u641c\u7d22\u672c\u5730 skill.md \u6587\u4ef6"
            ", \u5e76\u4e0a\u7f51\u641c\u7d20"
            " info article from 2025"
        )
        assert _looks_like_web_intent(typo_query) is True


class TestChatServiceToolLoopGating:
    def test_non_tool_query(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="where did kaka go"),
            resolved_skills=["houyi_grep", "houyi_read_file"],
        )

        assert decision.mode == "disabled_by_gating"
        assert decision.reason == "heuristic_no_tool_intent"
        assert decision.enabled_skills == []

    def test_logs_gate_summary(self, caplog: pytest.LogCaptureFixture):
        service = ChatService(json_store=MagicMock())

        with caplog.at_level("INFO"):
            decision = service._gate_tool_loop(
                request=SendMessageRequest(content="look up recent RocketMQ news", model="glm-4.5"),
                resolved_skills=["houyi_web_search"],
            )

        assert decision.mode == "enabled"
        assert any(
            "Chat tool-loop gate:" in record.message
            and "model=glm-4.5" in record.message
            and "reason=heuristic_web_intent" in record.message
            and "enabled_skills=['houyi_web_search']" in record.message
            for record in caplog.records
        )

    def test_explicit_skills(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="hi", enable_skills=["houyi_web_search"]),
            resolved_skills=["houyi_grep", "houyi_web_search"],
        )

        assert decision.mode == "enabled"
        assert decision.reason == "explicit_skill_request"
        assert decision.enabled_skills == ["houyi_web_search"]

    def test_web_search_filters(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="lookup skill.md file", enable_web_search=True),
            resolved_skills=["houyi_grep", "houyi_shell_exec", "houyi_web_search"],
        )

        assert decision.mode == "enabled"
        assert decision.reason == "explicit_web_search_mixed_intent"
        assert decision.enabled_skills == ["houyi_grep", "houyi_web_search"]

    def test_request_disable(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="find skill.md", enable_tool_calls=False),
            resolved_skills=["houyi_grep"],
        )

        assert decision.mode == "disabled_by_request"
        assert decision.reason == "request_disable"
        assert decision.enabled_skills == []

    def test_aggressive_strategy(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="hello", tool_call_strategy="aggressive"),
            resolved_skills=["houyi_grep", "houyi_read_file"],
        )

        assert decision.mode == "enabled"
        assert decision.reason == "strategy_aggressive_default_on"
        assert decision.enabled_skills == ["houyi_grep", "houyi_read_file"]

    def test_conservative_strategy(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="find skill.md", tool_call_strategy="conservative"),
            resolved_skills=["houyi_grep", "houyi_read_file"],
        )

        assert decision.mode == "disabled_by_gating"
        assert decision.reason == "strategy_conservative_requires_explicit"
        assert decision.enabled_skills == []

    def test_repo_query_search_only(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="https://github.com/snap-research/locomo"),
            resolved_skills=["houyi_grep", "houyi_shell_exec", "houyi_web_search"],
        )

        assert decision.mode == "enabled"
        assert decision.reason == "heuristic_mixed_intent"
        assert decision.enabled_skills == ["houyi_grep", "houyi_web_search"]

    def test_repo_query_adds_search(self):
        service = ChatService(json_store=MagicMock())
        resolved = service._resolve_enabled_chat_skills(
            SendMessageRequest(content="readme of github.com/foo/bar")
        )

        assert "houyi_read_file" in resolved
        assert "houyi_find_files" in resolved
        assert "houyi_list_dir" in resolved
        assert "houyi_grep" in resolved
        assert "houyi_shell_exec" not in resolved

    def test_web_query_adds_search(self):
        service = ChatService(json_store=MagicMock())
        resolved = service._resolve_enabled_chat_skills(
            SendMessageRequest(content="look up recent RocketMQ news")
        )

        assert "houyi_web_search" in resolved
        assert "houyi_shell_exec" not in resolved

    def test_mixed_resolve(self):
        service = ChatService(json_store=MagicMock())
        resolved = service._resolve_enabled_chat_skills(
            SendMessageRequest(
                content='Query the local "skill.md" document and conduct an online search for Feng Jia\'s 2025 publications on InfoQ'
            )
        )

        assert "houyi_find_files" in resolved
        assert "houyi_grep" in resolved
        assert "houyi_web_search" in resolved

    def test_not_add_shell_exec(self):
        service = ChatService(json_store=MagicMock())
        resolved = service._resolve_enabled_chat_skills(
            SendMessageRequest(content="look up recent RocketMQ news")
        )

        assert "houyi_shell_exec" not in resolved

    def test_explicit_tool_intent(self):
        service = ChatService(json_store=MagicMock())
        resolved = service._resolve_enabled_chat_skills(
            SendMessageRequest(content="run command `curl https://example.com`")
        )

        assert "houyi_shell_exec" in resolved

    def test_web_query_preference(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="look up recent RocketMQ news"),
            resolved_skills=["houyi_grep", "houyi_shell_exec", "houyi_web_search"],
        )

        assert decision.mode == "enabled"
        assert decision.reason == "heuristic_web_intent"
        assert decision.enabled_skills == ["houyi_web_search"]

    def test_heuristic_mixed_intent(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(
                content='Query the local "skill.md" document and conduct an online search for Agent topic on InfoQ'
            ),
            resolved_skills=[
                "houyi_find_files",
                "houyi_grep",
                "houyi_shell_exec",
                "houyi_web_search",
            ],
        )

        assert decision.mode == "enabled"
        assert decision.reason == "heuristic_mixed_intent"
        assert decision.enabled_skills == ["houyi_find_files", "houyi_grep", "houyi_web_search"]

    def test_disables_heuristic_toolloop(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="find references in the repo"),
            resolved_skills=["houyi_grep", "houyi_read_file"],
            context_usage={"used_tokens": 95000, "max_context_tokens": 128000},
            runtime_profile=SimpleNamespace(compression_threshold=0.7),
        )

        assert decision.mode == "disabled_by_pressure"
        assert decision.reason == "context_pressure_near_compaction"
        assert decision.enabled_skills == []

    def test_logs_pressure_override(self, caplog: pytest.LogCaptureFixture):
        service = ChatService(json_store=MagicMock())

        with caplog.at_level("INFO"):
            decision = service._gate_tool_loop(
                request=SendMessageRequest(
                    content="find references in the repo", model="deepseek-r1"
                ),
                resolved_skills=["houyi_grep", "houyi_read_file"],
                context_usage={"used_tokens": 95000, "max_context_tokens": 128000},
                runtime_profile=SimpleNamespace(compression_threshold=0.7),
            )

        assert decision.mode == "disabled_by_pressure"
        assert any(
            "Chat tool-loop gate:" in record.message
            and "mode=disabled_by_pressure" in record.message
            and "reason=context_pressure_near_compaction" in record.message
            for record in caplog.records
        )

    def test_keeps_explicit_skills(self):
        service = ChatService(json_store=MagicMock())
        decision = service._gate_tool_loop(
            request=SendMessageRequest(content="hi", enable_skills=["houyi_web_search"]),
            resolved_skills=["houyi_grep", "houyi_web_search"],
            context_usage={"used_tokens": 95000, "max_context_tokens": 128000},
            runtime_profile=SimpleNamespace(compression_threshold=0.7),
        )

        assert decision.mode == "enabled"
        assert decision.reason == "explicit_skill_request"
        assert decision.enabled_skills == ["houyi_web_search"]


class TestChatServiceRuntimeProfile:
    def test_default_profile(self):
        service = ChatService(json_store=MagicMock())

        profile = service._resolve_runtime_profile(SendMessageRequest(content="hello"))

        assert profile.name == "chat.default"
        assert profile.keep_n is None
        assert profile.compression_threshold == 0.7

    def test_toggle_profile(self):
        service = ChatService(json_store=MagicMock())

        profile = service._resolve_runtime_profile(
            SendMessageRequest(content="hello", enable_deep_research=True)
        )

        assert profile.name == "agent.deep_research"
        assert profile.keep_n == 5
        assert profile.compression_threshold == 0.6

    def test_skill_profile(self):
        service = ChatService(json_store=MagicMock())

        profile = service._resolve_runtime_profile(
            SendMessageRequest(content="hello", enable_skills=["deep_research"])
        )

        assert profile.name == "agent.deep_research"
