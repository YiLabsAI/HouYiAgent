import time
from types import SimpleNamespace

from houyi.application.context.context_lifecycle import ContextLifecycleHookService
from houyi.application.context.context_request_builder import (
    ContextRequestBuilder,
    ContextRequestBuildInput,
    ContextRequestSourceInput,
    SummarySemanticJudgeDecision,
)


class _Span:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class TestContextRequestBuilder:
    def test_builder_runs_hooks(self):
        span = _Span()
        builder = ContextRequestBuilder(
            hook_service=ContextLifecycleHookService(
                on_plan_assembled=lambda plan: plan,
                on_render=lambda messages: [*messages, {"role": "system", "content": "guard"}],
            )
        )

        result = builder.build(
            ContextRequestBuildInput(
                model="gpt-4o-mini",
                system_instructions="You are helpful",
                history_messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                conversation_messages=[],
                conversation_metadata=None,
                memory_text="session memory",
                span=span,
                input_budget=512,
            )
        )

        assert result.llm_messages[-1] == {"role": "system", "content": "guard"}
        assert result.context_usage["used_tokens"] > 0
        assert span.attributes["chat.context.history_message_count"] == 2

    def test_builder_usage(self):
        builder = ContextRequestBuilder()

        usage = builder.get_usage(
            ContextRequestBuildInput(
                model="gpt-4o-mini",
                system_instructions="You are helpful",
                history_messages=[{"role": "user", "content": "hello"}],
                conversation_messages=[],
                conversation_metadata=None,
                memory_text=None,
                span=SimpleNamespace(set_attribute=lambda *_args, **_kwargs: None),
            )
        )

        assert usage is not None
        assert usage["used_tokens"] > 0

    def test_builder_source_input(self):
        span = _Span()
        hook_service = ContextLifecycleHookService(on_memory_recall=lambda text: text + " filtered")
        source = SimpleNamespace(
            messages=["msg-1", "msg-2"],
            metadata={"topic": "demo"},
        )
        builder = ContextRequestBuilder(
            build_history_messages=lambda conversation, model: [
                {"role": "user", "content": f"{model}:{conversation.messages[0]}"}
            ],
            get_memory_text=lambda active_span: hook_service.run_memory_recall(
                "session memory", span=active_span
            ),
            sanitize_history_messages=lambda messages: [
                *messages,
                {"role": "system", "content": "sanitized"},
            ],
        )

        request_input = builder._build_input(
            ContextRequestSourceInput(
                source=source,
                model="gpt-4o-mini",
                system_instructions="You are helpful",
                span=span,
                input_budget=256,
                truncation_log_label="chat_send",
            )
        )

        assert request_input.model == "gpt-4o-mini"
        assert request_input.system_instructions == "You are helpful"
        assert request_input.memory_text == "session memory filtered"
        assert request_input.conversation_messages == ["msg-1", "msg-2"]
        assert request_input.conversation_metadata == {"topic": "demo"}
        assert request_input.input_budget == 256
        assert request_input.history_messages[-1] == {"role": "system", "content": "sanitized"}
        assert span.attributes["chat.hooks.on_memory_recall.called"] is True

    def test_builder_includes_summary(self):
        span = _Span()
        builder = ContextRequestBuilder()

        result = builder.build(
            ContextRequestBuildInput(
                model="gpt-4o-mini",
                system_instructions="You are helpful",
                history_messages=[
                    {
                        "message_id": "u1",
                        "role": "user",
                        "content": "Fix context summary gating bug",
                    },
                    {"message_id": "a1", "role": "assistant", "content": "Investigating"},
                ],
                conversation_messages=[],
                conversation_metadata={
                    "compaction_history": [
                        {
                            "compaction_id": "cmp_1",
                            "trigger": "pre_request_pressure",
                            "summary": "Context summary gating bug in repo search flow",
                            "created_at": time.time(),
                            "source_message_ids": ["old-1", "old-2"],
                            "metadata": {"summarization_mode": "llm"},
                        }
                    ]
                },
                memory_text=None,
                span=span,
                input_budget=512,
            )
        )

        assert any(
            "[Conversation Summary]" in str(message.get("content", ""))
            for message in result.llm_messages
        )

    def test_builder_excludes_summary(self):
        span = _Span()
        builder = ContextRequestBuilder()

        result = builder.build(
            ContextRequestBuildInput(
                model="gpt-4o-mini",
                system_instructions="You are helpful",
                history_messages=[
                    {
                        "message_id": "u1",
                        "role": "user",
                        "content": "Fix context summary gating bug",
                    },
                    {"message_id": "a1", "role": "assistant", "content": "Investigating"},
                ],
                conversation_messages=[],
                conversation_metadata={
                    "compaction_history": [
                        {
                            "compaction_id": "cmp_1",
                            "trigger": "post_turn_background",
                            "summary": "Context summary gating bug in repo search flow",
                            "created_at": time.time() - (60 * 20),
                            "source_message_ids": ["u1"],
                            "metadata": {"summarization_mode": "fallback_prune_only"},
                        }
                    ]
                },
                memory_text=None,
                span=span,
                input_budget=512,
            )
        )

        assert all(
            "[Conversation Summary]" not in str(message.get("content", ""))
            for message in result.llm_messages
        )

    def test_usage_tracks_prompt(self):
        span = _Span()
        builder = ContextRequestBuilder()

        result = builder.build(
            ContextRequestBuildInput(
                model="gpt-4o-mini",
                system_instructions="You are helpful",
                history_messages=[
                    {
                        "message_id": "u1",
                        "role": "user",
                        "content": "Fix `ContextPlanner` in context_planner.py",
                    },
                    {"message_id": "a1", "role": "assistant", "content": "Investigating"},
                ],
                conversation_messages=[],
                conversation_metadata={
                    "compaction_history": [
                        {
                            "compaction_id": "cmp_3",
                            "trigger": "pre_request_pressure",
                            "summary": "Fix `ContextPlanner` in context_planner.py by adjusting planner usage",
                            "created_at": time.time(),
                            "source_message_ids": ["old-1", "old-2"],
                            "metadata": {"summarization_mode": "llm"},
                        }
                    ]
                },
                memory_text=None,
                span=span,
                input_budget=512,
            )
        )

        assert (
            result.context_usage["assembled_prompt_tokens"] >= result.context_usage["used_tokens"]
        )
        assert result.context_usage["assembled_message_count"] == len(result.llm_messages)
        assert result.context_usage["summary_eligible"] is True
        assert result.context_usage["summary_reason"] == "eligible"
        assert (
            span.attributes["chat.context.assembled_prompt_tokens"]
            == result.context_usage["assembled_prompt_tokens"]
        )
        assert span.attributes["chat.context.summary_reason"] == "eligible"

    def test_summary_uses_judge(self):
        span = _Span()
        seen: dict[str, object] = {}

        def judge(payload):
            seen["base_score"] = payload.base_score
            seen["summary_text"] = payload.summary_text
            return SummarySemanticJudgeDecision(
                include=True,
                reason="semantic_ok",
                metadata={"boundary_id": "judge-boundary", "semantic_judge": "accepted"},
            )

        builder = ContextRequestBuilder(summary_semantic_judge=judge)

        result = builder.build(
            ContextRequestBuildInput(
                model="gpt-4o-mini",
                system_instructions="You are helpful",
                history_messages=[
                    {"message_id": "u1", "role": "user", "content": "Read the code and inspect"},
                    {"message_id": "a1", "role": "assistant", "content": "Investigating"},
                ],
                conversation_messages=[],
                conversation_metadata={
                    "compaction_history": [
                        {
                            "compaction_id": "cmp_4",
                            "trigger": "pre_request_pressure",
                            "summary": "Investigate `ContextPlanner` logic in context_planner.py",
                            "created_at": time.time(),
                            "source_message_ids": ["old-1", "old-2"],
                            "metadata": {"summarization_mode": "llm"},
                        }
                    ]
                },
                memory_text=None,
                span=span,
                input_budget=512,
            )
        )

        assert seen["base_score"] == 0
        assert any(
            "[Conversation Summary]" in str(message.get("content", ""))
            for message in result.llm_messages
        )
        assert result.context_usage["summary_reason"] == "eligible"

    def test_summary_task_signal(self):
        span = _Span()
        builder = ContextRequestBuilder()

        result = builder.build(
            ContextRequestBuildInput(
                model="gpt-4o-mini",
                system_instructions="You are helpful",
                history_messages=[
                    {
                        "message_id": "u1",
                        "role": "user",
                        "content": "Read the code and inspect the issue",
                    },
                    {"message_id": "a1", "role": "assistant", "content": "Investigating"},
                ],
                conversation_messages=[],
                conversation_metadata={
                    "compaction_history": [
                        {
                            "compaction_id": "cmp_2",
                            "trigger": "pre_request_pressure",
                            "summary": "Gemini tokenizer mismatch in vertex adapter billing usage",
                            "created_at": time.time(),
                            "source_message_ids": ["old-1", "old-2"],
                            "metadata": {"summarization_mode": "llm"},
                        }
                    ]
                },
                memory_text=None,
                span=span,
                input_budget=512,
            )
        )

        assert all(
            "[Conversation Summary]" not in str(message.get("content", ""))
            for message in result.llm_messages
        )
