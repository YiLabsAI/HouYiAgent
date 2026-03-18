from types import SimpleNamespace

from houyi_studio.server.chat.chat_context_adapter import ChatContextAdapter
from houyi_studio.server.chat.types import Message, MessageRole

from houyi.application.context.context_lifecycle import (
    ContextLifecycleHookService as ChatContextHookService,
)
from houyi.application.context.types import CompactionMetrics, CompactionRecord


class _Span:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def _record() -> CompactionRecord:
    return CompactionRecord(
        trigger="pre_request_pressure",
        summary="summary",
        metrics=CompactionMetrics(tokens_before=100, tokens_after=20),
    )


class TestContextHooks:
    def test_contract(self):
        service = ChatContextHookService()

        contract = service.describe_contract()

        assert contract["before_compress"] == {
            "boundary": "compaction",
            "mode": "mapping",
            "fallback": "keep_original",
        }
        assert contract["after_compress"] == {
            "boundary": "compaction",
            "mode": "record_update",
            "fallback": "keep_original",
        }
        assert contract["compress_error"] == {
            "boundary": "compaction",
            "mode": "effect",
            "fallback": "no_op",
        }
        assert contract["tool_result"]["boundary"] == "tool_loop"
        assert contract["plan_assembled"]["boundary"] == "context_runtime"
        assert contract["render"]["mode"] == "list_replace"
        assert contract["memory_recall"]["mode"] == "string_replace"

    def test_before_payload(self):
        span = _Span()
        service = ChatContextHookService(
            before_compress=lambda payload: {"pressure_level": "critical", "extra": "x"}
        )

        result = service.run_before_compress(
            {"trigger": "pre_request_pressure", "pressure_level": "elevated"},
            span=span,
        )

        assert result["trigger"] == "pre_request_pressure"
        assert result["pressure_level"] == "critical"
        assert result["extra"] == "x"
        assert span.attributes["chat.hooks.before_compress.called"] is True

    def test_before_error(self):
        span = _Span()
        service = ChatContextHookService(
            before_compress=lambda payload: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        payload = {"trigger": "pre_request_pressure", "pressure_level": "elevated"}

        result = service.run_before_compress(payload, span=span)

        assert result == payload
        assert span.attributes["chat.hooks.before_compress.error"] == "boom"

    def test_after_update(self):
        span = _Span()
        service = ChatContextHookService(after_compress=lambda record: {"restore_status": "hooked"})

        result = service.run_after_compress(_record(), span=span)

        assert result.restore_status == "hooked"
        assert span.attributes["chat.hooks.after_compress.called"] is True

    def test_after_error(self):
        span = _Span()
        service = ChatContextHookService(
            after_compress=lambda record: (_ for _ in ()).throw(RuntimeError("after-error"))
        )
        record = _record()

        result = service.run_after_compress(record, span=span)

        assert result.restore_status is None
        assert span.attributes["chat.hooks.after_compress.error"] == "after-error"

    def test_error_stage(self):
        captured = {}
        span = _Span()
        service = ChatContextHookService(on_compress_error=lambda payload: captured.update(payload))

        service.run_compress_error(stage="commit", error=RuntimeError("bad"), span=span)

        assert captured == {"stage": "commit", "error": "bad"}
        assert span.attributes["chat.hooks.on_compress_error.called"] is True
        assert span.attributes["chat.compaction.error_stage"] == "commit"

    def test_tool_result(self):
        span = _Span()
        replacement = [SimpleNamespace(content="hooked")]
        service = ChatContextHookService(on_tool_result=lambda messages, trace: replacement)

        result = service.run_tool_result(
            [SimpleNamespace(content="orig")], [{"tool_call_id": "1"}], span=span
        )

        assert result == replacement
        assert span.attributes["chat.hooks.on_tool_result.called"] is True

        error_span = _Span()
        error_service = ChatContextHookService(
            on_tool_result=lambda messages, trace: (_ for _ in ()).throw(RuntimeError("tool-error"))
        )
        original = [SimpleNamespace(content="orig")]

        fallback = error_service.run_tool_result(original, None, span=error_span)

        assert fallback == original
        assert error_span.attributes["chat.hooks.on_tool_result.error"] == "tool-error"

    def test_plan(self):
        span = _Span()
        service = ChatContextHookService(on_plan_assembled=lambda plan: {"plan": "hooked"})

        result = service.run_plan_assembled({"plan": "orig"}, span=span)

        assert result == {"plan": "hooked"}
        assert span.attributes["chat.hooks.on_plan_assembled.called"] is True

        error_span = _Span()
        error_service = ChatContextHookService(
            on_plan_assembled=lambda plan: (_ for _ in ()).throw(RuntimeError("plan-error"))
        )
        original = {"plan": "orig"}

        fallback = error_service.run_plan_assembled(original, span=error_span)

        assert fallback == original
        assert error_span.attributes["chat.hooks.on_plan_assembled.error"] == "plan-error"

    def test_render_memory(self):
        span = _Span()
        service = ChatContextHookService(
            on_render=lambda messages: [*messages, {"role": "system", "content": "guard"}],
            on_memory_recall=lambda text: text + " filtered",
        )

        rendered = service.run_render([{"role": "user", "content": "hi"}], span=span)
        memory = service.run_memory_recall("session memory", span=span)

        assert rendered[-1] == {"role": "system", "content": "guard"}
        assert memory == "session memory filtered"
        assert span.attributes["chat.hooks.on_render.called"] is True
        assert span.attributes["chat.hooks.on_memory_recall.called"] is True

        error_span = _Span()
        error_service = ChatContextHookService(
            on_render=lambda messages: (_ for _ in ()).throw(RuntimeError("render-error")),
            on_memory_recall=lambda text: (_ for _ in ()).throw(RuntimeError("memory-error")),
        )

        render_fallback = error_service.run_render(
            [{"role": "user", "content": "hi"}], span=error_span
        )
        memory_fallback = error_service.run_memory_recall("session memory", span=error_span)

        assert render_fallback == [{"role": "user", "content": "hi"}]
        assert memory_fallback == "session memory"
        assert error_span.attributes["chat.hooks.on_render.error"] == "render-error"
        assert error_span.attributes["chat.hooks.on_memory_recall.error"] == "memory-error"


class TestContextAdapter:
    def test_hooks(self):
        span = _Span()
        hook_service = ChatContextHookService(
            on_memory_recall=lambda text: text + " filtered",
            on_plan_assembled=lambda plan: plan,
            on_render=lambda messages: [*messages, {"role": "system", "content": "guard"}],
        )
        runtime = ChatContextAdapter(
            memory_store=SimpleNamespace(as_context_text=lambda scope: "session memory"),
            is_vision_model=lambda _model: False,
            sanitize_tool_loop_structure=lambda messages: messages,
            hook_service=hook_service,
        )
        conversation = SimpleNamespace(
            messages=[
                Message(role=MessageRole.USER, content="hello"),
                Message(role=MessageRole.ASSISTANT, content="hi"),
            ]
        )

        llm_messages, context_usage = runtime.build_context_messages(
            conversation=conversation,
            model="gpt-4o-mini",
            sys_instructions="You are helpful",
            span=span,
            input_budget=512,
        )

        assert llm_messages[-1] == {"role": "system", "content": "guard"}
        assert context_usage["used_tokens"] > 0
        assert span.attributes["chat.hooks.on_memory_recall.called"] is True
        assert span.attributes["chat.hooks.on_plan_assembled.called"] is True
        assert span.attributes["chat.hooks.on_render.called"] is True


def test_ctx_adapter_sets_attrs():
    span = _Span()
    runtime = ChatContextAdapter(
        memory_store=SimpleNamespace(as_context_text=lambda scope: "session memory"),
        is_vision_model=lambda _model: False,
        sanitize_tool_loop_structure=lambda messages: messages,
        hook_service=ChatContextHookService(),
    )
    conversation = SimpleNamespace(
        messages=[
            Message(role=MessageRole.USER, content="hello"),
            Message(role=MessageRole.ASSISTANT, content="hi"),
        ]
    )

    llm_messages, context_usage = runtime.build_context_messages(
        conversation=conversation,
        model="gpt-4o-mini",
        sys_instructions="You are helpful",
        span=span,
        input_budget=512,
    )

    assert context_usage["used_tokens"] > 0
    assert span.attributes["chat.context_tokens_used"] == context_usage["used_tokens"]
    assert span.attributes["chat.context_tokens_max"] == context_usage["max_context_tokens"]
    assert span.attributes["chat.llm_messages_count"] == len(llm_messages)
    assert span.attributes["chat.context.rendered_message_count"] == len(llm_messages)
    assert "chat.context_blocks" in span.attributes
