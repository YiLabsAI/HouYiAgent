import time

from houyi.application.context.context_lifecycle import (
    CompressionHookContext,
    ContextLifecycleHookService,
    ListHookResult,
    MappingHookResult,
    StringHookResult,
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


class TestContextLifecycleHookService:
    def test_contract(self):
        service = ContextLifecycleHookService()

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

    def test_before_compress(self):
        span = _Span()
        service = ContextLifecycleHookService(
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

    def test_after_compress(self):
        span = _Span()
        service = ContextLifecycleHookService(
            after_compress=lambda record: {"restore_status": "hooked"}
        )

        result = service.run_after_compress(_record(), span=span)

        assert result.restore_status == "hooked"
        assert span.attributes["chat.hooks.after_compress.called"] is True

    def test_render_and_memory(self):
        span = _Span()
        service = ContextLifecycleHookService(
            on_render=lambda messages: [*messages, {"role": "system", "content": "guard"}],
            on_memory_recall=lambda text: text + " filtered",
        )

        rendered = service.run_render([{"role": "user", "content": "hi"}], span=span)
        memory = service.run_memory_recall("session memory", span=span)

        assert rendered[-1] == {"role": "system", "content": "guard"}
        assert memory == "session memory filtered"
        assert span.attributes["chat.hooks.on_render.called"] is True
        assert span.attributes["chat.hooks.on_memory_recall.called"] is True

    def test_hook_order(self):
        span = _Span()

        class _FirstHook:
            name = "first"
            priority = 20

            def before_compress(self, ctx: CompressionHookContext):
                return MappingHookResult(update={"phase": "first"})

        class _SecondHook:
            name = "second"
            priority = 10

            def before_compress(self, ctx: CompressionHookContext):
                assert ctx.payload is not None
                return MappingHookResult(
                    update={"phase": "second", "seen": ctx.payload.get("trigger")}
                )

        service = ContextLifecycleHookService(hooks=[_FirstHook(), _SecondHook()])

        result = service.run_before_compress({"trigger": "pre_request_pressure"}, span=span)

        assert result == {
            "trigger": "pre_request_pressure",
            "phase": "first",
            "seen": "pre_request_pressure",
        }
        assert span.attributes["chat.hooks.before_compress.last_handler"] == "first"

    def test_hook_timeout(self):
        span = _Span()

        class _SlowHook:
            name = "slow"

            def on_render(self, ctx):
                time.sleep(0.01)
                return ListHookResult(
                    items=[*ctx.messages, {"role": "system", "content": "too-late"}]
                )

        service = ContextLifecycleHookService(
            hooks=[_SlowHook()],
            phase_timeouts={"render": 0.001},
        )

        rendered = service.run_render([{"role": "user", "content": "hi"}], span=span)

        assert rendered == [{"role": "user", "content": "hi"}]
        assert span.attributes["chat.hooks.render.timeout"] is True

    def test_hook_string_result(self):
        span = _Span()

        class _MemoryHook:
            name = "memory"

            def on_memory_recall(self, ctx):
                return StringHookResult(text=ctx.memory_text + " refined")

        service = ContextLifecycleHookService(hooks=[_MemoryHook()])

        memory = service.run_memory_recall("session memory", span=span)

        assert memory == "session memory refined"
