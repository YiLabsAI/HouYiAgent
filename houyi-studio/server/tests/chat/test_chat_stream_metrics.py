from houyi_studio.server.chat.chat_service import _extract_finish_reason, _finalize_stream_result

from houyi.application.context.stream_metrics import (
    backfill_reasoning_usage,
    build_generation_metadata,
    normalize_usage_payload,
)


class TestNormalizeUsagePayload:
    def test_prompt_mapping(self):
        usage = normalize_usage_payload(
            {
                "input_tokens": 12,
                "completion_tokens": 8,
                "reasoning_tokens": 3,
            }
        )

        assert usage is not None
        assert usage["prompt_tokens"] == 12
        assert usage["input_tokens"] == 12
        assert usage["answer_tokens"] == 5
        assert usage["total_tokens"] == 20
        assert usage["cache_hit"] is False
        assert usage["usage_confidence"] == "reported"

    def test_cache_hit(self):
        usage = normalize_usage_payload(
            {
                "input_tokens": 12,
                "completion_tokens": 8,
                "prompt_cache_hit_tokens": 6,
            }
        )

        assert usage is not None
        assert usage["cached_prompt_tokens"] == 6
        assert usage["cache_hit"] is True

    def test_nested_reasoning_tokens(self):
        usage = normalize_usage_payload(
            {
                "input_tokens": 12,
                "completion_tokens": 8,
                "completion_tokens_details": {
                    "reasoning_tokens": 3,
                },
            }
        )

        assert usage is not None
        assert usage["reasoning_tokens"] == 3
        assert usage["reasoning_tokens_reported"] is True
        assert usage["answer_tokens"] == 5
        assert usage["answer_tokens_reported"] is True

    def test_timing(self):
        usage = normalize_usage_payload(
            {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "first_token_ms": 120,
                "decode_tokens_per_second": 40,
                "end_to_end_tokens_per_second": 25,
            }
        )

        assert usage is not None
        assert usage["first_token_ms"] == 120.0
        assert usage["decode_tokens_per_second"] == 40.0
        assert usage["end_to_end_tokens_per_second"] == 25.0


class TestBuildGenerationMetadata:
    def test_usage_timing(self):
        metadata = build_generation_metadata(
            usage_payload={
                "first_token_ms": 111,
                "decode_tokens_per_second": 22.2,
                "end_to_end_tokens_per_second": 11.1,
            },
            first_token_ms=115,
            generation_time_ms=800,
        )

        assert metadata["first_token_latency_ms"] == 115
        assert metadata["generation_time_ms"] == 800
        assert metadata["decode_time_ms"] == 685
        assert metadata["first_token_ms"] == 111
        assert metadata["decode_tokens_per_second"] == 22.2
        assert metadata["end_to_end_tokens_per_second"] == 11.1
        assert metadata["tokens_per_second"] == 11.1


class TestBackfillReasoningUsage:
    def test_reasoning_backfill(
        self,
    ):
        usage = backfill_reasoning_usage(
            {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "reasoning_tokens": 0,
                "answer_tokens": 8,
            },
            reasoning_text="thinking step 1 + step 2",
            model="gpt-4o-mini",
        )

        assert usage is not None
        assert int(usage["reasoning_tokens"]) > 0
        assert int(usage["answer_tokens"]) < 8


class TestFinalizeStreamResult:
    def test_records_usage(self):
        class _Span:
            def __init__(self):
                self.attributes = {}
                self.tokens = None
                self.status = None
                self.ended = False

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def set_tokens(self, *, input_tokens, output_tokens):
                self.tokens = (input_tokens, output_tokens)

            def set_status(self, status, message=None):
                self.status = (status, message)

            def end(self):
                self.ended = True

        class _Adapter:
            last_usage = {"input_tokens": 12, "completion_tokens": 8, "reasoning_tokens": 3}
            last_finish_reason = "stop"

        span = _Span()
        usage, finish_reason, metadata = _finalize_stream_result(
            llm_adapter=_Adapter(),
            llm_span=span,
            first_token_ms=120,
            generation_time_ms=800,
            chunk_count=4,
        )

        assert usage is not None
        assert usage["prompt_tokens"] == 12
        assert finish_reason == "stop"
        assert metadata["generation_time_ms"] == 800
        assert span.attributes["chat.stream_chunk_count"] == 4
        assert span.tokens == (12, 8)
        assert span.ended is True

    def test_finish_reason_fallback(self):
        class _Span:
            def __init__(self):
                self.attributes = {}
                self.tokens = None
                self.status = None
                self.ended = False

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def set_tokens(self, *, input_tokens, output_tokens):
                self.tokens = (input_tokens, output_tokens)

            def set_status(self, status, message=None):
                self.status = (status, message)

            def end(self):
                self.ended = True

        class _Adapter:
            last_usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
            last_finish_reason = None

        span = _Span()
        usage, finish_reason, metadata = _finalize_stream_result(
            llm_adapter=_Adapter(),
            llm_span=span,
            first_token_ms=80,
            generation_time_ms=500,
            chunk_count=2,
            finish_reason_sources=({"metadata": {"finish_reason": "tool_calls"}},),
        )

        assert usage is not None
        assert finish_reason == "tool_calls"
        assert metadata["generation_time_ms"] == 500
        assert span.ended is True


class TestExtractFinishReason:
    def test_direct_reason(self):
        assert _extract_finish_reason(None, "", "length", {"finish_reason": "stop"}) == "length"
        assert _extract_finish_reason({"metadata": {"finish_reason": "tool_calls"}}) == "tool_calls"
