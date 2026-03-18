from houyi.application.context.stream_metrics import (
    backfill_reasoning_usage,
    build_generation_metadata,
    normalize_usage_payload,
)


class TestNormalizeUsagePayload:
    def test_usage_maps_prompt(self):
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

    def test_usage_cache_hit(self):
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

    def test_usage_keeps_timing(self):
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
    def test_metadata_reads_usage(self):
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
        assert metadata["first_token_ms"] == 111
        assert metadata["decode_tokens_per_second"] == 22.2
        assert metadata["end_to_end_tokens_per_second"] == 11.1
        assert metadata["tokens_per_second"] == 11.1


class TestBackfillReasoningUsage:
    def test_reasoning_backfill(self):
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
