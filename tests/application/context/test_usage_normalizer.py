from __future__ import annotations

from houyi.application.context.usage_normalizer import UsageNormalizer


class TestUsageNormalizer:
    def test_reported(self):
        normalizer = UsageNormalizer()
        usage = normalizer.normalize(
            usage={
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "reasoning_tokens": 30,
                "cached_prompt_tokens": 10,
                "total_tokens": 200,
            },
            timings={
                "first_token_ms": 150,
                "decode_duration_ms": 400,
                "end_to_end_ms": 1000,
            },
        )
        assert usage.prompt_tokens == 120
        assert usage.completion_tokens == 80
        assert usage.reasoning_tokens == 30
        assert usage.answer_tokens == 50
        assert usage.cached_prompt_tokens == 10
        assert usage.total_tokens == 200
        assert usage.usage_source == "reported"
        assert usage.usage_confidence == "reported"
        assert usage.first_token_ms == 150.0
        assert usage.decode_tokens_per_second == 200.0
        assert usage.end_to_end_tokens_per_second == 80.0

    def test_fallback(self):
        normalizer = UsageNormalizer()
        usage = normalizer.fallback(prompt_tokens=20, completion_tokens=10, reasoning_tokens=4)
        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 10
        assert usage.reasoning_tokens == 4
        assert usage.answer_tokens == 6
        assert usage.total_tokens == 30
        assert usage.usage_source == "fallback"
        assert usage.usage_confidence == "fallback"

    def test_payload(self):
        normalizer = UsageNormalizer()
        payload = normalizer.normalize_payload(
            usage={
                "input_tokens": 12,
                "completion_tokens": 8,
                "reasoning_tokens": 3,
            },
            include_input_tokens=True,
        )
        assert payload is not None
        assert payload["prompt_tokens"] == 12
        assert payload["input_tokens"] == 12
        assert payload["answer_tokens"] == 5
        assert payload["total_tokens"] == 20
