from __future__ import annotations

from typing import Any

from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.usage_normalizer import UsageNormalizer

_USAGE_NORMALIZER = UsageNormalizer()


def build_generation_metadata(
    *,
    usage_payload: dict[str, Any] | None,
    first_token_ms: float | None,
    generation_time_ms: float | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if first_token_ms is not None:
        metadata["first_token_latency_ms"] = round(first_token_ms, 2)
    if generation_time_ms is not None:
        metadata["generation_time_ms"] = round(generation_time_ms, 2)
    if first_token_ms is not None and generation_time_ms is not None:
        metadata["decode_time_ms"] = round(max(0.0, generation_time_ms - first_token_ms), 2)
    if isinstance(usage_payload, dict):
        if usage_payload.get("first_token_ms") is not None:
            metadata["first_token_ms"] = usage_payload.get("first_token_ms")
        if usage_payload.get("end_to_end_tokens_per_second") is not None:
            end_to_end_tps = float(usage_payload["end_to_end_tokens_per_second"])
            metadata["end_to_end_tokens_per_second"] = round(end_to_end_tps, 2)
            metadata["tokens_per_second"] = round(end_to_end_tps, 2)
        if usage_payload.get("decode_tokens_per_second") is not None:
            metadata["decode_tokens_per_second"] = round(
                float(usage_payload["decode_tokens_per_second"]),
                2,
            )
    return metadata


def normalize_usage_payload(
    usage_payload: dict[str, Any] | None,
    *,
    first_token_ms: float | None = None,
    generation_time_ms: float | None = None,
) -> dict[str, Any] | None:
    timings: dict[str, float | int | None] | None = None
    if first_token_ms is not None or generation_time_ms is not None:
        decode_duration_ms: float | None = None
        if generation_time_ms is not None and first_token_ms is not None:
            decode_duration_ms = max(0.0, generation_time_ms - first_token_ms)
        timings = {
            "first_token_ms": first_token_ms,
            "decode_duration_ms": decode_duration_ms,
            "end_to_end_ms": generation_time_ms,
        }
    return _USAGE_NORMALIZER.normalize_payload(
        usage=usage_payload,
        timings=timings,
        include_input_tokens=True,
    )


def backfill_reasoning_usage(
    usage_payload: dict[str, Any] | None,
    *,
    reasoning_text: str,
    model: str,
) -> dict[str, Any] | None:
    if not isinstance(usage_payload, dict):
        return usage_payload
    if usage_payload.get("reasoning_tokens"):
        return usage_payload
    if not reasoning_text.strip():
        return usage_payload
    reasoning_tokens = TokenEstimator(model=model).count_text(reasoning_text)
    if reasoning_tokens <= 0:
        return usage_payload
    normalized = dict(usage_payload)
    normalized["reasoning_tokens"] = reasoning_tokens
    completion_tokens = int(normalized.get("completion_tokens", 0) or 0)
    normalized["answer_tokens"] = max(0, completion_tokens - reasoning_tokens)
    return normalized
