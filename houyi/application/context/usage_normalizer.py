from __future__ import annotations

from typing import Any

from houyi.application.context.types import NormalizedUsage


class UsageNormalizer:
    def normalize_payload(
        self,
        *,
        usage: dict[str, Any] | None = None,
        timings: dict[str, float | int | None] | None = None,
        usage_source: str | None = None,
        usage_confidence: str | None = None,
        metadata: dict[str, Any] | None = None,
        include_input_tokens: bool = False,
    ) -> dict[str, Any] | None:
        if not isinstance(usage, dict) or not usage:
            return None
        normalized = self.normalize(
            usage=dict(usage),
            timings=timings,
            usage_source=usage_source,
            usage_confidence=usage_confidence,
            metadata=metadata,
        ).model_dump(mode="json")
        if include_input_tokens:
            normalized["input_tokens"] = int(
                usage.get("input_tokens", normalized.get("prompt_tokens", 0))
                or normalized.get("prompt_tokens", 0)
            )
        return normalized

    def normalize(
        self,
        *,
        usage: dict[str, Any] | None = None,
        timings: dict[str, float | int | None] | None = None,
        usage_source: str | None = None,
        usage_confidence: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NormalizedUsage:
        raw = usage or {}
        prompt_tokens = self._to_int(
            raw.get("prompt_tokens"),
            raw.get("input_tokens"),
            raw.get("prompt_token_count"),
        )
        completion_tokens = self._to_int(
            raw.get("completion_tokens"),
            raw.get("output_tokens"),
            raw.get("candidates_token_count"),
        )
        reasoning_tokens = self._to_int(
            raw.get("reasoning_tokens"),
            raw.get("thinking_tokens"),
        )
        cached_prompt_tokens = self._to_int(
            raw.get("cached_prompt_tokens"),
            raw.get("cache_read_input_tokens"),
        )
        total_tokens = self._to_int(
            raw.get("total_tokens"),
            prompt_tokens + completion_tokens,
        )
        answer_tokens = max(0, completion_tokens - reasoning_tokens)
        normalized_timings = self._normalize_timings(
            raw=raw,
            completion_tokens=completion_tokens,
            timings=timings or {},
        )
        resolved_source = usage_source or ("reported" if usage else "fallback")
        resolved_confidence = usage_confidence or ("reported" if usage else "fallback")
        return NormalizedUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            total_tokens=total_tokens,
            usage_confidence=resolved_confidence,
            usage_source=resolved_source,
            first_token_ms=normalized_timings["first_token_ms"],
            decode_tokens_per_second=normalized_timings["decode_tokens_per_second"],
            end_to_end_tokens_per_second=normalized_timings["end_to_end_tokens_per_second"],
            metadata=metadata or {},
        )

    def fallback(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        timings: dict[str, float | int | None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NormalizedUsage:
        return self.normalize(
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            timings=timings,
            usage_source="fallback",
            usage_confidence="fallback",
            metadata=metadata,
        )

    @staticmethod
    def _to_int(*values: Any) -> int:
        for value in values:
            if value is None:
                continue
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
        return 0

    def _normalize_timings(
        self,
        *,
        raw: dict[str, Any],
        completion_tokens: int,
        timings: dict[str, float | int | None],
    ) -> dict[str, float | None]:
        first_token_ms = self._to_float(
            timings.get("first_token_ms"),
            raw.get("first_token_ms"),
        )
        decode_duration_ms = self._to_float(timings.get("decode_duration_ms"))
        end_to_end_ms = self._to_float(timings.get("end_to_end_ms"))
        decode_tokens_per_second = self._to_float(raw.get("decode_tokens_per_second"))
        end_to_end_tokens_per_second = self._to_float(raw.get("end_to_end_tokens_per_second"))
        if decode_duration_ms and decode_duration_ms > 0:
            decode_tokens_per_second = completion_tokens / (decode_duration_ms / 1000.0)
        if end_to_end_ms and end_to_end_ms > 0:
            end_to_end_tokens_per_second = completion_tokens / (end_to_end_ms / 1000.0)
        return {
            "first_token_ms": first_token_ms,
            "decode_tokens_per_second": decode_tokens_per_second,
            "end_to_end_tokens_per_second": end_to_end_tokens_per_second,
        }

    @staticmethod
    def _to_float(*values: Any) -> float | None:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None
