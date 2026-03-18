from __future__ import annotations


def resolve_watermarks(
    *,
    default_low_watermark: float,
    default_pressure_threshold: float,
    default_overflow_threshold: float,
    low_watermark: float | None = None,
    pressure_threshold: float | None = None,
    overflow_threshold: float | None = None,
) -> tuple[float, float, float]:
    resolved_high = (
        max(0.0, float(pressure_threshold))
        if pressure_threshold is not None
        else float(default_pressure_threshold)
    )
    resolved_critical = max(
        resolved_high,
        float(overflow_threshold)
        if overflow_threshold is not None
        else float(default_overflow_threshold),
    )
    resolved_low = (
        max(0.0, float(low_watermark))
        if low_watermark is not None
        else float(default_low_watermark)
    )
    return min(resolved_low, resolved_high), resolved_high, resolved_critical


def resolve_utilization(
    *,
    used_units: int | None,
    max_units: int | None,
    current_tokens: int,
    max_input_tokens: int,
) -> tuple[float, str]:
    if isinstance(used_units, int) and isinstance(max_units, int) and max_units > 0:
        return max(0.0, float(used_units) / float(max_units)), "conversation_context_state"
    return float(current_tokens) / float(max(1, int(max_input_tokens))), "token_estimate"


def estimate_post_compaction_utilization(
    *,
    used_units: int | None,
    max_units: int | None,
    current_tokens: int,
    max_input_tokens: int,
    released_units: int,
) -> float:
    if isinstance(used_units, int) and isinstance(max_units, int) and max_units > 0:
        return round(max(0.0, float(max(0, used_units - released_units)) / float(max_units)), 4)
    return round(
        max(
            0.0,
            float(max(0, current_tokens - released_units)) / float(max(1, int(max_input_tokens))),
        ),
        4,
    )


def resolve_compaction_trigger(
    *,
    trigger_kind: str | None,
    message_count: int,
    utilization_ratio: float,
    utilization_source: str,
    default_low_watermark: float,
    default_pressure_threshold: float,
    default_overflow_threshold: float,
    low_watermark: float | None = None,
    pressure_threshold: float | None = None,
    overflow_threshold: float | None = None,
) -> dict[str, str | float] | None:
    if message_count < 3:
        return None
    resolved_low, resolved_high, resolved_critical = resolve_watermarks(
        default_low_watermark=default_low_watermark,
        default_pressure_threshold=default_pressure_threshold,
        default_overflow_threshold=default_overflow_threshold,
        low_watermark=low_watermark,
        pressure_threshold=pressure_threshold,
        overflow_threshold=overflow_threshold,
    )
    rounded_utilization = round(utilization_ratio, 4)
    if trigger_kind == "manual":
        pressure_level = "critical" if utilization_ratio >= resolved_critical else "elevated"
        return build_trigger_payload(
            trigger="manual",
            pressure_level=pressure_level,
            reason="manual_request",
            utilization_ratio=rounded_utilization,
            utilization_source=utilization_source,
            low_watermark=resolved_low,
            high_watermark=resolved_high,
            critical_watermark=resolved_critical,
        )
    if trigger_kind == "post_turn_background":
        return _resolve_threshold_trigger(
            trigger="post_turn_background",
            reason="post_turn_pressure",
            utilization_ratio=utilization_ratio,
            rounded_utilization=rounded_utilization,
            utilization_source=utilization_source,
            high_watermark=resolved_high,
            critical_watermark=resolved_critical,
            low_watermark=resolved_low,
        )
    return _resolve_threshold_trigger(
        trigger="pre_request_pressure",
        reason="token_window_pressure",
        overflow_trigger="overflow_recovery",
        overflow_reason="token_window_overflow",
        utilization_ratio=utilization_ratio,
        rounded_utilization=rounded_utilization,
        utilization_source=utilization_source,
        high_watermark=resolved_high,
        critical_watermark=resolved_critical,
        low_watermark=resolved_low,
    )


def build_trigger_payload(
    *,
    trigger: str,
    pressure_level: str,
    reason: str,
    utilization_ratio: float,
    utilization_source: str,
    low_watermark: float,
    high_watermark: float,
    critical_watermark: float,
) -> dict[str, str | float]:
    return {
        "trigger": trigger,
        "pressure_level": pressure_level,
        "reason": reason,
        "utilization_ratio": utilization_ratio,
        "utilization_source": utilization_source,
        "low_watermark": round(low_watermark, 4),
        "high_watermark": round(high_watermark, 4),
        "critical_watermark": round(critical_watermark, 4),
    }


def _resolve_threshold_trigger(
    *,
    trigger: str,
    reason: str,
    utilization_ratio: float,
    rounded_utilization: float,
    utilization_source: str,
    high_watermark: float,
    critical_watermark: float,
    low_watermark: float,
    overflow_trigger: str | None = None,
    overflow_reason: str | None = None,
) -> dict[str, str | float] | None:
    if utilization_ratio >= critical_watermark:
        return build_trigger_payload(
            trigger=overflow_trigger or trigger,
            pressure_level="critical",
            reason=overflow_reason or reason,
            utilization_ratio=rounded_utilization,
            utilization_source=utilization_source,
            low_watermark=low_watermark,
            high_watermark=high_watermark,
            critical_watermark=critical_watermark,
        )
    if utilization_ratio >= high_watermark:
        return build_trigger_payload(
            trigger=trigger,
            pressure_level="elevated",
            reason=reason,
            utilization_ratio=rounded_utilization,
            utilization_source=utilization_source,
            low_watermark=low_watermark,
            high_watermark=high_watermark,
            critical_watermark=critical_watermark,
        )
    return None
