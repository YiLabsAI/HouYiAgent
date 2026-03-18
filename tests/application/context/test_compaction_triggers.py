from houyi.application.context.compaction_triggers import (
    estimate_post_compaction_utilization,
    resolve_compaction_trigger,
    resolve_utilization,
    resolve_watermarks,
)


class TestResolveWatermarks:
    def test_watermarks_clamp_low(self):
        low, high, critical = resolve_watermarks(
            default_low_watermark=0.6,
            default_pressure_threshold=0.7,
            default_overflow_threshold=0.9,
            low_watermark=0.8,
            pressure_threshold=0.7,
            overflow_threshold=0.9,
        )

        assert (low, high, critical) == (0.7, 0.7, 0.9)


class TestResolveUtilization:
    def test_utilization_prefers_state(self):
        ratio, source = resolve_utilization(
            used_units=750,
            max_units=1000,
            current_tokens=200,
            max_input_tokens=1000,
        )

        assert ratio == 0.75
        assert source == "conversation_context_state"

    def test_utilization_falls_back(self):
        ratio, source = resolve_utilization(
            used_units=None,
            max_units=None,
            current_tokens=250,
            max_input_tokens=1000,
        )

        assert ratio == 0.25
        assert source == "token_estimate"


class TestResolveCompactionTrigger:
    def test_trigger_manual(self):
        trigger = resolve_compaction_trigger(
            trigger_kind="manual",
            message_count=8,
            utilization_ratio=0.75,
            utilization_source="conversation_context_state",
            default_low_watermark=0.6,
            default_pressure_threshold=0.7,
            default_overflow_threshold=0.9,
        )

        assert trigger is not None
        assert trigger["trigger"] == "manual"
        assert trigger["pressure_level"] == "elevated"

    def test_trigger_post_turn(self):
        trigger = resolve_compaction_trigger(
            trigger_kind="post_turn_background",
            message_count=8,
            utilization_ratio=0.95,
            utilization_source="conversation_context_state",
            default_low_watermark=0.6,
            default_pressure_threshold=0.7,
            default_overflow_threshold=0.9,
        )

        assert trigger is not None
        assert trigger["trigger"] == "post_turn_background"
        assert trigger["pressure_level"] == "critical"

    def test_trigger_overflow(self):
        trigger = resolve_compaction_trigger(
            trigger_kind=None,
            message_count=10,
            utilization_ratio=0.03,
            utilization_source="token_estimate",
            default_low_watermark=0.6,
            default_pressure_threshold=0.01,
            default_overflow_threshold=0.02,
        )

        assert trigger is not None
        assert trigger["trigger"] == "overflow_recovery"
        assert trigger["pressure_level"] == "critical"

    def test_trigger_pressure(self):
        trigger = resolve_compaction_trigger(
            trigger_kind=None,
            message_count=10,
            utilization_ratio=0.75,
            utilization_source="conversation_context_state",
            default_low_watermark=0.6,
            default_pressure_threshold=0.7,
            default_overflow_threshold=0.9,
        )

        assert trigger is not None
        assert trigger["trigger"] == "pre_request_pressure"
        assert trigger["pressure_level"] == "elevated"


class TestEstimatePostCompactionUtilization:
    def test_post_utilization_prefers_state(self):
        ratio = estimate_post_compaction_utilization(
            used_units=800,
            max_units=1000,
            current_tokens=300,
            max_input_tokens=1000,
            released_units=200,
        )

        assert ratio == 0.6

    def test_post_utilization_falls_back(self):
        ratio = estimate_post_compaction_utilization(
            used_units=None,
            max_units=None,
            current_tokens=300,
            max_input_tokens=1000,
            released_units=50,
        )

        assert ratio == 0.25
