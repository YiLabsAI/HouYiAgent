from types import SimpleNamespace

from houyi.application.context.compaction_policy import (
    build_prune_only_summary,
    detect_incomplete_turn_gate,
    evaluate_safety_gate,
    partition_messages_for_compaction,
)


class _Role:
    def __init__(self, value: str):
        self.value = value


def _message(message_id: str, role: str, **kwargs):
    return SimpleNamespace(message_id=message_id, role=_Role(role), **kwargs)


class TestDetectIncompleteTurnGate:
    def test_gate_active_tool_loop(self):
        messages = [
            _message(
                "a1",
                "assistant",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search"}}],
            ),
            _message("u1", "user"),
        ]

        assert detect_incomplete_turn_gate(messages) == "active_tool_loop"

    def test_gate_split_turn(self):
        messages = [
            _message(
                "a1",
                "assistant",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search"}}],
            ),
            _message("t1", "tool", tool_call_id="call-2"),
        ]

        assert detect_incomplete_turn_gate(messages) == "split_incomplete_turn"

    def test_gate_complete_turn(self):
        messages = [
            _message(
                "a1",
                "assistant",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search"}}],
            ),
            _message("t1", "tool", tool_call_id="call-1"),
            _message("a2", "assistant", tool_calls=None),
        ]

        assert detect_incomplete_turn_gate(messages) is None


class TestEvaluateSafetyGate:
    def test_gate_insufficient_history(self):
        messages = [_message("u1", "user"), _message("u2", "user")]

        assert (
            evaluate_safety_gate(
                messages=messages,
                active_streaming_message_id=None,
                last_compacted_message_count=None,
                last_compacted_at=None,
                now=100.0,
                recent_window=2,
                cooldown_messages=0,
                cooldown_seconds=0.0,
            )
            == "insufficient_history"
        )

    def test_gate_cooldown_active(self):
        messages = [_message(f"u{i}", "user") for i in range(5)]

        assert (
            evaluate_safety_gate(
                messages=messages,
                active_streaming_message_id=None,
                last_compacted_message_count=4,
                last_compacted_at=95.0,
                now=100.0,
                recent_window=2,
                cooldown_messages=2,
                cooldown_seconds=10.0,
            )
            == "cooldown_active"
        )


class TestPartitionMessagesForCompaction:
    def test_partition_keeps_protected(self):
        messages = [_message(f"m{i}", "user") for i in range(1, 8)]

        kept, dropped = partition_messages_for_compaction(
            messages,
            protected_message_ids={"m2"},
            recent_window=3,
        )

        assert [message.message_id for message in kept] == ["m2", "m5", "m6", "m7"]
        assert [message.message_id for message in dropped] == ["m1", "m3", "m4"]


class TestBuildPruneOnlySummary:
    def test_summary_prune_range(self):
        messages = [_message("m1", "user"), _message("m4", "assistant")]

        summary = build_prune_only_summary(messages)

        assert summary == "Pruned 2 earlier messages to recover context budget. Range: m1..m4."
