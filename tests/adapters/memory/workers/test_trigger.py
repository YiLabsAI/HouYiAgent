from __future__ import annotations

from houyi.adapters.memory.workers.trigger import (
    ActivityMonitor,
    HybridTriggerConfig,
    HybridTriggerPolicy,
    ManualTriggerPolicy,
)


class _Clock:
    """Manually advanced monotonic clock for deterministic trigger tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_monitor_idle_failures() -> None:
    clock = _Clock()
    monitor = ActivityMonitor(clock=clock)

    clock.advance(50)
    assert monitor.idle_seconds() == 50

    monitor.record_activity()
    assert monitor.idle_seconds() == 0

    monitor.record_failure()
    monitor.record_failure(3)
    assert monitor.failure_count == 4

    monitor.reset_failures()
    assert monitor.failure_count == 0


def test_manual_policy_never_fires() -> None:
    policy = ManualTriggerPolicy()
    assert policy.should_run() is False
    policy.record_run()  # no-op, must not raise
    assert policy.should_run() is False


def test_hybrid_respects_min_interval() -> None:
    clock = _Clock()
    monitor = ActivityMonitor(clock=clock)
    policy = HybridTriggerPolicy(
        monitor,
        config=HybridTriggerConfig(min_interval_s=100, idle_threshold_s=10),
        clock=clock,
    )

    # Idle threshold met but min_interval has not elapsed since construction.
    clock.advance(20)
    assert policy.should_run() is False


def test_hybrid_idle_fires() -> None:
    clock = _Clock()
    monitor = ActivityMonitor(clock=clock)
    policy = HybridTriggerPolicy(
        monitor,
        config=HybridTriggerConfig(min_interval_s=100, idle_threshold_s=10),
        clock=clock,
    )

    monitor.record_activity()
    clock.advance(150)  # past interval, idle for 150s >= 10s
    assert policy.should_run() is True


def test_hybrid_busy_holds() -> None:
    clock = _Clock()
    monitor = ActivityMonitor(clock=clock)
    policy = HybridTriggerPolicy(
        monitor,
        config=HybridTriggerConfig(min_interval_s=100, idle_threshold_s=60),
        clock=clock,
    )

    clock.advance(150)  # past interval
    monitor.record_activity()  # but just had activity -> not idle
    assert policy.should_run() is False


def test_hybrid_failure_pressure_overrides() -> None:
    clock = _Clock()
    monitor = ActivityMonitor(clock=clock)
    policy = HybridTriggerPolicy(
        monitor,
        config=HybridTriggerConfig(
            min_interval_s=10_000, idle_threshold_s=10_000, failure_pressure=3
        ),
        clock=clock,
    )

    monitor.record_activity()  # busy and well within interval
    assert policy.should_run() is False

    monitor.record_failure(3)
    assert policy.should_run() is True


def test_hybrid_record_resets() -> None:
    clock = _Clock()
    monitor = ActivityMonitor(clock=clock)
    policy = HybridTriggerPolicy(
        monitor,
        config=HybridTriggerConfig(min_interval_s=100, idle_threshold_s=10, failure_pressure=3),
        clock=clock,
    )

    monitor.record_failure(5)
    assert policy.should_run() is True

    policy.record_run()
    assert monitor.failure_count == 0
    # Right after a run the interval floor blocks the next fire.
    assert policy.should_run() is False
