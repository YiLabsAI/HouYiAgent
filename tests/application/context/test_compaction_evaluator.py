from __future__ import annotations

from houyi.application.context.compaction_evaluator import CompactionEvaluator
from houyi.application.context.token_estimator import TokenEstimator


def test_evaluate_returns_compaction_record_with_ratio_and_refs():
    evaluator = CompactionEvaluator(
        TokenEstimator(context_window_override=10000, output_reserve=2000)
    )
    before_messages = [
        {
            "message_id": "m1",
            "role": "user",
            "content": "Investigate RepoAlpha failure in ServiceBeta for Tenant42",
        },
        {
            "message_id": "m2",
            "role": "assistant",
            "content": "RepoAlpha fails after deploy because ServiceBeta misses ENV_TOKEN",
        },
    ]

    record = evaluator.evaluate(
        before_messages=before_messages,
        summary="RepoAlpha failure is caused by missing ENV_TOKEN in ServiceBeta for Tenant42.",
        source_message_ids=["m1", "m2"],
        retained_refs=["https://example.com/runbook"],
        metadata={"trigger": "threshold_70"},
        trigger="threshold_70",
    )

    assert record.trigger == "threshold_70"
    assert record.source_message_ids == ["m1", "m2"]
    assert record.retained_refs == ["https://example.com/runbook"]
    assert record.metrics.tokens_before > 0
    assert record.metrics.tokens_after > 0
    assert 0 < record.metrics.compression_ratio <= 1
    assert record.metrics.retained_refs_count == 1
    assert record.metrics.retained_entity_coverage > 0
    assert record.metadata["trigger"] == "threshold_70"


def test_evaluate_marks_missing_pinned_content_as_pin_violation():
    evaluator = CompactionEvaluator(
        TokenEstimator(context_window_override=10000, output_reserve=2000)
    )
    before_messages = [
        {
            "message_id": "p1",
            "role": "user",
            "content": "PinnedInvariant must never be omitted",
        },
        {
            "message_id": "m2",
            "role": "assistant",
            "content": "Other detail about RepoAlpha",
        },
    ]

    record = evaluator.evaluate(
        before_messages=before_messages,
        summary="RepoAlpha details were condensed.",
        pinned_message_ids=["p1"],
        source_message_ids=["p1", "m2"],
    )

    assert record.pinned_message_ids == ["p1"]
    assert record.metrics.pin_violation_count == 1


def test_evaluate_uses_full_entity_coverage_when_no_entities_exist():
    evaluator = CompactionEvaluator(
        TokenEstimator(context_window_override=10000, output_reserve=2000)
    )
    before_messages = [
        {
            "message_id": "m1",
            "role": "user",
            "content": "all lowercase content without entities",
        }
    ]

    record = evaluator.evaluate(
        before_messages=before_messages,
        summary="short lowercase summary",
        source_message_ids=["m1"],
    )

    assert record.metrics.retained_entity_coverage == 1.0
