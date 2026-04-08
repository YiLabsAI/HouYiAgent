from __future__ import annotations

from houyi.application.research.runtime.search_telemetry import TelemetryEmitter
from houyi.application.research.types import SufficiencyDecision, SufficiencyFeatures


def _decision() -> SufficiencyDecision:
    return SufficiencyDecision(sufficient=True, rationale="ok")


def _features() -> SufficiencyFeatures:
    return SufficiencyFeatures(
        source_count=2,
        relevant_source_count=2,
        domain_count=2,
        provider_count=1,
        authority_source_count=1,
        recent_source_count=0,
        relevance_score=1.0,
        diversity_score=0.8,
        authority_score=0.5,
        recency_score=0.0,
        has_primary_source=True,
        missing_dimensions=[],
    )


class TestTelemetryEmitter:
    async def test_budget_consumed_event(self):
        captured: list[tuple[str, dict]] = []

        async def _notify(event_type: str, data: dict) -> None:
            captured.append((event_type, data))

        emitter = TelemetryEmitter(notify=_notify)
        await emitter.budget_consumed(
            question_id="q1",
            round_index=1,
            layer="round",
            reason_code="exhausted",
            budget_ms=1000,
            remaining_ms=0,
        )
        assert captured[0][0] == "search.budget_consumed"
        assert captured[0][1]["layer"] == "round"

    async def test_round_timing_event(self):
        captured: list[tuple[str, dict]] = []

        async def _notify(event_type: str, data: dict) -> None:
            captured.append((event_type, data))

        emitter = TelemetryEmitter(notify=_notify)
        await emitter.round_timing(
            question_id="q1",
            round_number=1,
            elapsed_ms=100.0,
            query_count=2,
            skipped_queries=0,
            cancelled_queries=0,
            hit_count=3,
            source_count=3,
            decision=_decision(),
            stop_layer="",
        )
        assert captured[0][0] == "search.round_timing"
        assert captured[0][1]["query_count"] == 2

    async def test_sufficiency_features_event(self):
        captured: list[tuple[str, dict]] = []

        async def _notify(event_type: str, data: dict) -> None:
            captured.append((event_type, data))

        emitter = TelemetryEmitter(notify=_notify)
        await emitter.sufficiency_features(question_id="q1", round_index=1, features=_features())
        assert captured[0][0] == "search.sufficiency_features"

    async def test_query_cancelled_emits(self):
        captured: list[tuple[str, dict]] = []

        async def _notify(event_type: str, data: dict) -> None:
            captured.append((event_type, data))

        emitter = TelemetryEmitter(notify=_notify)
        await emitter.query_cancelled("q1", 1, ["qa", "qb"], reason="timeout")
        assert len(captured) == 2
        assert all(e == "search.query_cancelled" for e, _ in captured)
