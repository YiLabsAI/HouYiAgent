"""Unit tests for A/B Experiment framework."""

from __future__ import annotations

from houyi.arena.ab_experiment import (
    ExperimentArm,
    ExperimentReport,
    _combined_score,
    _determine_winner,
)


class TestCombinedScore:
    def test_both_present(self):
        score = _combined_score(80.0, 90.0)
        assert score == 80.0 * 0.6 + 90.0 * 0.4

    def test_race_only(self):
        score = _combined_score(80.0, None)
        assert score == 80.0 * 0.6

    def test_fact_only(self):
        score = _combined_score(None, 90.0)
        assert score == 90.0 * 0.4

    def test_both_none(self):
        assert _combined_score(None, None) == 0.0


class TestDetermineWinner:
    def test_arm_a_wins(self):
        a = ExperimentArm(arm_id="a", mode="delegate", quality_race=85, quality_fact=90)
        b = ExperimentArm(arm_id="b", mode="autonomous", quality_race=60, quality_fact=70)
        winner, margin, rec = _determine_winner(a, b)
        assert winner == "arm_a"
        assert margin > 0

    def test_arm_b_wins(self):
        a = ExperimentArm(arm_id="a", mode="delegate", quality_race=50, quality_fact=60)
        b = ExperimentArm(arm_id="b", mode="autonomous", quality_race=90, quality_fact=95)
        winner, margin, rec = _determine_winner(a, b)
        assert winner == "arm_b"

    def test_tie(self):
        a = ExperimentArm(
            arm_id="a",
            mode="delegate",
            quality_race=80,
            quality_fact=80,
            duration_seconds=10.0,
        )
        b = ExperimentArm(
            arm_id="b",
            mode="autonomous",
            quality_race=80.5,
            quality_fact=80.5,
            duration_seconds=5.0,
        )
        winner, margin, rec = _determine_winner(a, b)
        assert winner is None
        assert "Tie" in rec

    def test_arm_a_error(self):
        a = ExperimentArm(arm_id="a", mode="delegate", error="timeout")
        b = ExperimentArm(arm_id="b", mode="autonomous", quality_race=70, quality_fact=80)
        winner, _, _ = _determine_winner(a, b)
        assert winner == "arm_b"

    def test_arm_b_error(self):
        a = ExperimentArm(arm_id="a", mode="delegate", quality_race=70, quality_fact=80)
        b = ExperimentArm(arm_id="b", mode="autonomous", error="crash")
        winner, _, _ = _determine_winner(a, b)
        assert winner == "arm_a"

    def test_both_error(self):
        a = ExperimentArm(arm_id="a", mode="delegate", error="fail")
        b = ExperimentArm(arm_id="b", mode="autonomous", error="fail")
        winner, _, rec = _determine_winner(a, b)
        assert winner is None
        assert "Both" in rec


class TestModels:
    def test_arm_defaults(self):
        arm = ExperimentArm(arm_id="test", mode="delegate")
        assert arm.quality_race is None
        assert arm.error is None

    def test_report_defaults(self):
        report = ExperimentReport(query="test")
        assert report.winner is None
        assert report.margin == 0.0


class TestABExperimentRun:
    async def test_init(self):
        from unittest.mock import AsyncMock

        from houyi.arena.ab_experiment import ABExperiment

        llm = AsyncMock()
        ws = AsyncMock()
        exp = ABExperiment(llm, ws, temperature=0.5)
        assert exp._llm is llm
        assert exp._web_search is ws
        assert exp._llm_kwargs == {"temperature": 0.5}

    async def test_run_both_arms_succeed(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from houyi.arena import ab_experiment as _mod
        from houyi.arena.ab_experiment import ABExperiment

        llm = AsyncMock()
        ws = AsyncMock()
        exp = ABExperiment(llm, ws)

        def _make_runtime(**kwargs):
            s = AsyncMock()
            s.run_id = "test_rid"
            qs = MagicMock()
            qs.race.overall = 80.0
            qs.fact.citation_accuracy = 90.0
            s.quality_score = qs
            rpt = MagicMock()
            rpt.metadata.source_count = 5
            s.get_report = AsyncMock(return_value=rpt)
            return s

        with patch.object(_mod, "ResearchRuntime", side_effect=_make_runtime):
            report = await exp.run("AI frameworks")

        assert report.arm_a is not None
        assert report.arm_b is not None
        assert report.arm_a.error is None
        assert report.arm_b.error is None
        assert report.arm_a.quality_race == 80.0
        assert report.arm_a.quality_fact == 90.0
        assert report.arm_a.source_count == 5

    async def test_run_arm_error_handling(self):
        from unittest.mock import AsyncMock, patch

        from houyi.arena import ab_experiment as _mod
        from houyi.arena.ab_experiment import ABExperiment

        llm = AsyncMock()
        ws = AsyncMock()
        exp = ABExperiment(llm, ws)

        def _make_failing_runtime(**kwargs):
            s = AsyncMock()
            s.run_id = "fail_rid"
            s.start = AsyncMock(side_effect=RuntimeError("LLM connection lost"))
            return s

        with patch.object(_mod, "ResearchRuntime", side_effect=_make_failing_runtime):
            report = await exp.run("test query")

        assert report.arm_a.error is not None
        assert "LLM connection lost" in report.arm_a.error
        assert report.arm_b.error is not None
