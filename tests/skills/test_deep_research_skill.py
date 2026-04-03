"""Unit tests for deep_research skill — Chatbox integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.skills.builtin.deep_research import (
    DeepResearchInput,
    DeepResearchOutput,
    build_deep_research_skill,
    execute_deep_research,
    set_research_service,
)


class TestBuildSkill:
    def test_skill_spec_valid(self):
        spec = build_deep_research_skill()
        assert spec.name == "deep_research"
        assert spec.input_schema is DeepResearchInput
        assert spec.output_schema is DeepResearchOutput

    def test_skill_description(self):
        spec = build_deep_research_skill()
        assert "research" in spec.description.lower()

    def test_skill_requires_network(self):
        spec = build_deep_research_skill()
        assert spec.permissions.network.enabled is True


class TestExecuteNoService:
    """When no ResearchService is injected, returns a placeholder."""

    async def test_no_services_returns_stub(self):
        with patch("houyi.skills.builtin.deep_research._research_service_ref", None):
            result = await execute_deep_research(query="test", depth="quick")
            assert result["session_id"] == ""
            assert "No ResearchService" in result["summary"]

    async def test_depth_parameter_accepted(self):
        with patch("houyi.skills.builtin.deep_research._research_service_ref", None):
            for depth in ("quick", "standard", "deep"):
                result = await execute_deep_research(query="test", depth=depth)
                assert isinstance(result, dict)


class TestExecuteWithService:
    """When a ResearchService is injected, runs a full session."""

    @pytest.fixture(autouse=True)
    def _mock_service(self):
        mock_report = MagicMock()
        mock_report.title = "Test Report"
        mock_section = MagicMock()
        mock_section.title = "Intro"
        mock_section.content = "Some findings."
        mock_report.sections = [mock_section]
        mock_ref = MagicMock()
        mock_ref.url = "https://example.com"
        mock_report.references = [mock_ref]
        mock_report.quality_score = MagicMock()
        mock_report.quality_score.race_overall = 8.0
        mock_report.quality_score.fact_overall = 7.0

        mock_session = MagicMock()
        mock_session.session_id = "rs_test123"

        self.svc = MagicMock()
        self.svc.create_session = AsyncMock(return_value=(mock_session, MagicMock()))
        self.svc.confirm_and_execute = AsyncMock()
        self.svc.get_report = AsyncMock(return_value=mock_report)

        set_research_service(self.svc)
        yield
        set_research_service(None)

    async def test_returns_session_id(self):
        result = await execute_deep_research(query="AI frameworks", depth="standard")
        assert result["session_id"] == "rs_test123"

    async def test_returns_report_summary(self):
        result = await execute_deep_research(query="AI frameworks")
        assert "Test Report" in result["summary"]
        assert "Intro" in result["summary"]

    async def test_returns_report_url(self):
        result = await execute_deep_research(query="AI frameworks")
        assert result["report_url"] == "#/research/rs_test123"

    async def test_returns_quality_score(self):
        result = await execute_deep_research(query="AI frameworks")
        assert result["quality_score"] == pytest.approx(7.5)

    async def test_returns_sources_count(self):
        result = await execute_deep_research(query="AI frameworks")
        assert result["sources_count"] == 1

    async def test_calls_service_methods(self):
        await execute_deep_research(query="AI frameworks", depth="deep")
        self.svc.create_session.assert_awaited_once()
        self.svc.confirm_and_execute.assert_awaited_once_with("rs_test123")
        self.svc.get_report.assert_awaited_once_with("rs_test123")

    async def test_handles_service_error_gracefully(self):
        self.svc.create_session = AsyncMock(side_effect=RuntimeError("boom"))
        result = await execute_deep_research(query="fail")
        assert result["session_id"] == ""
        assert "failed" in result["summary"].lower()


class TestInputModel:
    def test_defaults(self):
        inp = DeepResearchInput(query="test")
        assert inp.depth == "quick"

    def test_custom_depth(self):
        inp = DeepResearchInput(query="test", depth="deep")
        assert inp.depth == "deep"


class TestOutputModel:
    def test_defaults(self):
        out = DeepResearchOutput(session_id="s1", summary="done")
        assert out.report_url is None
        assert out.sources_count == 0
