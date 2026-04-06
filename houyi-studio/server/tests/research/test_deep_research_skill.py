"""Tests for deep_research skill registration and invocation."""

from __future__ import annotations

from unittest.mock import patch

from houyi.skills.builtin import deep_research as _dr_mod
from houyi.skills.builtin.deep_research import (
    DeepResearchInput,
    DeepResearchOutput,
    build_deep_research_skill,
)


class TestSkillSpec:
    def test_skill_name(self):
        skill = build_deep_research_skill()
        assert skill.name == "deep_research"

    def test_skill_schemas(self):
        skill = build_deep_research_skill()
        assert skill.input_schema is DeepResearchInput
        assert skill.output_schema is DeepResearchOutput

    def test_input_validation(self):
        inp = DeepResearchInput(query="test research", depth="deep")
        assert inp.query == "test research"
        assert inp.depth == "deep"

    def test_default_depth(self):
        inp = DeepResearchInput(query="q")
        assert inp.depth == "quick"


class TestSkillExecution:
    async def test_executor_returns_dict(self):
        skill = build_deep_research_skill()
        with patch.object(_dr_mod, "_research_service_ref", None):
            result = await skill.executor(query="AI frameworks", depth="standard")
        assert isinstance(result, dict)
        assert "run_id" in result
        assert "summary" in result

    async def test_executor_placeholder(self):
        skill = build_deep_research_skill()
        with patch.object(_dr_mod, "_research_service_ref", None):
            result = await skill.executor(query="test")
        assert "ResearchService" in result["summary"]
