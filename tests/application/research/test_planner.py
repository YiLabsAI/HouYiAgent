"""Unit tests for ResearchPlanner."""

from __future__ import annotations

import pytest

from houyi.application.research.planner import ResearchPlanner, _parse_json_response
from houyi.application.research.types import (
    OrchestrationMode,
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchPlan,
    ResearchSettings,
    SearchStrategy,
    SubQuestion,
)

from .conftest import MockLLM


class TestGeneratePlan:
    async def test_generates_sub_questions(self, mock_llm):
        planner = ResearchPlanner(mock_llm)
        plan = await planner.generate_plan("AI agent frameworks")
        assert 1 <= len(plan.sub_questions) <= 8
        assert plan.status == PlanStatus.DRAFT
        assert plan.query == "AI agent frameworks"

    async def test_plan_has_outline(self, mock_llm):
        planner = ResearchPlanner(mock_llm)
        plan = await planner.generate_plan("AI agent frameworks")
        assert len(plan.outline) >= 1
        assert plan.outline[0].title

    async def test_plan_with_settings(self, mock_llm):
        settings = ResearchSettings(max_search_rounds=5)
        planner = ResearchPlanner(mock_llm)
        plan = await planner.generate_plan("test", settings=settings)
        assert plan.settings.max_search_rounds == 5

    async def test_plan_with_memory(self, mock_llm):
        planner = ResearchPlanner(mock_llm)
        plan = await planner.generate_plan("test", memory_context="User prefers Python")
        assert plan is not None

    async def test_returns_clarification(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}],"estimated_duration_min":5,"clarification":{"needs_clarification":true,"confidence":0.4,"issues":["Missing time horizon"],"suggested_questions":["What year range matters?"],"refined_query":"AI agent frameworks in 2026"}}'
            ]
        )
        planner = ResearchPlanner(llm)
        draft = await planner.generate_plan_draft("AI agent frameworks")
        assert draft.plan.query == "AI agent frameworks"
        assert draft.clarification is not None
        assert draft.clarification.needs_clarification is True
        assert draft.clarification.refined_query == "AI agent frameworks in 2026"

    async def test_malformed_json_retries(self):
        llm = MockLLM(
            responses=[
                "not valid json at all",
                '{"sub_questions":[{"question":"Retry question","priority":5,"search_strategy":"web","expected_sources":3}],"outline":[{"title":"Retry section","objective":"Retry objective","related_question_ids":[0]}],"estimated_duration_min":5}',
            ]
        )
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan("test")
        assert len(plan.sub_questions) == 1
        assert len(plan.outline) == 1
        assert llm._call_count == 2

    async def test_malformed_json_raises(self):
        llm = MockLLM(responses=["not valid json at all", "still not valid"])
        planner = ResearchPlanner(llm)
        with pytest.raises(ValueError, match="invalid JSON"):
            await planner.generate_plan("test")


class TestRefinePlan:
    async def test_add_question(self):
        plan = ResearchPlan(query="test", sub_questions=[], status=PlanStatus.DRAFT)
        llm = MockLLM()
        planner = ResearchPlanner(llm)
        edit = PlanEdit(op=PlanEditOperation.ADD, target_question="New question?")
        updated = await planner.refine_plan(plan, [edit])
        assert len(updated.sub_questions) == 1
        assert updated.version == 2

    async def test_delete_question(self):
        sq = SubQuestion(question="Remove me")
        plan = ResearchPlan(query="test", sub_questions=[sq], status=PlanStatus.DRAFT)
        planner = ResearchPlanner(MockLLM())
        edit = PlanEdit(op=PlanEditOperation.DELETE, question_id=sq.question_id)
        updated = await planner.refine_plan(plan, [edit])
        assert len(updated.sub_questions) == 0

    async def test_update_question(self):
        sq = SubQuestion(question="Old text")
        plan = ResearchPlan(query="test", sub_questions=[sq], status=PlanStatus.DRAFT)
        planner = ResearchPlanner(MockLLM())
        edit = PlanEdit(
            op=PlanEditOperation.UPDATE, question_id=sq.question_id, target_question="New text"
        )
        updated = await planner.refine_plan(plan, [edit])
        assert updated.sub_questions[0].question == "New text"

    async def test_set_priority(self):
        sq = SubQuestion(question="Q1", priority=3)
        plan = ResearchPlan(query="test", sub_questions=[sq], status=PlanStatus.DRAFT)
        planner = ResearchPlanner(MockLLM())
        edit = PlanEdit(
            op=PlanEditOperation.SET_PRIORITY, question_id=sq.question_id, new_priority=5
        )
        updated = await planner.refine_plan(plan, [edit])
        assert updated.sub_questions[0].priority == 5

    async def test_set_strategy(self):
        sq = SubQuestion(question="Q1")
        plan = ResearchPlan(query="test", sub_questions=[sq], status=PlanStatus.DRAFT)
        planner = ResearchPlanner(MockLLM())
        edit = PlanEdit(
            op=PlanEditOperation.SET_STRATEGY,
            question_id=sq.question_id,
            new_search_strategy=SearchStrategy.RAG,
        )
        updated = await planner.refine_plan(plan, [edit])
        assert updated.sub_questions[0].search_strategy == SearchStrategy.RAG

    async def test_reject_executing_plan(self):
        plan = ResearchPlan(query="test", sub_questions=[], status=PlanStatus.EXECUTING)
        planner = ResearchPlanner(MockLLM())
        with pytest.raises(ValueError, match="Cannot edit"):
            await planner.refine_plan(plan, [])

    async def test_version_conflict_rejected(self):
        plan = ResearchPlan(query="test", sub_questions=[], version=3, status=PlanStatus.DRAFT)
        planner = ResearchPlanner(MockLLM())
        with pytest.raises(ValueError, match="version conflict"):
            await planner.refine_plan(plan, [], expected_version=1)

    async def test_version_match_accepted(self):
        plan = ResearchPlan(query="test", sub_questions=[], version=3, status=PlanStatus.DRAFT)
        planner = ResearchPlanner(MockLLM())
        updated = await planner.refine_plan(plan, [], expected_version=3)
        assert updated.version == 4


class TestParseJson:
    def test_plain_json(self):
        data = _parse_json_response('{"key": "val"}')
        assert data["key"] == "val"

    def test_fenced_json(self):
        data = _parse_json_response('```json\n{"key": "val"}\n```')
        assert data["key"] == "val"

    def test_invalid_returns_none(self):
        data = _parse_json_response("garbage")
        assert data is None


class TestBoundaryAndInteraction:
    async def test_empty_query_produces_plan(self, mock_llm):
        planner = ResearchPlanner(mock_llm)
        plan = await planner.generate_plan("")
        assert plan.query == ""
        assert plan.status == PlanStatus.DRAFT

    async def test_garbage_json_raises(self):
        llm = MockLLM(responses=["<html>not json at all</html>", "still bad"])
        planner = ResearchPlanner(llm)
        with pytest.raises(ValueError, match="invalid JSON"):
            await planner.generate_plan("test")

    async def test_llm_called_with_query(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}]}'
            ]
        )
        planner = ResearchPlanner(llm)
        await planner.generate_plan("quantum computing")
        assert llm._call_count == 1

    async def test_invalid_strategy_fallback(self):
        raw = '{"sub_questions":[{"question":"Q1","search_strategy":"quantum"}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}]}'
        llm = MockLLM(responses=[raw])
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan("test")
        assert plan.sub_questions[0].search_strategy == SearchStrategy.WEB

    async def test_move_reorders_questions(self):
        sq1 = SubQuestion(question="First")
        sq2 = SubQuestion(question="Second")
        sq3 = SubQuestion(question="Third")
        plan = ResearchPlan(query="test", sub_questions=[sq1, sq2, sq3], status=PlanStatus.DRAFT)
        planner = ResearchPlanner(MockLLM())
        edit = PlanEdit(
            op=PlanEditOperation.MOVE,
            question_id=sq3.question_id,
            after_question_id=sq1.question_id,
        )
        updated = await planner.refine_plan(plan, [edit])
        assert [sq.question for sq in updated.sub_questions] == ["First", "Third", "Second"]

    async def test_move_to_front(self):
        sq1 = SubQuestion(question="First")
        sq2 = SubQuestion(question="Second")
        plan = ResearchPlan(query="test", sub_questions=[sq1, sq2], status=PlanStatus.DRAFT)
        planner = ResearchPlanner(MockLLM())
        edit = PlanEdit(
            op=PlanEditOperation.MOVE,
            question_id=sq2.question_id,
        )
        updated = await planner.refine_plan(plan, [edit])
        assert updated.sub_questions[-1].question == "Second"


def test_research_settings_defaults() -> None:
    settings = ResearchSettings()
    assert settings.orchestration_mode == OrchestrationMode.DELEGATE
    assert settings.depth.value == "standard"
    assert settings.max_agents == 5
