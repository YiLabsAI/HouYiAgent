from __future__ import annotations

import pytest

from houyi.application.research.planner import (
    _VALID_QUERY_TYPES,
    _VALID_SECTION_ARCHETYPES,
    ResearchPlanner,
    _build_plan,
    _force_identity_contract,
    _parse_json_response,
)
from houyi.application.research.types import (
    AnswerCoverageContract,
    CoverageFacet,
    OrchestrationMode,
    OutlineSection,
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchPlan,
    ResearchSettings,
    SearchStrategy,
    SubQuestion,
)

from .conftest import MockLLM


def test_research_settings_defaults() -> None:
    settings = ResearchSettings()
    assert settings.orchestration_mode == OrchestrationMode.DELEGATE
    assert settings.depth.value == "standard"
    assert settings.max_agents == 5


class TestPlannerMetadata:
    def test_subquestion_defaults(self):
        question = SubQuestion(question="What is X?")
        assert question.query_type == "factual"
        assert question.disambiguation_needed is False

    def test_outline_defaults(self):
        section = OutlineSection(title="Overview", objective="Survey")
        assert section.section_archetype == "overview_and_synthesis"

    def test_build_plan_reads_query_type(self):
        plan = _build_plan(
            "test query",
            {
                "sub_questions": [
                    {
                        "question": "Who is person X?",
                        "priority": 5,
                        "query_type": "entity",
                        "disambiguation_needed": True,
                    }
                ],
                "outline": [
                    {
                        "title": "Identity",
                        "objective": "Confirm identity",
                        "related_question_ids": [0],
                        "section_archetype": "overview_and_synthesis",
                    }
                ],
            },
            ResearchSettings(),
        )
        assert plan.sub_questions[0].query_type == "entity"
        assert plan.sub_questions[0].disambiguation_needed is True

    def test_build_plan_reads_archetype(self):
        plan = _build_plan(
            "test",
            {
                "sub_questions": [{"question": "Compare A vs B?", "priority": 5}],
                "outline": [
                    {
                        "title": "Comparison",
                        "objective": "A vs B",
                        "related_question_ids": [0],
                        "section_archetype": "comparison",
                    }
                ],
            },
            ResearchSettings(),
        )
        assert plan.outline[0].section_archetype == "comparison"

    def test_build_plan_fallsback(self):
        plan = _build_plan(
            "q",
            {
                "sub_questions": [{"question": "Q?", "priority": 3, "query_type": "unknown"}],
                "outline": [],
            },
            ResearchSettings(),
        )
        assert plan.sub_questions[0].query_type == "factual"

    def test_build_plan_fallback_archetype(self):
        plan = _build_plan(
            "q",
            {
                "sub_questions": [{"question": "Q?", "priority": 3}],
                "outline": [
                    {
                        "title": "S",
                        "objective": "O",
                        "related_question_ids": [0],
                        "section_archetype": "bogus",
                    }
                ],
            },
            ResearchSettings(),
        )
        assert plan.outline[0].section_archetype == "overview_and_synthesis"

    def test_query_type_values(self):
        assert {"entity", "analytic", "factual"} == _VALID_QUERY_TYPES

    def test_archetype_values(self):
        assert "comparison" in _VALID_SECTION_ARCHETYPES
        assert "risk_and_caveat" in _VALID_SECTION_ARCHETYPES
        assert "trend_and_state" in _VALID_SECTION_ARCHETYPES

    def test_identity_contract_adds_facet(self):
        contract = _force_identity_contract(AnswerCoverageContract())
        assert "identity" in [facet.name for facet in contract.must_cover_facets]

    def test_identity_contract_stays_unique(self):
        contract = _force_identity_contract(
            AnswerCoverageContract(
                must_cover_facets=[CoverageFacet(name="identity", intent="test")]
            )
        )
        identity_count = sum(1 for facet in contract.must_cover_facets if facet.name == "identity")
        assert identity_count == 1

    def test_disambiguation_adds_identity(self):
        plan = _build_plan(
            "weather",
            {
                "sub_questions": [
                    {
                        "question": "What is the weather today?",
                        "priority": 5,
                        "query_type": "entity",
                        "disambiguation_needed": True,
                    }
                ],
                "outline": [],
            },
            ResearchSettings(),
        )
        facet_names = [
            facet.name for facet in plan.sub_questions[0].coverage_contract.must_cover_facets
        ]
        assert "identity" in facet_names

    def test_outline_title_preserved(self):
        """Section titles from LLM output are passed through without
        language normalization — the planner prompt LANGUAGE RULE is the
        correct mechanism for language consistency."""
        plan = _build_plan(
            "English query",
            {
                "sub_questions": [{"question": "Q?", "priority": 3}],
                "outline": [
                    {
                        "title": "Market Overview",
                        "objective": "O",
                        "related_question_ids": [0],
                        "section_archetype": "comparison",
                    }
                ],
            },
            ResearchSettings(),
        )
        assert plan.outline[0].title == "Market Overview"


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

    async def test_standard_floor(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}],"estimated_duration_min":5}',
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}],"estimated_duration_min":5}',
            ]
        )
        planner = ResearchPlanner(llm)
        with pytest.raises(ValueError, match="fewer than 5 sub-questions"):
            await planner.generate_plan(
                "AI agent frameworks", settings=ResearchSettings(depth="standard")
            )

    async def test_returns_clarification(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}],"estimated_duration_min":5,"clarification":{"needs_clarification":true,"confidence":0.4,"issues":["Missing time horizon"],"suggested_questions":["What year range matters?"],"refined_query":"AI agent frameworks in 2026"}}'
            ]
        )
        planner = ResearchPlanner(llm)
        draft = await planner.generate_plan_draft(
            "AI agent frameworks",
            settings=ResearchSettings(depth="quick"),
        )
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
        plan = await planner.generate_plan("test", settings=ResearchSettings(depth="quick"))
        assert len(plan.sub_questions) == 1
        assert len(plan.outline) == 1
        assert llm._call_count == 2

    async def test_malformed_json_raises(self):
        llm = MockLLM(responses=["not valid json at all", "still not valid"])
        planner = ResearchPlanner(llm)
        with pytest.raises(ValueError, match="invalid JSON"):
            await planner.generate_plan("test")

    async def test_parses_coverage_contract(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3,"coverage_contract":{"must_cover_facets":[{"name":"current role","intent":"identify employer","evidence_hint":"official profile","bilingual_terms":["current role","current employer"]}],"required_caveats":["distinguish current from historical roles"]}}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0],"coverage_contract":{"required_caveats":["note uncertainty"]}}],"plan_contract":{"must_cover_facets":[{"name":"identity","intent":"establish who the subject is"}],"comparison_axes":["time"],"time_scope":"2024-2026"},"estimated_duration_min":5}'
            ]
        )
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan("test", settings=ResearchSettings(depth="quick"))
        assert plan.plan_contract.time_scope == "2024-2026"
        assert plan.plan_contract.must_cover_facets[0].name == "identity"
        local_facet = next(
            facet
            for facet in plan.sub_questions[0].coverage_contract.must_cover_facets
            if facet.name == "current role"
        )
        assert local_facet.bilingual_terms == [
            "current role",
            "current employer",
        ]
        assert (
            "distinguish current from historical roles"
            in plan.outline[0].coverage_contract.required_caveats
        )

    async def test_merges_shared_contract(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3,"coverage_contract":{"must_cover_facets":[{"name":"local facet","intent":"local obligation"}]}}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}],"plan_contract":{"must_cover_facets":[{"name":"shared facet","intent":"global obligation"}],"comparison_axes":["time"],"time_scope":"2024-2026","required_caveats":["note uncertainty"]},"estimated_duration_min":5}'
            ]
        )
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan("test", settings=ResearchSettings(depth="quick"))
        facet_names = [
            facet.name for facet in plan.sub_questions[0].coverage_contract.must_cover_facets
        ]
        assert facet_names == ["local facet", "shared facet"]
        assert plan.sub_questions[0].coverage_contract.time_scope == "2024-2026"
        assert plan.sub_questions[0].coverage_contract.required_caveats == ["note uncertainty"]

    async def test_derives_outline_contract(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3,"coverage_contract":{"must_cover_facets":[{"name":"facet a","intent":"local a"}],"required_caveats":["caveat a"]}},{"question":"Q2","priority":4,"search_strategy":"web","expected_sources":3,"coverage_contract":{"must_cover_facets":[{"name":"facet b","intent":"local b"}],"evidence_expectations":["expectation b"]}}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0,1]}],"plan_contract":{"must_cover_facets":[{"name":"shared facet","intent":"global obligation"}],"comparison_axes":["time"],"required_caveats":["global caveat"]},"estimated_duration_min":5}'
            ]
        )
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan("test", settings=ResearchSettings(depth="quick"))
        outline_contract = plan.outline[0].coverage_contract
        assert [facet.name for facet in outline_contract.must_cover_facets] == [
            "facet a",
            "facet b",
            "shared facet",
        ]
        assert outline_contract.comparison_axes == ["time"]
        assert outline_contract.required_caveats == ["global caveat", "caveat a"]
        assert outline_contract.evidence_expectations == ["expectation b"]

    async def test_outline_shared_fallback(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Q1","priority":5,"search_strategy":"web","expected_sources":3}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}],"plan_contract":{"must_cover_facets":[{"name":"shared facet","intent":"global obligation"}],"comparison_axes":["time"]},"estimated_duration_min":5}'
            ]
        )
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan("test", settings=ResearchSettings(depth="quick"))
        outline_contract = plan.outline[0].coverage_contract
        assert [facet.name for facet in outline_contract.must_cover_facets] == ["shared facet"]
        assert outline_contract.comparison_axes == ["time"]

    async def test_entity_contract_added(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"Who is Sample Person?","priority":5,"search_strategy":"web","expected_sources":3}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}],"estimated_duration_min":5}'
            ]
        )
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan(
            "Who is Sample Person?", settings=ResearchSettings(depth="quick")
        )
        contract = plan.sub_questions[0].coverage_contract
        assert contract.must_cover_facets[0].name == "identity"
        assert "disambiguate same-name entities before making claims" in contract.required_caveats
        assert "official identity evidence" in contract.evidence_expectations

    async def test_expand_deep_outline(self):
        llm = MockLLM(
            responses=[
                '{"sub_questions":[{"question":"How is the nine-tier social structure defined?","priority":5,"search_strategy":"web","expected_sources":3,"coverage_contract":{"must_cover_facets":[{"name":"stratification model","intent":"Explain the classification standard"}]}},{"question":"How do income and household finances differ across tiers?","priority":4,"search_strategy":"web","expected_sources":3,"coverage_contract":{"must_cover_facets":[{"name":"income and finance","intent":"Summarize income, assets, and liabilities"}]}},{"question":"How is the middle class defined and characterized?","priority":3,"search_strategy":"web","expected_sources":3,"coverage_contract":{"must_cover_facets":[{"name":"middle class definition","intent":"Define the middle class and key traits"}]}},{"question":"How can middle-class size and financial strength be estimated?","priority":2,"search_strategy":"web","expected_sources":3,"coverage_contract":{"must_cover_facets":[{"name":"middle class scale","intent":"Estimate scale and financial capacity"}]}}],"outline":[{"title":"Overview","objective":"Summarize stratification and middle-class dynamics","related_question_ids":[0,1,2,3]}],"plan_contract":{"must_cover_facets":[{"name":"limits and disputes","intent":"Explain methodological limits and disagreements"}]},"estimated_duration_min":8}'
            ]
        )
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan(
            "Summarize the nine-tier income structure and middle-class conditions",
            settings=ResearchSettings(depth="deep"),
        )
        assert len(plan.outline) >= 5
        related_sizes = [len(section.related_question_ids) for section in plan.outline]
        assert any(size == 1 for size in related_sizes)


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

    def test_embedded_json_text(self):
        data = _parse_json_response('Plan draft follows: {"key": "val", "items": [1, 2]} End.')
        assert data == {"key": "val", "items": [1, 2]}

    def test_trailing_commas(self):
        data = _parse_json_response('{"key": "val", "items": [1, 2,],}')
        assert data == {"key": "val", "items": [1, 2]}

    def test_smart_quotes(self):
        data = _parse_json_response('“{"key": “val”}”')
        assert data == {"key": "val"}

    def test_unescaped_inner_quotes_repaired(self):
        data = _parse_json_response(
            '```json\n{"sub_questions": [{"question": "topic "alpha" analysis", "priority": 3}], "outline": []}\n```'
        )
        assert data is not None
        assert data["sub_questions"][0]["question"] == 'topic "alpha" analysis'


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
        await planner.generate_plan("quantum computing", settings=ResearchSettings(depth="quick"))
        assert llm._call_count == 1

    async def test_invalid_strategy_fallback(self):
        raw = '{"sub_questions":[{"question":"Q1","search_strategy":"quantum"}],"outline":[{"title":"Overview","objective":"Explain the topic","related_question_ids":[0]}]}'
        llm = MockLLM(responses=[raw])
        planner = ResearchPlanner(llm)
        plan = await planner.generate_plan("test", settings=ResearchSettings(depth="quick"))
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
