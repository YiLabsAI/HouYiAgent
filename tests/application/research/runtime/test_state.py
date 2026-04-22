from houyi.application.research.runtime.state import ResearchRunState


class TestResearchRunState:
    def test_state_defaults(self) -> None:
        state = ResearchRunState()
        assert state.plan is None
        assert state.search_results == []
        assert state.intermediate_reports == []
        assert state.conflicts == []
        assert state.aggregated_sources is None
        assert state.report is None
        assert state.quality_score is None
        assert state.memory_candidates == []
        assert state.error is None
        assert state.event_sequence == 0
        assert state.cancelled is False
        assert state.execution_phase == "init"

    def test_state_mutation(self) -> None:
        state = ResearchRunState()
        state.execution_phase = "search"
        state.event_sequence = 5
        state.cancelled = True
        state.error = "timeout"
        assert state.execution_phase == "search"
        assert state.event_sequence == 5
        assert state.cancelled is True
        assert state.error == "timeout"

    def test_state_list_fields(self) -> None:
        s1 = ResearchRunState()
        s2 = ResearchRunState()
        s1.search_results.append("x")  # type: ignore[arg-type]
        assert s2.search_results == [], "field factories must be independent"
