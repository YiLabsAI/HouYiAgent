"""Export-surface tests for houyi.rag.skills."""

from __future__ import annotations


class TestSkillsExports:
    def test_skill_imports(self) -> None:
        from houyi.rag.skills import (
            kb_analyze_skill,
            kb_graph_skill,
            kb_ingest_skill,
            kb_search_skill,
        )

        assert kb_search_skill is not None
        assert kb_ingest_skill is not None
        assert kb_graph_skill is not None
        assert kb_analyze_skill is not None

    def test_skill_definitions(self) -> None:
        from houyi.rag.skills import (
            kb_analyze_skill,
            kb_graph_skill,
            kb_ingest_skill,
            kb_search_skill,
        )

        assert kb_search_skill.name == "kb-search"
        assert kb_ingest_skill.name == "kb-ingest"
        assert kb_graph_skill.name == "kb-graph"
        assert kb_analyze_skill.name == "kb-analyze"
