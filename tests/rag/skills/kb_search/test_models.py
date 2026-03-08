"""Schema tests for kb_search skill models."""

from __future__ import annotations

from houyi.rag.skills.kb_search.skill import (
    KBSearchInput,
    KBSearchOutput,
    Source,
)


class TestKBSearchInput:
    def test_default_values(self) -> None:
        input_data = KBSearchInput(query="test query")
        assert input_data.query == "test query"
        assert input_data.knowledge_dir == "knowledge/"
        assert input_data.mode == "auto"
        assert input_data.max_rounds == 5

    def test_custom_values(self) -> None:
        input_data = KBSearchInput(
            query="custom query",
            knowledge_dir="/custom/path",
            mode="agentic",
            max_rounds=3,
        )
        assert input_data.query == "custom query"
        assert input_data.knowledge_dir == "/custom/path"
        assert input_data.mode == "agentic"
        assert input_data.max_rounds == 3


class TestKBSearchOutput:
    def test_output_creation(self) -> None:
        output = KBSearchOutput(
            answer="Test answer",
            sources=[Source(file_path="/path/to/file.md", location="line 10")],
            confidence=0.85,
        )
        assert output.answer == "Test answer"
        assert len(output.sources) == 1
        assert output.confidence == 0.85

    def test_output_default_sources(self) -> None:
        output = KBSearchOutput(answer="Answer")
        assert output.sources == []
        assert output.confidence == 0.0
