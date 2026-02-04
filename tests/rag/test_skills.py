"""Tests for RAG skills."""


from houyi.rag.skills.kb_search.skill import (
    KBSearchInput,
    KBSearchOutput,
    Source,
    kb_search_skill,
)


class TestKBSearchInput:
    """Tests for KBSearchInput schema."""

    def test_default_values(self) -> None:
        """Test default input values."""
        input_data = KBSearchInput(query="test query")
        assert input_data.query == "test query"
        assert input_data.knowledge_dir == "knowledge/"
        assert input_data.mode == "auto"
        assert input_data.max_rounds == 5

    def test_custom_values(self) -> None:
        """Test custom input values."""
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
    """Tests for KBSearchOutput schema."""

    def test_output_creation(self) -> None:
        """Test creating output."""
        output = KBSearchOutput(
            answer="Test answer",
            sources=[
                Source(file_path="/path/to/file.md", location="line 10"),
            ],
            confidence=0.85,
        )
        assert output.answer == "Test answer"
        assert len(output.sources) == 1
        assert output.confidence == 0.85

    def test_output_default_sources(self) -> None:
        """Test output with default empty sources."""
        output = KBSearchOutput(answer="Answer")
        assert output.sources == []
        assert output.confidence == 0.0


class TestKBSearchSkill:
    """Tests for kb_search_skill definition."""

    def test_skill_name(self) -> None:
        """Test skill name."""
        assert kb_search_skill.name == "kb-search"

    def test_skill_description(self) -> None:
        """Test skill has description."""
        assert kb_search_skill.description is not None
        assert len(kb_search_skill.description) > 0

    def test_skill_schemas(self) -> None:
        """Test skill has input/output schemas."""
        assert kb_search_skill.input_schema == KBSearchInput
        assert kb_search_skill.output_schema == KBSearchOutput

    def test_skill_executor(self) -> None:
        """Test skill has executor."""
        assert kb_search_skill.executor is not None

    def test_skill_metadata(self) -> None:
        """Test skill metadata."""
        assert kb_search_skill.metadata.get("category") == "rag"
        assert kb_search_skill.metadata.get("version") == "1.0.0"
