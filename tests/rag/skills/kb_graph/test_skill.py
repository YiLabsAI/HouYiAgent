"""Tests for kb-graph skill."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import houyi.rag.skills.kb_graph.skill as kb_graph_module
from houyi.rag.skills.kb_graph.skill import (
    Entity,
    KBGraphInput,
    KBGraphOutput,
    Relation,
    execute_kb_graph,
    kb_graph_skill,
)


class TestModels:
    """Tests for data models."""

    def test_entity_creation(self) -> None:
        """Test Entity creation."""
        entity = Entity(id="e1", name="Alice", type="Person", score=0.9)
        assert entity.id == "e1"
        assert entity.name == "Alice"
        assert entity.type == "Person"
        assert entity.score == 0.9

    def test_entity_defaults(self) -> None:
        """Test Entity default values."""
        entity = Entity(id="e1", name="Bob")
        assert entity.type == ""
        assert entity.score == 0.0

    def test_relation_creation(self) -> None:
        """Test Relation creation."""
        relation = Relation(source="e1", target="e2", type="knows")
        assert relation.source == "e1"
        assert relation.target == "e2"
        assert relation.type == "knows"

    def test_kb_graph_input_defaults(self) -> None:
        """Test KBGraphInput defaults."""
        input_data = KBGraphInput(query="find Alice")
        assert input_data.query == "find Alice"
        assert input_data.knowledge_dir == "knowledge/"
        assert input_data.start_entities == []
        assert input_data.max_hops == 3
        assert input_data.top_k == 10
        assert input_data.use_ppr is True

    def test_kb_graph_input_custom(self) -> None:
        """Test KBGraphInput custom values."""
        input_data = KBGraphInput(
            query="find relations",
            start_entities=["e1", "e2"],
            max_hops=5,
            top_k=20,
            use_ppr=False,
        )
        assert input_data.start_entities == ["e1", "e2"]
        assert input_data.max_hops == 5
        assert input_data.top_k == 20
        assert input_data.use_ppr is False

    def test_kb_graph_output_defaults(self) -> None:
        """Test KBGraphOutput defaults."""
        output = KBGraphOutput()
        assert output.entities == []
        assert output.relations == []
        assert output.paths == []
        assert output.confidence == 0.0

    def test_output_with_data(self) -> None:
        """Test KBGraphOutput with data."""
        output = KBGraphOutput(
            entities=[Entity(id="e1", name="Alice")],
            relations=[Relation(source="e1", target="e2", type="knows")],
            paths=[["e1", "e2"]],
            confidence=0.85,
        )
        assert len(output.entities) == 1
        assert len(output.relations) == 1
        assert output.confidence == 0.85


class TestExecuteKBGraph:
    """Tests for execute_kb_graph function."""

    @pytest.mark.asyncio
    async def test_execute_success(self, patch_skill_rag_builder) -> None:
        """Test successful graph query execution."""
        mock_entity = MagicMock()
        mock_entity.id = "e1"
        mock_entity.name = "Alice"
        mock_entity.type = "Person"
        mock_entity.score = 0.9

        mock_relation = MagicMock()
        mock_relation.source = "e1"
        mock_relation.target = "e2"
        mock_relation.type = "knows"

        mock_result = MagicMock()
        mock_result.entities = [mock_entity]
        mock_result.relations = [mock_relation]
        mock_result.paths = [["e1", "e2"]]
        mock_result.confidence = 0.85

        mock_rag = MagicMock()
        mock_rag.graph_query = AsyncMock(return_value=mock_result)

        with patch_skill_rag_builder(kb_graph_module, mock_rag):
            input_data = KBGraphInput(query="find Alice")
            result = await execute_kb_graph(input_data)

        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"
        assert len(result.relations) == 1
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_execute_with_start_entities(self, patch_skill_rag_builder) -> None:
        """Test graph query with start entities."""
        mock_result = MagicMock()
        mock_result.entities = []
        mock_result.relations = []
        mock_result.paths = []
        mock_result.confidence = 0.5

        mock_rag = MagicMock()
        mock_rag.graph_query = AsyncMock(return_value=mock_result)

        with patch_skill_rag_builder(kb_graph_module, mock_rag) as mock_rag_builder:
            input_data = KBGraphInput(
                query="find relations",
                knowledge_dir="/custom/kb",
                start_entities=["e1", "e2"],
                max_hops=5,
            )
            result = await execute_kb_graph(input_data)

        call_kwargs = mock_rag.graph_query.call_args.kwargs
        mock_rag_builder.assert_called_once_with(
            mode="indexed",
            knowledge_dir="/custom/kb",
        )
        assert call_kwargs["start_entities"] == ["e1", "e2"]
        assert call_kwargs["max_hops"] == 5
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_execute_failure(self, patch_skill_rag_builder) -> None:
        """Test graph query failure handling."""
        mock_rag = MagicMock()
        mock_rag.graph_query = AsyncMock(side_effect=Exception("Graph query failed"))

        with patch_skill_rag_builder(kb_graph_module, mock_rag):
            input_data = KBGraphInput(query="test")
            result = await execute_kb_graph(input_data)

        assert result.entities == []
        assert result.relations == []
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_execute_multiple_entities(self, patch_skill_rag_builder) -> None:
        """Test graph query with multiple entities."""
        mock_entity1 = MagicMock()
        mock_entity1.id = "e0"
        mock_entity1.name = "Entity0"
        mock_entity1.type = "Type"
        mock_entity1.score = 0.9

        mock_entity2 = MagicMock()
        mock_entity2.id = "e1"
        mock_entity2.name = "Entity1"
        mock_entity2.type = "Type"
        mock_entity2.score = 0.8

        mock_entity3 = MagicMock()
        mock_entity3.id = "e2"
        mock_entity3.name = "Entity2"
        mock_entity3.type = "Type"
        mock_entity3.score = 0.7

        mock_result = MagicMock()
        mock_result.entities = [mock_entity1, mock_entity2, mock_entity3]
        mock_result.relations = []
        mock_result.paths = []
        mock_result.confidence = 0.8

        mock_rag = MagicMock()
        mock_rag.graph_query = AsyncMock(return_value=mock_result)

        with patch_skill_rag_builder(kb_graph_module, mock_rag):
            input_data = KBGraphInput(query="find all", top_k=3)
            result = await execute_kb_graph(input_data)

        assert len(result.entities) == 3
        assert result.entities[0].name == "Entity0"

    @pytest.mark.asyncio
    async def test_execute_ppr_disabled(self, patch_skill_rag_builder) -> None:
        """Test graph query with PPR disabled."""
        mock_result = MagicMock()
        mock_result.entities = []
        mock_result.relations = []
        mock_result.paths = []
        mock_result.confidence = 0.5

        mock_rag = MagicMock()
        mock_rag.graph_query = AsyncMock(return_value=mock_result)

        with patch_skill_rag_builder(kb_graph_module, mock_rag):
            input_data = KBGraphInput(query="test", use_ppr=False)
            await execute_kb_graph(input_data)

        call_kwargs = mock_rag.graph_query.call_args.kwargs
        assert call_kwargs["use_ppr"] is False


class TestKBGraphSkill:
    """Tests for skill definition."""

    def test_skill_definition(self) -> None:
        """Test skill is properly defined."""
        assert kb_graph_skill.name == "kb-graph"
        assert kb_graph_skill.input_schema == KBGraphInput
        assert kb_graph_skill.output_schema == KBGraphOutput
        assert kb_graph_skill.executor == execute_kb_graph
        assert kb_graph_skill.version == "1.0.0"
        assert kb_graph_skill.user_invocable is True
        assert "Read" in kb_graph_skill.allowed_tools
        assert "Grep" in kb_graph_skill.allowed_tools
        assert len(kb_graph_skill.hooks) == 3
