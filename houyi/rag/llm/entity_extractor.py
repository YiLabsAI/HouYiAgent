"""Extract entities and relations from text using LLM."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from houyi.rag.types import Chunk, Entity, Relation

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class LLMEntityExtractor:
    """Extract entities and relations using LLM for knowledge graph construction.

    Extracts:
    - Named entities with types (person, org, concept, technology, etc.)
    - Relations between entities (causal, temporal, contains, relates_to)
    - Entity descriptions for better embedding

    Reference: GraphRAG (Microsoft Research)
    """

    SYSTEM_PROMPT = """You are a knowledge graph extraction expert.
Extract entities and relations from the given text.

Return ONLY valid JSON in this format:
{
  "entities": [
    {"name": "Entity Name", "type": "concept|person|org|technology|location|event", "description": "brief description"}
  ],
  "relations": [
    {"source": "Entity1", "target": "Entity2", "type": "causes|contains|relates_to|uses|part_of|located_in", "description": "relationship description"}
  ]
}

Guidelines:
- Extract 5-15 most important entities
- Focus on domain-specific concepts and proper nouns
- Create relations only between extracted entities
- Use consistent entity names (no duplicates with slight variations)
- Descriptions should be 10-30 words"""

    def __init__(
        self,
        adapter: LLMAdapter,
        temperature: float = 0.2,
        max_entities: int = 20,
    ) -> None:
        """Initialize entity extractor.

        Args:
            adapter: LLM adapter for making API calls
            temperature: LLM temperature (lower = more consistent)
            max_entities: Maximum entities to extract per chunk
        """
        self._adapter = adapter
        self._temperature = temperature
        self._max_entities = max_entities

    async def extract(
        self,
        chunk: Chunk,
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from a chunk.

        Args:
            chunk: Text chunk to process

        Returns:
            Tuple of (entities, relations)
        """
        from houyi.llm.base import LLMMessage, MessageRole

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
            LLMMessage(
                role=MessageRole.USER,
                content=f"Extract entities and relations from:\n\n{chunk.content}",
            ),
        ]

        try:
            response = await self._adapter.chat(
                messages=messages,
                temperature=self._temperature,
                max_tokens=1000,
            )
            return self._parse_response(response.content, chunk.chunk_id)
        except Exception as e:
            logger.warning("Entity extraction failed: %s, using simple extraction", e)
            return self._simple_extract(chunk)

    async def extract_batch(
        self,
        chunks: list[Chunk],
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract entities from multiple chunks with deduplication.

        Args:
            chunks: List of chunks to process

        Returns:
            Tuple of (deduplicated entities, all relations)
        """
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        entity_name_map: dict[str, str] = {}  # name -> entity_id

        for chunk in chunks:
            entities, relations = await self.extract(chunk)

            # Deduplicate entities by name
            for entity in entities:
                name_lower = entity.name.lower()
                if name_lower not in entity_name_map:
                    entity_name_map[name_lower] = entity.entity_id
                    all_entities.append(entity)

            # Update relation IDs to use deduplicated entity IDs
            for relation in relations:
                source_lower = relation.metadata.get("source_name", "").lower()
                target_lower = relation.metadata.get("target_name", "").lower()

                if source_lower in entity_name_map and target_lower in entity_name_map:
                    updated = Relation(
                        rel_id=relation.rel_id,
                        source_id=entity_name_map[source_lower],
                        target_id=entity_name_map[target_lower],
                        rel_type=relation.rel_type,
                        weight=relation.weight,
                        metadata=relation.metadata,
                    )
                    all_relations.append(updated)

        return all_entities, all_relations

    def _parse_response(
        self,
        content: str,
        chunk_id: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """Parse LLM response to extract entities and relations."""
        try:
            # Clean up response
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(line for line in lines if not line.startswith("```"))

            data = json.loads(content)

            # Parse entities
            entities: list[Entity] = []
            entity_name_to_id: dict[str, str] = {}

            for e in data.get("entities", [])[: self._max_entities]:
                name = e.get("name", "").strip()
                if not name:
                    continue

                entity_id = str(uuid.uuid4())
                entity_name_to_id[name.lower()] = entity_id

                entities.append(
                    Entity(
                        entity_id=entity_id,
                        name=name,
                        entity_type=e.get("type", "concept"),
                        metadata={
                            "description": e.get("description", ""),
                            "source_chunk": chunk_id,
                        },
                    )
                )

            # Parse relations
            relations: list[Relation] = []

            for r in data.get("relations", []):
                source = r.get("source", "").strip()
                target = r.get("target", "").strip()

                source_id = entity_name_to_id.get(source.lower())
                target_id = entity_name_to_id.get(target.lower())

                if source_id and target_id:
                    relations.append(
                        Relation(
                            rel_id=str(uuid.uuid4()),
                            source_id=source_id,
                            target_id=target_id,
                            rel_type=r.get("type", "relates_to"),
                            weight=1.0,
                            metadata={
                                "description": r.get("description", ""),
                                "source_name": source,
                                "target_name": target,
                                "source_chunk": chunk_id,
                            },
                        )
                    )

            return entities, relations

        except json.JSONDecodeError:
            logger.warning("Failed to parse entity extraction response")
            return [], []

    def _simple_extract(
        self,
        chunk: Chunk,
    ) -> tuple[list[Entity], list[Relation]]:
        """Simple rule-based extraction as fallback."""
        import re

        entities: list[Entity] = []

        # Extract capitalized phrases (2-4 words)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
        matches = re.findall(pattern, chunk.content)

        seen: set[str] = set()
        for match in matches:
            name = match.strip()
            if len(name) > 3 and name.lower() not in seen:
                seen.add(name.lower())
                entities.append(
                    Entity(
                        entity_id=str(uuid.uuid4()),
                        name=name,
                        entity_type="concept",
                        metadata={"source_chunk": chunk.chunk_id},
                    )
                )

        # Create co-occurrence relations
        relations: list[Relation] = []
        entity_list = entities[:10]  # Limit

        for i, e1 in enumerate(entity_list):
            for e2 in entity_list[i + 1 :]:
                relations.append(
                    Relation(
                        rel_id=str(uuid.uuid4()),
                        source_id=e1.entity_id,
                        target_id=e2.entity_id,
                        rel_type="co_occurs",
                        weight=0.5,
                        metadata={
                            "source_name": e1.name,
                            "target_name": e2.name,
                            "source_chunk": chunk.chunk_id,
                        },
                    )
                )

        return entities[: self._max_entities], relations
