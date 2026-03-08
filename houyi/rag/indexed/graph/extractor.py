"""Heuristic entity and relation extraction for indexed graph ingest."""

from __future__ import annotations

import uuid

from houyi.rag.types import Chunk, Entity, Relation


async def extract_entities(
    chunks: list[Chunk],
    llm_model: str = "",
) -> tuple[list[Entity], list[Relation]]:
    """Extract entities and relations from chunks for graph ingest.

    Args:
        chunks: Chunks to process
        llm_model: Reserved compatibility parameter for future model-driven extraction

    Returns:
        Tuple of (entities, relations)
    """
    entities: list[Entity] = []
    relations: list[Relation] = []
    entity_map: dict[str, str] = {}  # name -> entity_id

    for chunk in chunks:
        chunk_entities = _simple_entity_extraction(chunk.content)

        for name, entity_type in chunk_entities:
            if name not in entity_map:
                entity_id = str(uuid.uuid4())
                entity_map[name] = entity_id
                entities.append(
                    Entity(
                        entity_id=entity_id,
                        name=name,
                        entity_type=entity_type,
                        metadata={"source_chunk": chunk.chunk_id},
                    )
                )

        entity_list = list(chunk_entities)
        for i, (name1, _) in enumerate(entity_list):
            for name2, _ in entity_list[i + 1 :]:
                relations.append(
                    Relation(
                        rel_id=str(uuid.uuid4()),
                        source_id=entity_map[name1],
                        target_id=entity_map[name2],
                        rel_type="co_occurs",
                        weight=1.0,
                        metadata={"source_chunk": chunk.chunk_id},
                    )
                )

    return entities, relations


def _simple_entity_extraction(text: str) -> list[tuple[str, str]]:
    """Simple rule-based entity extraction."""
    import re

    entities: list[tuple[str, str]] = []

    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
    matches = re.findall(pattern, text)

    for match in matches:
        if any(kw in match.lower() for kw in ["company", "inc", "corp", "ltd"]):
            entity_type = "organization"
        elif any(
            kw in match.lower()
            for kw in [
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ]
        ):
            entity_type = "date"
        else:
            entity_type = "concept"

        if len(match) > 3:
            entities.append((match, entity_type))

    seen = set()
    unique_entities = []
    for name, etype in entities:
        if name not in seen:
            seen.add(name)
            unique_entities.append((name, etype))

    return unique_entities[:50]
