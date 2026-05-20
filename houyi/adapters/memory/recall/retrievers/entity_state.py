"""Entity-state retriever for single-hop memory lookup.

This retriever serves current factual lookup and yes/no absence checks
from the materialized entity-state view. It intentionally keeps query
parsing conservative: caller-provided hints win; lightweight regex
heuristics are used only when hints are absent. Ambiguous thematic
queries return [] and let fallback retrievers handle them.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from houyi.adapters.memory.backends.base import EntityStateView
from houyi.adapters.memory.recall.retrievers.base import Retriever
from houyi.adapters.memory.recall.types import (
    RecallCandidate,
    RecallQuery,
    RetrieverContext,
    RetrieverKind,
)
from houyi.adapters.memory.types import AtomicFact, EntityStateRecord


@dataclass(frozen=True)
class EntityAttributeHint:
    """Best-effort parsed entity and optional attribute."""

    entity: str
    attribute: str | None = None
    source: str = "heuristic"


_ZH_ATTR_WORDS = (
    # CJK escape glossary: phone, email, address, birthday, occupation, name, age, identity, company.
    "\u7535\u8bdd",  # phone
    "\u90ae\u7bb1",  # email
    "\u5730\u5740",  # address
    "\u751f\u65e5",  # birthday
    "\u804c\u4e1a",  # occupation
    "\u540d\u5b57",  # name
    "\u5e74\u9f84",  # age
    "\u8eab\u4efd",  # identity
    "\u516c\u53f8",  # company
)

_EN_ATTR_WORDS = (
    "phone",
    "email",
    "address",
    "birthday",
    "job",
    "name",
    "age",
    "company",
)

_ZH_ATTR_RE = re.compile(
    r"(?P<entity>\S+)\s*\u7684\s*(?P<attribute>"
    + "|".join(re.escape(w) for w in _ZH_ATTR_WORDS)
    + r")"
)
_EN_ATTR_OF_RE = re.compile(
    r"(?P<attribute>"
    + "|".join(re.escape(w) for w in _EN_ATTR_WORDS)
    + r")\s+of\s+(?P<entity>\w+)",
    re.IGNORECASE,
)
_EN_WH_RE = re.compile(
    r"^(?:who|what|where|whose|which|why)\b\s+(?:is|are|was|were)?\s*(?P<entity>[^?]+)",
    re.IGNORECASE,
)


class EntityStateRetriever(Retriever):
    """Retrieve active rows from EntityStateView.

    The retriever is optimized for exact entity/attribute lookups. It
    does not scan all entities; if no entity can be inferred it returns
    [] so the orchestrator can fall back to broader strategies.
    """

    def __init__(self, view: EntityStateView) -> None:
        if view is None:
            raise ValueError("view is required")
        self._view = view

    async def retrieve(
        self,
        query: RecallQuery,
        ctx: RetrieverContext,
    ) -> list[RecallCandidate]:
        hint = _infer_entity_attribute(query)
        if hint is None:
            return []

        rows = await asyncio.to_thread(
            self._view.get_active,
            query.namespace,
            hint.entity,
            hint.attribute,
        )
        return [_candidate_from_row(row, self.name, hint) for row in rows]


def _infer_entity_attribute(query: RecallQuery) -> EntityAttributeHint | None:
    """Infer a lookup target using hints first, then simple patterns."""
    if query.entity_hint:
        return EntityAttributeHint(
            entity=query.entity_hint.strip(),
            attribute=query.attribute_hint.strip() if query.attribute_hint else None,
            source="caller_hint",
        )

    text = query.text.strip()
    zh = _ZH_ATTR_RE.search(text)
    if zh:
        return EntityAttributeHint(
            entity=zh.group("entity").strip(),
            attribute=zh.group("attribute").strip(),
            source="zh_attribute",
        )

    en_attr = _EN_ATTR_OF_RE.search(text)
    if en_attr:
        return EntityAttributeHint(
            entity=en_attr.group("entity").strip(),
            attribute=en_attr.group("attribute").lower().strip(),
            source="en_attribute_of",
        )

    en_wh = _EN_WH_RE.search(text)
    if en_wh:
        entity = en_wh.group("entity").strip()
        if entity:
            return EntityAttributeHint(entity=entity, source="en_wh_question")

    return None


def _candidate_from_row(
    row: EntityStateRecord,
    retriever_name: str,
    hint: EntityAttributeHint,
) -> RecallCandidate:
    """Convert an entity-state row back into an atomic recall candidate."""
    fact = AtomicFact(
        subject=row.entity,
        predicate=row.attribute,
        object=row.value,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        certainty=row.certainty,
        source_anchor=row.source_unit_id or row.state_id,
        qualifiers=row.qualifiers,
    )
    exact_attribute = hint.attribute is not None and hint.attribute == row.attribute
    score = 10.0 if exact_attribute else 5.0
    return RecallCandidate(
        fact=fact,
        score=score,
        matched_by=RetrieverKind.ENTITY_STATE,
        retriever_name=retriever_name,
        signals={
            "entity": row.entity,
            "attribute": row.attribute,
            "hint_source": hint.source,
            "exact_attribute": exact_attribute,
        },
        explanation=f"active entity-state row for {row.entity}.{row.attribute}",
    )


__all__ = ["EntityAttributeHint", "EntityStateRetriever"]
