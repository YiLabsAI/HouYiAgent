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
    r"^(?:who|what|where|when|whose|which|why|how)\b\s+"
    r"(?:is|are|was|were|did|do|does|has|have|will|would|can|could)?\s*"
    r"(?P<entity>[^?]+)",
    re.IGNORECASE,
)

# Words that look like entities to dumb regexes but never refer to a real
# subject in the entity-state view. Any inferred or caller-provided entity
# that collapses to one of these is dropped so the retriever returns []
# instead of running a doomed lookup. Comparison is case-insensitive.
_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "when",
        "where",
        "what",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "which",
    }
)

# Auxiliary verbs and common active verbs that partition proper nouns from predicates.
_VERB_BREAKS: frozenset[str] = frozenset(
    {
        "go",
        "goes",
        "went",
        "gone",
        "going",
        "lose",
        "loses",
        "lost",
        "losing",
        "have",
        "has",
        "had",
        "having",
        "join",
        "joins",
        "joined",
        "joining",
        "do",
        "does",
        "did",
        "done",
        "doing",
        "visit",
        "visits",
        "visited",
        "visiting",
        "meet",
        "meets",
        "met",
        "meeting",
        "start",
        "starts",
        "started",
        "starting",
        "get",
        "gets",
        "got",
        "getting",
        "buy",
        "buys",
        "bought",
        "buying",
        "sell",
        "sells",
        "sold",
        "selling",
        "work",
        "works",
        "worked",
        "working",
        "live",
        "lives",
        "lived",
        "living",
        "move",
        "moves",
        "moved",
        "moving",
        "become",
        "becomes",
        "became",
        "becoming",
    }
)


def _clean_entity(entity: str) -> str:
    """Strip action verbs and trailing predicate content from captured entity."""
    match = re.match(r"^([^\'\s]+)'s\b", entity, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    words = entity.split()
    cleaned = []
    for w in words:
        w_low = w.lower().strip(".,!?:;")
        if w_low in _VERB_BREAKS:
            break
        cleaned.append(w)
    return " ".join(cleaned)


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

        query_words = _extract_query_words(query.text)
        is_cumulative = bool(query_words.intersection(_CORE_WORDS)) or (
            hint.attribute is not None
            and any(
                _stemish(w) in _CORE_WORDS
                for w in re.sub(r"[^a-z0-9\s]", " ", hint.attribute.lower()).split()
            )
        )

        if is_cumulative:
            rows = await asyncio.to_thread(
                self._view.get_history,
                query.namespace,
                hint.entity,
                hint.attribute,
            )
        else:
            rows = await asyncio.to_thread(
                self._view.get_active,
                query.namespace,
                hint.entity,
                hint.attribute,
            )
        return [_candidate_from_row(row, self.name, hint, query_words) for row in rows]


_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "on",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "what",
    "are",
    "is",
    "was",
    "were",
    "did",
    "do",
    "does",
    "has",
    "have",
    "had",
    "his",
    "hi",
    "her",
    "my",
    "your",
    "its",
    "their",
    "our",
    "me",
    "you",
    "him",
    "them",
    "us",
    "i",
    "we",
    "he",
    "she",
    "it",
    "they",
}

_CORE_WORDS = {
    "goal",
    "plan",
    "problem",
    "issue",
    "health",
    "disease",
    "like",
    "dislike",
    "pref",
    "value",
    "dream",
    "wish",
}


def _stemish(token: str) -> str:
    t = token.strip().lower()
    if len(t) > 4 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 3 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _extract_query_words(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {
        _stemish(w) for w in normalized.split() if len(w) >= 2 and _stemish(w) not in _STOPWORDS
    }


def _is_question_word(token: str) -> bool:
    """Return True when the cleaned token is a wh-question word."""
    return token.strip().lower() in _QUESTION_WORDS


def _infer_entity_attribute(query: RecallQuery) -> EntityAttributeHint | None:
    """Infer a lookup target using hints first, then simple patterns.

    Any candidate entity that resolves to a wh-question word (When, Where,
    What, ...) is rejected because such tokens never refer to a real
    subject and would silently steer entity-state lookups into empty
    results. This applies to caller-provided hints as well, so upstream
    code does not need its own filter.
    """
    if query.entity_hint:
        entity = query.entity_hint.strip()
        if not entity or _is_question_word(entity):
            return None
        return EntityAttributeHint(
            entity=entity,
            attribute=query.attribute_hint.strip() if query.attribute_hint else None,
            source="caller_hint",
        )

    text = query.text.strip()
    zh = _ZH_ATTR_RE.search(text)
    if zh:
        entity = zh.group("entity").strip()
        if entity and not _is_question_word(entity):
            return EntityAttributeHint(
                entity=entity,
                attribute=zh.group("attribute").strip(),
                source="zh_attribute",
            )

    en_attr = _EN_ATTR_OF_RE.search(text)
    if en_attr:
        entity = en_attr.group("entity").strip()
        if entity and not _is_question_word(entity):
            return EntityAttributeHint(
                entity=entity,
                attribute=en_attr.group("attribute").lower().strip(),
                source="en_attribute_of",
            )

    en_wh = _EN_WH_RE.search(text)
    if en_wh:
        entity = _clean_entity(en_wh.group("entity").strip())
        # Defensive: even after the regex stripped the leading wh-word and
        # auxiliary, the capture can still start with another question
        # word (rare, but seen in chained queries like "what when X"). A
        # single-token entity that itself is a question word is dropped.
        if entity and not _is_question_word(entity):
            return EntityAttributeHint(entity=entity, source="en_wh_question")

    return None


def _candidate_from_row(
    row: EntityStateRecord,
    retriever_name: str,
    hint: EntityAttributeHint,
    query_words: set[str] | None = None,
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
    if query_words:
        fact_text = f"{row.attribute} {row.value}".lower()
        fact_words = {_stemish(w) for w in re.sub(r"[^a-z0-9\s]", " ", fact_text).split()}
        overlap = query_words.intersection(fact_words)
        for w in overlap:
            weight = 3.0 if w in _CORE_WORDS else 1.0
            score += weight * 1.5
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
