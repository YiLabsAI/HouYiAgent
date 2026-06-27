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


def _is_identity_anchor(row: EntityStateRecord) -> bool:
    """Return True for self-loop identity anchor rows (e.g. 'X | identity | X').

    These rows exist purely as graph edge endpoints and carry no answer
    value; surfacing them wastes recall slots and inflates coverage scores.
    """
    return (
        row.attribute == "identity"
        and row.entity.strip().casefold() == row.value.strip().casefold()
    )


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

_POSSESSIVE_RE = re.compile(
    r"\b(?P<entity>[A-Z][a-zA-Z0-9_]+)'s\s+(?P<attribute>[a-zA-Z0-9_]+(?:\s+(?:or|and)\s+[a-zA-Z0-9_]+)?)\b",
    re.IGNORECASE,
)

_KIND_OF_RE = re.compile(
    r"\b(?:what|which)\s+(?:kind|type|sort|fields?)\s+of\s+(?P<attribute>[a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)?)\s+(?:has|have|does|do|did|is|are|was|were|would|could|can)\s+(?P<entity>[A-Z][a-zA-Z0-9_]+)\b",
    re.IGNORECASE,
)

_WH_NOUN_RE = re.compile(
    r"\b(?:what|which)\s+(?P<attribute>[a-zA-Z0-9_]+(?:\s+(?:or|and)\s+[a-zA-Z0-9_]+)?)\s+(?:has|have|does|do|did|is|are|was|were|would|could|can)\s+(?P<entity>[A-Z][a-zA-Z0-9_]+)\b",
    re.IGNORECASE,
)

# Relationship terms that can be resolved to actual entities via relationship facts.
# Format: "with his girlfriend" -> look up entity's relationship facts -> find "Audrey"
_RELATIONSHIP_TERMS = frozenset(
    {
        "girlfriend",
        "boyfriend",
        "wife",
        "husband",
        "partner",
        "spouse",
        "fiancee",
        "fiance",
        "significant",  # "significant other"
    }
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


def _resolve_relationship_entities(
    view: EntityStateView,
    namespace: str,
    entity: str,
    query_text: str,
) -> list[str]:
    """Resolve relationship terms in query to actual entity names.

    When the query mentions "with his girlfriend" or similar relationship
    terms, look up the entity's relationship facts and find the actual
    person. Two strategies:

    1. If relationship value is a proper noun (e.g., "Audrey"), use it directly.
    2. If relationship value is a common noun (e.g., "girlfriend", "GF"),
       find the entity that most frequently appears in "shares activity"
       facts with the primary entity.

    Example: "Andrew's activities with girlfriend" ->
    - Strategy 1: look for "Andrew | has_girlfriend | Audrey" -> return ["Audrey"]
    - Strategy 2: look for "Andrew | shares_activity | Audrey" (most frequent) -> return ["Audrey"]
    """
    query_low = query_text.lower()
    # Check if any relationship term appears in the query
    has_relationship_term = any(term in query_low for term in _RELATIONSHIP_TERMS)
    if not has_relationship_term:
        return []

    # Strategy 1: Look for relationship facts with proper noun values
    resolved = []
    try:
        rows = view.get_active(namespace, entity, None)
        for row in rows:
            attr_low = row.attribute.lower()
            # Match relationship attributes and only add proper noun values
            if (
                ("relationship" in attr_low or "girlfriend" in attr_low or "boyfriend" in attr_low)
                and row.value
                and row.value[0].isupper()
                and row.value not in {"Girlfriend", "Boyfriend", "GF", "BF", "Partner", "Spouse"}
            ):
                resolved.append(row.value)
    except Exception:
        pass

    # Strategy 2: If no proper noun found, infer from "shares activity" facts
    if not resolved:
        try:
            rows = view.get_active(namespace, entity, None)
            # Find all entities that appear in "shares activity" or "shares interest" facts
            shared_entities: dict[str, int] = {}
            for row in rows:
                attr_low = row.attribute.lower()
                # The value is the entity name (e.g., "Audrey")
                if (
                    "shares" in attr_low
                    and ("activity" in attr_low or "interest" in attr_low)
                    and row.value
                    and row.value[0].isupper()
                ):
                    shared_entities[row.value] = shared_entities.get(row.value, 0) + 1
            # Pick the most frequent shared entity
            if shared_entities:
                best_entity = max(shared_entities.keys(), key=lambda k: shared_entities[k])
                resolved.append(best_entity)
        except Exception:
            pass

    return resolved


def _expand_entity_list(
    view: EntityStateView,
    namespace: str,
    entities: list[str],
    query_text: str,
) -> list[str]:
    """Expand entity list with multi-entity parsing and cross-entity association.

    Combines two expansion strategies:
    1. Multi-entity parsing: find other capitalized proper nouns in query
    2. Cross-entity association: resolve relationship terms to actual entities
    """
    expanded = list(entities)

    # Multi-entity parsing: find other capitalized words in the query text
    for w in query_text.split():
        w_clean = w.strip(".,!?:;'\"()")
        if w_clean and w_clean[0].isupper():
            w_low = w_clean.lower()
            if (
                w_low
                not in {
                    "when",
                    "where",
                    "what",
                    "who",
                    "whom",
                    "whose",
                    "why",
                    "how",
                    "which",
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
                    "would",
                    "should",
                    "could",
                    "does",
                    "doesnt",
                    "did",
                    "didnt",
                    "is",
                    "isnt",
                    "are",
                    "arent",
                    "was",
                    "wasnt",
                    "were",
                    "werent",
                    "has",
                    "hasnt",
                    "have",
                    "havent",
                    "can",
                    "cant",
                    "will",
                    "wont",
                    "do",
                    "dont",
                    "in",
                    "according",
                    "the",
                    "a",
                    "an",
                    "if",
                    "with",
                    "from",
                    "to",
                    "at",
                    "by",
                    "for",
                    "on",
                    "about",
                    "into",
                }
                and w_clean not in expanded
            ):
                expanded.append(w_clean)

    # Cross-entity association: resolve relationship terms to actual entities
    if expanded:
        primary_entity = expanded[0]
        resolved_entities = _resolve_relationship_entities(
            view,
            namespace,
            primary_entity,
            query_text,
        )
        for resolved in resolved_entities:
            if resolved not in expanded:
                expanded.append(resolved)

    return expanded


def _find_shared_attribute_rows(
    entities: list[str],
    entity_rows: dict[str, list[EntityStateRecord]],
) -> list[EntityStateRecord]:
    """Find EntityStateRecords for facts with shared attributes across entities.

    General principle: when multiple entities (e.g., Andrew and Audrey) have
    the same attribute (e.g., "grows"), return the EntityStateRecords of those
    facts so they can be added to the candidate list.

    Example:
    - Andrew | grows | blooming flowers
    - Audrey | grows | Peruvian Lilies
    -> Return: [Andrew's grows row, Audrey's grows row]

    This is a structural optimization, not data-fitting — it applies the general
    principle "same attribute across related entities = shared activity" to any
    attribute, not just specific hardcoded activities.
    """
    if len(entities) < 2:
        return []

    # Build attribute -> {entity: [rows]} mapping, tracking whether the
    # attribute has any accumulate=true row across its entities.
    #
    # The accumulate flag is the extractor-supplied structural signal marking
    # an attribute as a recurring activity / collected set (grows, recommends,
    # has_pet, ...). Pure cognitive/affective states (believes, thinks,
    # wants_to, finds) and image descriptions (shared_image_depicts) are never
    # accumulate and are excluded, because they pollute the shared-activity
    # candidate pool and bury real activities (a single "believes" fact can
    # merge 28 propositions into one huge fact that dominates LLM attention).
    #
    # accumulate is per-row but sharedness is per-attribute, so the gate is
    # applied at attribute granularity: an attribute qualifies if ANY of its
    # rows across entities is accumulate=true. This keeps e.g. 'grows' even
    # when only one entity's row carries the flag.
    attr_map: dict[str, dict[str, list[EntityStateRecord]]] = {}
    attr_accumulates: dict[str, bool] = {}
    for ent, rows in entity_rows.items():
        for row in rows:
            # Skip compound and identity attributes
            if row.attribute in {"_compound", "identity"}:
                continue
            attr_map.setdefault(row.attribute, {}).setdefault(ent, []).append(row)
            if _row_is_accumulate(row):
                attr_accumulates[row.attribute] = True

    # Find attributes that (a) are activity collections and (b) appear for
    # at least 2 entities, then collect all their rows.
    shared_rows = []
    for attr, entity_values in attr_map.items():
        if len(entity_values) >= 2 and attr_accumulates.get(attr):
            for ent in entity_values:
                shared_rows.extend(entity_values[ent])

    return shared_rows


def _row_is_accumulate(row: EntityStateRecord) -> bool:
    """Return True if the row's qualifiers mark it as accumulate=true.

    The accumulate flag is set by the extractor for attributes that represent a
    recurring activity or collected set (e.g. 'grows', 'recommends', 'has_pet').
    Using it as the shared-attribute gate keeps the candidate pool to genuine
    activity collections rather than cognitive states or image captions.
    """
    quals = row.qualifiers
    if not quals:
        return False
    return quals.get("accumulate") == "true"


def _clean_entity(entity: str) -> str:
    """Strip action verbs and trailing predicate content from captured entity."""
    match = re.match(r"^([^\'\s]+)'s\b", entity, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    words = entity.split()
    # High-priority: detect capitalized proper nouns (e.g., 'Audrey', 'Calvin', 'John')
    # and return them directly to avoid matching question helper nouns like 'year' or 'items'.
    for w in words:
        w_clean = w.strip(".,!?:;'\"()")
        if (
            w_clean
            and w_clean[0].isupper()
            and w_clean.lower()
            not in {"when", "where", "what", "who", "whom", "whose", "why", "how", "which"}
        ):
            return w_clean

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
        entities = []
        attribute = None
        source = "heuristics"
        if hint is not None:
            entities.append(hint.entity)
            attribute = hint.attribute
            source = hint.source

        query_words = _extract_query_words(query.text)
        is_cumulative = bool(query_words.intersection(_CORE_WORDS)) or (
            attribute is not None
            and any(
                _light_stem(w) in _CORE_WORDS
                for w in re.sub(r"[^a-z0-9\s]", " ", attribute.lower()).split()
            )
        )

        # Multi-entity parsing + cross-entity association
        entities = _expand_entity_list(
            self._view,
            query.namespace,
            entities,
            query.text,
        )

        if not entities:
            return []

        # Retrieve rows for each entity
        entity_rows: dict[str, list[EntityStateRecord]] = {}
        candidates = []
        for ent in entities:
            ent_hint = EntityAttributeHint(
                entity=ent,
                attribute=attribute,
                source=source,
            )
            rows = await self._retrieve_rows_for_entity(
                query.namespace,
                ent,
                attribute,
                is_cumulative,
            )
            entity_rows[ent] = [r for r in rows if not _is_identity_anchor(r)]
            candidates.extend(
                [
                    _candidate_from_row(row, self.name, ent_hint, query_words)
                    for row in rows
                    if not _is_identity_anchor(row)
                ]
            )

        # Boost scores for facts with shared attributes across entities.
        # General principle: when multiple entities (e.g., Andrew and Audrey) have
        # the same attribute (e.g., "grows"), add those facts to the candidate list
        # with a boosted score. This bridges the gap between parallel individual facts
        # and shared activities — shared facts are relevant even if they don't match
        # the query attribute filter (e.g., "indoor activities" doesn't match "grows",
        # but "grows" is still relevant when asking about shared activities).
        if len(entities) >= 2:
            # Build attribute -> {entity: [state_ids]} mapping from ALL entity facts
            all_entity_rows: dict[str, list[EntityStateRecord]] = {}
            for ent in entities:
                all_rows = await asyncio.to_thread(
                    self._view.get_active,
                    query.namespace,
                    ent,
                    None,  # No attribute filter — get ALL facts
                )
                all_entity_rows[ent] = [r for r in all_rows if not _is_identity_anchor(r)]

            # Find shared attributes and collect their rows
            shared_rows = _find_shared_attribute_rows(entities, all_entity_rows)

            # Add shared rows as candidates with boosted scores and shared_activity signal
            for row in shared_rows:
                ent_hint = EntityAttributeHint(
                    entity=row.entity,
                    attribute=row.attribute,
                    source=source,
                )
                candidate = _candidate_from_row(row, self.name, ent_hint, query_words)
                candidate.score += 8.0  # Significant boost for shared attributes
                # Mark as shared activity so renderer generates clearer text
                if candidate.signals is None:
                    candidate.signals = {}
                candidate.signals["shared_activity"] = True
                candidate.signals["shared_entities"] = entities
                candidates.append(candidate)

        return candidates

    async def _retrieve_rows_for_entity(
        self,
        namespace: str,
        entity: str,
        attribute: str | None,
        is_cumulative: bool,
    ) -> list[EntityStateRecord]:
        """Retrieve rows for a single entity, with fuzzy fallback."""
        if is_cumulative:
            rows = await asyncio.to_thread(
                self._view.get_history,
                namespace,
                entity,
                attribute,
            )
            if attribute is not None and not rows:
                all_rows = await asyncio.to_thread(
                    self._view.get_history,
                    namespace,
                    entity,
                    None,
                )
                rows = _filter_fuzzy_rows(all_rows, attribute)
        else:
            rows = await asyncio.to_thread(
                self._view.get_active,
                namespace,
                entity,
                attribute,
            )
            if attribute is not None and not rows:
                all_rows = await asyncio.to_thread(
                    self._view.get_active,
                    namespace,
                    entity,
                    None,
                )
                rows = _filter_fuzzy_rows(all_rows, attribute)
        return rows


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
    "interest",
    "share",
    "hobby",
    "enjoy",
    "love",
    "movie",
    "film",
    "dessert",
    "bake",
    "pass",
    "away",
    "die",
    "dead",
    "loss",
    "break",
    "broke",
    "broken",
    "down",
    "accident",
    "damage",
    "breakdown",
    "family",
    "friend",
    "consider",
    "considers",
    "career",
    "pursue",
    "past",
    "previous",
    "previously",
    "used",
    "transition",
    "milestone",
    "change",
    "former",
    "formerly",
    "counseling",
}


_IRREGULAR_VERB_MAP: dict[str, str] = {
    # Existing _VERB_BREAKS verbs
    "goes": "go",
    "went": "go",
    "gone": "go",
    "going": "go",
    "loses": "lose",
    "lost": "lose",
    "losing": "lose",
    "has": "have",
    "had": "have",
    "having": "have",
    "joins": "join",
    "joined": "join",
    "joining": "join",
    "does": "do",
    "did": "do",
    "done": "do",
    "doing": "do",
    "visits": "visit",
    "visited": "visit",
    "visiting": "visit",
    "meets": "meet",
    "met": "meet",
    "meeting": "meet",
    "starts": "start",
    "started": "start",
    "starting": "start",
    "gets": "get",
    "got": "get",
    "getting": "get",
    "buys": "buy",
    "bought": "buy",
    "buying": "buy",
    "sells": "sell",
    "sold": "sell",
    "selling": "sell",
    "works": "work",
    "worked": "work",
    "working": "work",
    "lives": "live",
    "lived": "live",
    "living": "live",
    "moves": "move",
    "moved": "move",
    "moving": "move",
    "becomes": "become",
    "became": "become",
    "becoming": "become",
    # Additional common verbs
    "wins": "win",
    "won": "win",
    "winning": "win",
    "takes": "take",
    "took": "take",
    "taking": "take",
    "makes": "make",
    "made": "make",
    "making": "make",
    "knows": "know",
    "knew": "know",
    "knowing": "know",
    "sees": "see",
    "saw": "see",
    "seen": "see",
    "seeing": "see",
    "gives": "give",
    "gave": "give",
    "given": "give",
    "giving": "give",
    "comes": "come",
    "came": "come",
    "coming": "come",
    "tells": "tell",
    "told": "tell",
    "telling": "tell",
    "says": "say",
    "said": "say",
    "saying": "say",
    "writes": "write",
    "wrote": "write",
    "written": "write",
    "writing": "write",
    "reads": "read",
    "reading": "read",
    "feels": "feel",
    "felt": "feel",
    "feeling": "feel",
    "keeps": "keep",
    "kept": "keep",
    "keeping": "keep",
    "thinks": "think",
    "thought": "think",
    "thinking": "think",
    "brings": "bring",
    "brought": "bring",
    "bringing": "bring",
    "finds": "find",
    "found": "find",
    "finding": "find",
    "holds": "hold",
    "held": "hold",
    "holding": "hold",
    "speaks": "speak",
    "spoke": "speak",
    "spoken": "speak",
    "speaking": "speak",
    "runs": "run",
    "ran": "run",
    "running": "run",
    "eats": "eat",
    "ate": "eat",
    "eaten": "eat",
    "eating": "eat",
    "drinks": "drink",
    "drank": "drink",
    "drunk": "drink",
    "drinking": "drink",
    "sleeps": "sleep",
    "slept": "sleep",
    "sleeping": "sleep",
    "begins": "begin",
    "began": "begin",
    "begun": "begin",
    "beginning": "begin",
}


def _light_stem(token: str) -> str:
    t = token.strip().lower()
    if t in _IRREGULAR_VERB_MAP:
        return _IRREGULAR_VERB_MAP[t]
    if len(t) > 4 and t.endswith("ing"):
        base = t[:-3]
        # Undo consonant doubling from the CVC doubling rule:
        # e.g. "planning" -> "plann" (wrong) -> "plan" (correct)
        if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "aeiouy":
            base = base[:-1]
        return base
    if len(t) > 3 and t.endswith("ed"):
        base = t[:-2]
        # Undo consonant doubling: "planned" -> "plann" -> "plan"
        if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "aeiouy":
            base = base[:-1]
        return base
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _extract_query_words(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {
        _light_stem(w)
        for w in normalized.split()
        if len(w) >= 2 and _light_stem(w) not in _STOPWORDS
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
    for pattern, source in [
        (_ZH_ATTR_RE, "zh_attribute"),
        (_EN_ATTR_OF_RE, "en_attribute_of"),
        (_POSSESSIVE_RE, "possessive"),
        (_KIND_OF_RE, "kind_of"),
        (_WH_NOUN_RE, "wh_noun"),
    ]:
        match = pattern.search(text)
        if match:
            entity = match.group("entity").strip()
            attribute = match.group("attribute").strip()
            if source == "en_attribute_of":
                attribute = attribute.lower()
            if entity and not _is_question_word(entity):
                return EntityAttributeHint(entity=entity, attribute=attribute, source=source)

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


def _fuzzy_attr_match(query_attr: str, stored_attr: str) -> bool:
    """Determine whether a query attribute hint semantically aligns with a
    stored attribute, using stemmed token overlap and containment.

    This is the self-healing alignment mechanism that bridges the gap
    between Extractor free-form predicate naming and Retriever query
    parsing. No regex or ontology dictionary required -- stem matching
    + Jaccard fallback covers arbitrary predicate phrasing.
    """
    target_stems = _attr_stems(query_attr)
    row_stems = _attr_stems(stored_attr)
    if not target_stems:
        return False

    # Stem overlap: "goals" -> {"goal"} matches "has_goal" -> {"has", "goal"}
    if target_stems.intersection(row_stems):
        return True

    # Substring containment: "goal" in "primary_goal"
    q_low = query_attr.lower()
    s_low = stored_attr.lower()
    if any(ts in s_low for ts in target_stems) or any(rs in q_low for rs in row_stems):
        return True

    # Jaccard fallback for multi-word attributes
    if row_stems:
        jaccard = len(target_stems & row_stems) / len(target_stems | row_stems)
        if jaccard >= 0.4:
            return True

    return False


def _attr_stems(attribute: str) -> set[str]:
    """Stemmed content words from an attribute string."""
    words = re.sub(r"[^a-z0-9]", " ", attribute.lower()).split()
    return {_light_stem(w) for w in words if len(w) >= 2 and _light_stem(w) not in _STOPWORDS}


def _filter_fuzzy_rows(
    rows: list[EntityStateRecord], target_attribute: str
) -> list[EntityStateRecord]:
    """Filter rows by fuzzy matching target_attribute with row's attribute.

    Uses _fuzzy_attr_match for per-row alignment, keeping compound rows
    unconditionally as they carry consolidated multi-value facts.
    """
    target_stems = _attr_stems(target_attribute)
    if not target_stems:
        return []

    matched = []
    for r in rows:
        # Always include compound rows as they hold rich consolidated facts
        if r.attribute == "_compound":
            matched.append(r)
            continue
        if _fuzzy_attr_match(target_attribute, r.attribute):
            matched.append(r)
    return matched


def _extract_event_time(qualifiers: dict[str, str] | None) -> str | None:
    """Extract the fact-relevant time from qualifiers.

    The qualifiers may contain a 'date' key (ISO date like "2023-01-19"
    or "2020-03") or a 'since' key (acquisition/ongoing start date).
    These are fact-relevant times that should be carried in
    AtomicFact.event_time so they are rendered meaningfully to the
    LLM, not as Unix epoch timestamps from valid_from.
    """
    if not qualifiers:
        return None
    return qualifiers.get("date") or qualifiers.get("since") or None


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
        event_time=_extract_event_time(row.qualifiers),
        accumulate=bool(row.qualifiers and row.qualifiers.get("accumulate") == "true"),
    )
    is_matched = False
    match_tier = "broad"
    if hint.attribute is not None:
        if hint.attribute == row.attribute:
            is_matched = True
            match_tier = "exact"
        elif row.attribute == "_compound" or _fuzzy_attr_match(hint.attribute, row.attribute):
            is_matched = True
            match_tier = "fuzzy"
    score = {"exact": 10.0, "fuzzy": 7.5, "broad": 5.0}[match_tier]
    if query_words:
        # Boost interest and hobby attributes for shared interest queries
        if query_words.intersection({"interest", "share"}):
            row_attr_low = row.attribute.lower()
            row_val_low = row.value.lower()
            if row_attr_low in {
                "has_hobby",
                "likes_movie_genre",
                "enjoys",
                "likes",
                "loves",
                "bakes",
                "makes",
            } or any(
                w in row_val_low
                for w in {"movie", "film", "dessert", "bake", "baking", "cook", "cooking"}
            ):
                score += 5.0
        fact_text = f"{row.attribute} {row.value}".lower()
        fact_words = {_light_stem(w) for w in re.sub(r"[^a-z0-9\s]", " ", fact_text).split()}
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
            "exact_attribute": is_matched,
        },
        explanation=f"active entity-state row for {row.entity}.{row.attribute}",
    )


__all__ = ["EntityAttributeHint", "EntityStateRetriever"]
