"""Enumeration-aware recall boosting.

Aggregation questions ("What books has X read?", "How many tournaments
has Y entered?") need *family coverage*: every fact belonging to the
asked category should outrank generic facts about the same subject.
Plain lexical scoring misses family members whose surface text does not
contain the category word (e.g. "Tim likes The Hobbit" for a books
question), so they lose fusion budget slots to unrelated same-score
facts and never reach the reasoner.

The booster closes that gap deterministically, with no category list
and no per-question rules — the category comes from the query text and
the family closure comes from the corpus itself:

1. Detect enumeration intent and extract the category head noun
   ("books" -> "book", "video game tournaments" -> "tournament").
2. Anchor pass: candidates whose predicate/object mention the category
   stem are family anchors. An FTS probe over the backend widens the
   anchor text pool beyond what the retrievers happened to return.
3. Closure pass: a candidate whose object value appears inside any
   anchor text joins the family (entity-category co-occurrence: "The
   Hobbit" is a book because the corpus says someone "reads book The
   Hobbit").
4. Family members receive a flat score boost before fusion so they win
   budget slots; ranking among non-family candidates is untouched.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import Any

from houyi.adapters.memory.recall.types import RecallCandidate

logger = logging.getLogger(__name__)

# Verb/aux alternatives shared by the intent patterns below.
_AUX = r"(?:has|have|had|did|does|do|is|are|was|were|will|would|can|could)"

# Structural enumeration/aggregation intents. Ordered specific-first;
# the first match wins. Each pattern captures the noun phrase whose
# head noun names the category being enumerated.
_ENUM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\bhow\s+many\s+(?P<np>[\w' -]+?)\s+{_AUX}\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:what|which)\s+(?:kinds?|sorts?|types?)\s+of\s+(?P<np>[\w' -]+?)\s+{_AUX}\b",
        re.IGNORECASE,
    ),
    re.compile(rf"\b(?:what|which)\s+(?P<np>[\w' -]+?s)\s+{_AUX}\b", re.IGNORECASE),
    re.compile(
        r"\blist\s+(?:all\s+)?(?:the\s+)?(?P<np>[\w' -]+?)(?:\s+of\b|[?.!]|$)", re.IGNORECASE
    ),
)

# Head nouns too generic to define a useful category. Structural
# filler words only — never benchmark answer data.
_GENERIC_HEADS: frozenset[str] = frozenset(
    {"thing", "one", "time", "way", "kind", "sort", "type", "people", "person"}
)

_MIN_OBJECT_CHARS = 4
"""Closure guard: objects shorter than this are too ambiguous to match."""

# Capitalized tokens that are sentence mechanics rather than entity
# names; used when extracting subject entities from the query text.
_NON_ENTITY_CAPS: frozenset[str] = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "the",
        "a",
        "an",
        "list",
        "many",
        "has",
        "have",
        "had",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
    }
)


def _query_entities(text: str) -> set[str]:
    """Casefolded proper-noun candidates from the query text."""
    return {
        w.casefold()
        for w in re.findall(r"[A-Z][\w']+", text or "")
        if w.casefold() not in _NON_ENTITY_CAPS
    }


def _stem(word: str) -> str:
    """Light plural/inflection stemmer for category head nouns.

    Both plural ("movies", "activities") and singular ("movie",
    "activity") forms collapse onto the same canonical token; the
    canonical form is only used for matching, never displayed.
    """
    w = word.strip().lower()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-1]
    if len(w) > 2 and w.endswith("y"):
        return w[:-1] + "ie"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {_stem(w) for w in normalized.split() if w}


def detect_enumeration_category(text: str) -> str | None:
    """Return the stemmed category head noun for enumeration queries.

    Returns None when the query carries no enumeration intent or the
    head noun is too generic to anchor a family.
    """
    for pattern in _ENUM_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        phrase = match.group("np").strip()
        if not phrase:
            continue
        head = _stem(phrase.split()[-1])
        if len(head) >= 3 and head not in _GENERIC_HEADS:
            return head
    return None


class EnumerationBooster:
    """Boost category-family candidates ahead of fusion budget cuts."""

    def __init__(
        self,
        backend: Any | None = None,
        *,
        instance_boost: float = 8.0,
        mention_boost: float = 2.0,
        fts_limit: int = 50,
    ) -> None:
        self._backend = backend
        self._instance_boost = instance_boost
        self._mention_boost = mention_boost
        self._fts_limit = fts_limit

    async def apply(
        self,
        query_text: str,
        candidates: Sequence[RecallCandidate],
    ) -> int:
        """Boost family members in place; return how many were boosted.

        Two tiers keep the enumerable items ahead of category chatter:

        - instance: the fact names a concrete family member — its
          predicate carries the category stem while the object is the
          member value ("reads_book -> The Hobbit"), or its object
          matches a known member value via closure ("likes -> The
          Hobbit"). These are the answers to enumerate.
        - mention: the fact merely talks about the category ("likes
          fantasy books", "book sitting on a table"). Useful context,
          but it must not crowd instances out of the budget.

        When the query names subject entities ("... has Tim read"),
        instance rank is reserved for facts about those subjects;
        other subjects' family facts drop to mention rank so they
        cannot crowd the asked-about subject out of the budget.
        """
        category = detect_enumeration_category(query_text)
        if category is None or not candidates:
            return 0

        subjects = _query_entities(query_text)
        instances, mentions = await self._classify(category, subjects, candidates)

        for cand in instances:
            self._mark(cand, category, "instance", self._instance_boost)
        for cand in mentions:
            self._mark(cand, category, "mention", self._mention_boost)
        total = len(instances) + len(mentions)
        if total:
            logger.info(
                "enumeration booster: category=%r instances=%d mentions=%d of %d candidates",
                category,
                len(instances),
                len(mentions),
                len(candidates),
            )
        return total

    @staticmethod
    def _mark(cand: RecallCandidate, category: str, tier: str, boost: float) -> None:
        cand.score += boost
        cand.signals = dict(cand.signals)
        cand.signals["enumeration_family"] = category
        cand.signals["enumeration_tier"] = tier

    async def _classify(
        self,
        category: str,
        subjects: set[str],
        candidates: Sequence[RecallCandidate],
    ) -> tuple[list[RecallCandidate], list[RecallCandidate]]:
        """Split candidates into (instances, mentions) family tiers."""

        def _is_query_subject(cand: RecallCandidate) -> bool:
            return not subjects or cand.fact.subject.strip().casefold() in subjects

        anchor_texts: list[str] = []
        instances: list[RecallCandidate] = []
        mentions: list[RecallCandidate] = []
        rest: list[RecallCandidate] = []
        for cand in candidates:
            pred_words = _words(cand.fact.predicate)
            obj_words = _words(cand.fact.object)
            if category not in pred_words and category not in obj_words:
                rest.append(cand)
                continue
            anchor_texts.append(f"{cand.fact.predicate} {cand.fact.object}".lower())
            if category in pred_words and category not in obj_words and _is_query_subject(cand):
                instances.append(cand)
            else:
                mentions.append(cand)

        anchor_texts.extend(await self._fts_anchor_texts(category))

        for cand in rest:
            obj = (cand.fact.object or "").strip().lower()
            if len(obj) < _MIN_OBJECT_CHARS:
                continue
            if any(obj in anchor for anchor in anchor_texts):
                (instances if _is_query_subject(cand) else mentions).append(cand)
        return instances, mentions

    async def _fts_anchor_texts(self, category: str) -> list[str]:
        """Probe backend FTS for corpus texts mentioning the category."""
        backend = self._backend
        search = getattr(backend, "search_fts", None)
        if search is None:
            return []
        try:
            hits = await asyncio.to_thread(search, category, limit=self._fts_limit)
        except Exception as exc:
            logger.debug("enumeration FTS probe skipped: %s", exc)
            return []
        texts: list[str] = []
        for hit in hits:
            record = hit[0] if isinstance(hit, tuple) else hit
            content = getattr(record, "content", "") or ""
            if category in _words(content):
                texts.append(content.lower())
        return texts


__all__ = ["EnumerationBooster", "detect_enumeration_category"]
