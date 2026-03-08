"""Query Analyzer for adaptive strategy selection.

Analyzes user queries to determine optimal retrieval strategies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.adapters.llm.base import LLMAdapter

from houyi.rag.types import RetrievalStrategy

logger = logging.getLogger(__name__)


DEFAULT_RETRIEVAL_STRATEGIES = [
    RetrievalStrategy.BM25,
    RetrievalStrategy.VECTOR,
]


QUERY_TYPE_STRATEGY_MAP = {
    "factual": [RetrievalStrategy.BM25, RetrievalStrategy.VECTOR],
    "conceptual": [RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH],
    "procedural": [RetrievalStrategy.VECTOR, RetrievalStrategy.BM25],
    "analytical": [RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH],
    "keyword": [RetrievalStrategy.BM25],
    "exploratory": [
        RetrievalStrategy.VECTOR,
        RetrievalStrategy.BM25,
        RetrievalStrategy.GRAPH,
    ],
}


LLM_STRATEGY_MAP = {
    "bm25": RetrievalStrategy.BM25,
    "vector": RetrievalStrategy.VECTOR,
    "graph": RetrievalStrategy.GRAPH,
}


def _base_strategies_for_query_type(query_type: QueryType) -> list[RetrievalStrategy]:
    return list(QUERY_TYPE_STRATEGY_MAP.get(query_type.value, DEFAULT_RETRIEVAL_STRATEGIES))


def _parse_llm_strategies(strategy_names: list[str] | None) -> list[RetrievalStrategy]:
    names = strategy_names or ["bm25", "vector"]
    return [LLM_STRATEGY_MAP[name] for name in names if name in LLM_STRATEGY_MAP]


class QueryType(str, Enum):
    """Query type classification."""

    FACTUAL = "factual"  # Fact-based questions (who, what, when, where)
    CONCEPTUAL = "conceptual"  # Concept explanation, definition
    PROCEDURAL = "procedural"  # How-to, step-by-step
    ANALYTICAL = "analytical"  # Why, comparison, cause-effect
    KEYWORD = "keyword"  # Simple keyword search
    EXPLORATORY = "exploratory"  # Open-ended exploration


@dataclass
class QueryAnalysis:
    """Result of query analysis."""

    query: str
    query_type: QueryType
    keywords: list[str]
    entities: list[str]
    intent: str
    recommended_strategies: list[RetrievalStrategy]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for UI."""
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "keywords": self.keywords,
            "entities": self.entities,
            "intent": self.intent,
            "recommended_strategies": [s.value for s in self.recommended_strategies],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class QueryAnalyzer:
    """Analyzes queries to recommend optimal retrieval strategies.

    Strategy selection logic:
    - Factual queries → BM25 + Vector (exact keyword match important)
    - Conceptual queries → Vector + Graph (semantic understanding)
    - Procedural queries → Vector (step-by-step content)
    - Analytical queries → Vector + Graph (relationships matter)
    - Keyword queries → BM25 only (simple keyword match)
    - Exploratory queries → All strategies (broad search)
    """

    # Question word patterns for classification
    FACTUAL_PATTERNS = [
        r"^(who|what|when|where|which)\b",
        r"\b(name|list|identify|define)\b",
    ]

    CONCEPTUAL_PATTERNS = [
        r"^(what is|explain|describe|define)\b",
        r"\b(concept|meaning|definition)\b",
    ]

    PROCEDURAL_PATTERNS = [
        r"^(how to|how do|how can)\b",
        r"\b(steps?|process|procedure|guide)\b",
    ]

    ANALYTICAL_PATTERNS = [
        r"^(why|compare|contrast|analyze)\b",
        r"\b(difference|similar|cause|effect|impact)\b",
    ]

    def __init__(self, adapter: LLMAdapter | None = None) -> None:
        """Initialize query analyzer.

        Args:
            adapter: Optional LLM adapter for advanced analysis
        """
        self._adapter = adapter

    async def analyze(self, query: str, **kwargs: Any) -> QueryAnalysis:
        """Analyze a query and recommend strategies.

        Args:
            query: User query string
            **kwargs: Additional options

        Returns:
            QueryAnalysis with recommendations
        """
        # Try LLM analysis if available
        if self._adapter:
            try:
                return await self._analyze_with_llm(query)
            except Exception as e:
                logger.warning("LLM query analysis failed: %s, using heuristic", e)

        # Fall back to heuristic analysis
        return self._analyze_heuristic(query)

    def _analyze_heuristic(self, query: str) -> QueryAnalysis:
        """Analyze query using heuristics.

        Args:
            query: User query string

        Returns:
            QueryAnalysis with recommendations
        """
        query_lower = query.lower().strip()

        # Extract keywords (simple tokenization)
        keywords = self._extract_keywords(query)

        # Extract potential entities (capitalized words, quoted phrases)
        entities = self._extract_entities(query)

        # Classify query type
        query_type = self._classify_query_type(query_lower)

        # Determine intent
        intent = self._determine_intent(query_type)

        # Select strategies based on query type
        strategies = self._select_strategies(query_type, keywords, entities)

        # Calculate confidence based on pattern matches
        confidence = self._calculate_confidence(query_lower, query_type)

        return QueryAnalysis(
            query=query,
            query_type=query_type,
            keywords=keywords,
            entities=entities,
            intent=intent,
            recommended_strategies=strategies,
            confidence=confidence,
            metadata={
                "analysis_method": "heuristic",
            },
        )

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query."""
        # Remove common stop words
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "about",
            "against",
            "among",
            "throughout",
            "despite",
            "towards",
            "upon",
            "concerning",
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "any",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
        }

        # Tokenize and filter
        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 1]

        return keywords[:10]  # Limit to top 10

    def _extract_entities(self, query: str) -> list[str]:
        """Extract potential entities from query."""
        entities = []

        # Quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Capitalized words (potential proper nouns)
        # Skip first word which might be capitalized due to sentence start
        words = query.split()
        if len(words) > 1:
            for word in words[1:]:
                if word[0].isupper() and word.isalpha():
                    entities.append(word)

        # Technical terms (CamelCase, snake_case)
        camel_case = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", query)
        entities.extend(camel_case)

        return list(set(entities))[:5]  # Dedupe and limit

    def _classify_query_type(self, query_lower: str) -> QueryType:
        """Classify query type based on patterns."""
        # Check patterns in order of specificity
        # PROCEDURAL first (how-to questions are very specific)
        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.PROCEDURAL

        # ANALYTICAL (why, compare questions)
        for pattern in self.ANALYTICAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.ANALYTICAL

        # FACTUAL before CONCEPTUAL (factual uses same "what is" but asks for facts)
        # Check for factual patterns first
        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.FACTUAL

        # CONCEPTUAL (explain, describe - more generic)
        for pattern in self.CONCEPTUAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.CONCEPTUAL

        # Short queries without question words → keyword search
        if len(query_lower.split()) <= 3 and "?" not in query_lower:
            return QueryType.KEYWORD

        # Default to exploratory for longer, unclassified queries
        return QueryType.EXPLORATORY

    def _determine_intent(self, query_type: QueryType) -> str:
        """Determine user intent based on query type."""
        intent_map = {
            QueryType.FACTUAL: "find_specific_fact",
            QueryType.CONCEPTUAL: "understand_concept",
            QueryType.PROCEDURAL: "learn_process",
            QueryType.ANALYTICAL: "analyze_relationship",
            QueryType.KEYWORD: "find_matching_content",
            QueryType.EXPLORATORY: "explore_topic",
        }
        return intent_map.get(query_type, "unknown")

    def _select_strategies(
        self,
        query_type: QueryType,
        keywords: list[str],
        entities: list[str],
    ) -> list[RetrievalStrategy]:
        """Select retrieval strategies based on analysis."""
        strategies = _base_strategies_for_query_type(query_type)

        # Add graph if entities detected (might benefit from relationship traversal)
        if entities and RetrievalStrategy.GRAPH not in strategies:
            strategies.append(RetrievalStrategy.GRAPH)

        return strategies

    def _calculate_confidence(self, query_lower: str, query_type: QueryType) -> float:
        """Calculate confidence in the analysis."""
        base_confidence = 0.5

        # Higher confidence for clear question patterns
        if query_type in (QueryType.FACTUAL, QueryType.PROCEDURAL):
            base_confidence += 0.3

        # Higher confidence for longer queries (more context)
        word_count = len(query_lower.split())
        if word_count >= 5:
            base_confidence += 0.1
        elif word_count <= 2:
            base_confidence -= 0.1

        # Question mark indicates clear question intent
        if "?" in query_lower:
            base_confidence += 0.1

        return min(max(base_confidence, 0.0), 1.0)

    async def _analyze_with_llm(self, query: str) -> QueryAnalysis:
        """Analyze query using LLM for better accuracy.

        Args:
            query: User query string

        Returns:
            QueryAnalysis with LLM-enhanced recommendations
        """
        system_prompt = """You are a query analyzer for a RAG system. Analyze the user query and return JSON:
{
  "query_type": "factual|conceptual|procedural|analytical|keyword|exploratory",
  "keywords": ["keyword1", "keyword2"],
  "entities": ["Entity1", "Entity2"],
  "intent": "brief intent description",
  "strategies": ["bm25", "vector", "graph"],
  "confidence": 0.8
}

Strategy selection guide:
- factual: bm25 + vector (exact match + semantic)
- conceptual: vector + graph (semantic understanding)
- procedural: vector + bm25 (step-by-step content)
- analytical: vector + graph (relationships matter)
- keyword: bm25 only (simple match)
- exploratory: all three (broad search)

Return only valid JSON, no explanation."""

        from houyi.adapters.llm.base import LLMMessage, MessageRole

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=system_prompt),
            LLMMessage(role=MessageRole.USER, content=f"Analyze: {query}"),
        ]

        response = await self._adapter.chat(messages, max_tokens=300, temperature=0.1)

        import json

        result = json.loads(response.content)
        strategies = _parse_llm_strategies(result.get("strategies"))

        return QueryAnalysis(
            query=query,
            query_type=QueryType(result.get("query_type", "exploratory")),
            keywords=result.get("keywords", []),
            entities=result.get("entities", []),
            intent=result.get("intent", ""),
            recommended_strategies=strategies,
            confidence=result.get("confidence", 0.5),
            metadata={
                "analysis_method": "llm",
            },
        )


async def analyze_query(
    query: str,
    adapter: LLMAdapter | None = None,
) -> QueryAnalysis:
    """Convenience function to analyze a query.

    Args:
        query: User query string
        adapter: Optional LLM adapter

    Returns:
        QueryAnalysis with recommendations
    """
    analyzer = QueryAnalyzer(adapter=adapter)
    return await analyzer.analyze(query)
