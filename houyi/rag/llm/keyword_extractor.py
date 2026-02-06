"""Extract search keywords from query using LLM."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Extract search keywords from natural language query using LLM.

    Used in both Agentic and Indexed modes for:
    - Initial keyword extraction from user query
    - Query expansion with synonyms/related terms
    - Multi-round keyword refinement based on previous results
    """

    SYSTEM_PROMPT = """You are a keyword extraction expert for information retrieval.
Extract search keywords from the user's query.

Return ONLY valid JSON in this exact format:
{"keywords": ["keyword1", "keyword2"], "synonyms": {"keyword1": ["synonym1", "synonym2"]}}

Guidelines:
- Extract 3-10 meaningful keywords
- Focus on technical terms, proper nouns, and key concepts
- Include both English and Chinese keywords if present
- For synonyms, only include highly relevant alternatives
- Do not include common stop words"""

    def __init__(
        self,
        adapter: LLMAdapter,
        temperature: float = 0.3,
        max_keywords: int = 10,
    ) -> None:
        """Initialize keyword extractor.

        Args:
            adapter: LLM adapter for making API calls
            temperature: LLM temperature (lower = more deterministic)
            max_keywords: Maximum number of keywords to extract
        """
        self._adapter = adapter
        self._temperature = temperature
        self._max_keywords = max_keywords

    async def extract(self, query: str) -> dict[str, Any]:
        """Extract keywords from query.

        Args:
            query: User's natural language query

        Returns:
            Dict with "keywords" list and "synonyms" dict
        """
        from houyi.llm.base import LLMMessage, MessageRole

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
            LLMMessage(role=MessageRole.USER, content=f"Extract keywords from: {query}"),
        ]

        try:
            response = await self._adapter.chat(
                messages=messages,
                temperature=self._temperature,
                max_tokens=500,
            )
            return self._parse_response(response.content)
        except Exception as e:
            logger.warning("Keyword extraction failed: %s, using simple extraction", e)
            return self._simple_extract(query)

    async def expand(
        self,
        keywords: list[str],
        context: str = "",
        previous_results: list[str] | None = None,
    ) -> list[str]:
        """Expand keywords with synonyms and related terms.

        Args:
            keywords: Initial keywords to expand
            context: Optional context about the knowledge base
            previous_results: Results from previous search rounds

        Returns:
            Expanded list of keywords
        """
        from houyi.llm.base import LLMMessage, MessageRole

        prompt = f"""Given these search keywords: {keywords}
{"Context: " + context if context else ""}
{"Previous search found: " + str(previous_results[:3]) if previous_results else ""}

Suggest 5-10 additional related search terms that might help find relevant information.
Return ONLY a JSON array: ["term1", "term2", ...]"""

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are a search query expansion expert."),
            LLMMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await self._adapter.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=300,
            )
            expanded = self._parse_array(response.content)
            return list(set(keywords + expanded))
        except Exception as e:
            logger.warning("Keyword expansion failed: %s", e)
            return keywords

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response to extract keywords."""
        try:
            # Try to find JSON in response
            content = content.strip()
            if content.startswith("```"):
                # Remove markdown code blocks
                lines = content.split("\n")
                content = "\n".join(line for line in lines if not line.startswith("```"))

            data = json.loads(content)

            keywords = data.get("keywords", [])
            if isinstance(keywords, list):
                keywords = keywords[: self._max_keywords]

            return {
                "keywords": keywords,
                "synonyms": data.get("synonyms", {}),
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, using simple extraction")
            return self._simple_extract(content)

    def _parse_array(self, content: str) -> list[str]:
        """Parse a JSON array from LLM response."""
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(line for line in lines if not line.startswith("```"))
            data = json.loads(content)
            if isinstance(data, list):
                return [str(item) for item in data]
            return []
        except json.JSONDecodeError:
            return []

    def _simple_extract(self, text: str) -> dict[str, Any]:
        """Simple rule-based keyword extraction as fallback."""
        # Remove common stop words and split
        stop_words = {
            "的",
            "是",
            "在",
            "了",
            "和",
            "与",
            "或",
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
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }

        words = text.replace("?", " ").replace("？", " ").replace(",", " ").split()
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 1]

        return {
            "keywords": keywords[: self._max_keywords],
            "synonyms": {},
        }
