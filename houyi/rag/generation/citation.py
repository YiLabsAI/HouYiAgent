"""Citation generation for RAG answers.

Provides inline citation markers and reference lists for generated answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from houyi.rag.types import SearchResult, Source


@dataclass
class Citation:
    """A citation reference."""

    id: int
    source: Source
    snippet: str = ""
    relevance: float = 0.0


@dataclass
class CitedAnswer:
    """An answer with inline citations."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    references: str = ""  # Formatted reference list

    def to_markdown(self) -> str:
        """Convert to markdown with footnotes."""
        if not self.references:
            return self.text
        return f"{self.text}\n\n---\n\n{self.references}"


class CitationGenerator:
    """Generates citations for RAG answers.

    Adds inline citation markers [1], [2], etc. and builds reference lists.
    """

    def __init__(self, max_citations: int = 10) -> None:
        """Initialize citation generator.

        Args:
            max_citations: Maximum number of citations to include
        """
        self.max_citations = max_citations

    def generate_citations(
        self,
        answer: str,
        sources: list[Source],
        results: list[SearchResult] | None = None,
    ) -> CitedAnswer:
        """Generate citations for an answer.

        Args:
            answer: The generated answer text
            sources: List of source references
            results: Optional search results with relevance scores

        Returns:
            CitedAnswer with inline citations and references
        """
        if not sources:
            return CitedAnswer(text=answer)

        # Build citation list
        citations: list[Citation] = []
        for i, source in enumerate(sources[: self.max_citations], 1):
            relevance = 0.0
            if results:
                for r in results:
                    if r.file_path == source.file_path:
                        relevance = r.score
                        break

            citations.append(
                Citation(
                    id=i,
                    source=source,
                    snippet=source.snippet[:200] if source.snippet else "",
                    relevance=relevance,
                )
            )

        # Add inline citations to answer if not present
        cited_text = self._add_inline_citations(answer, citations)

        # Build reference list
        references = self._format_references(citations)

        return CitedAnswer(
            text=cited_text,
            citations=citations,
            references=references,
        )

    def _add_inline_citations(self, text: str, citations: list[Citation]) -> str:
        """Add inline citation markers to text.

        If the text already has citation markers, keep them.
        Otherwise, add markers based on content matching.

        Args:
            text: Original text
            citations: List of citations

        Returns:
            Text with inline citation markers
        """
        # Check if already has citations
        if re.search(r"\[\d+\]", text):
            return text

        # Simple heuristic: add citation at end of each paragraph
        # that might reference a source
        paragraphs = text.split("\n\n")
        cited_paragraphs = []

        for para in paragraphs:
            if not para.strip():
                cited_paragraphs.append(para)
                continue

            # Find matching citations for this paragraph
            matching_ids = []
            para_lower = para.lower()

            for citation in citations:
                # Match by snippet content
                if citation.snippet:
                    snippet_words = set(citation.snippet.lower().split()[:10])
                    para_words = set(para_lower.split())
                    overlap = len(snippet_words & para_words)
                    if overlap >= 3:
                        matching_ids.append(citation.id)

            # Add citation markers
            if matching_ids:
                markers = "".join(f"[{id}]" for id in matching_ids[:3])
                para = f"{para.rstrip()} {markers}"

            cited_paragraphs.append(para)

        return "\n\n".join(cited_paragraphs)

    def _format_references(self, citations: list[Citation]) -> str:
        """Format reference list.

        Args:
            citations: List of citations

        Returns:
            Formatted reference list as markdown
        """
        if not citations:
            return ""

        lines = ["**References:**\n"]
        for citation in citations:
            source = citation.source
            location = f", {source.location}" if source.location else ""
            lines.append(f"[{citation.id}] {source.file_path}{location}")

        return "\n".join(lines)

    def extract_cited_sources(self, text: str, sources: list[Source]) -> list[Source]:
        """Extract sources that are actually cited in the text.

        Args:
            text: Text with citation markers
            sources: All available sources

        Returns:
            List of sources that are cited
        """
        # Find all citation markers
        markers = re.findall(r"\[(\d+)\]", text)
        cited_ids = {int(m) for m in markers}

        # Return corresponding sources
        return [s for i, s in enumerate(sources, 1) if i in cited_ids]
