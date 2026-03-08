"""Tests for citation generation."""

from __future__ import annotations

from houyi.rag.generation.citation import Citation, CitationGenerator, CitedAnswer
from houyi.rag.types import SearchResult, Source


class TestCitationGenerator:
    def test_generate_citations_empty_sources(self) -> None:
        generator = CitationGenerator()
        result = generator.generate_citations("Test answer", [])
        assert result.text == "Test answer"
        assert result.citations == []
        assert result.references == ""

    def test_generate_citations_with_sources(self) -> None:
        generator = CitationGenerator()
        sources = [
            Source(file_path="/path/to/doc1.md", location="line 10", snippet="test content"),
            Source(file_path="/path/to/doc2.md", location="line 20", snippet="more content"),
        ]
        result = generator.generate_citations("Test answer about content", sources)

        assert len(result.citations) == 2
        assert result.citations[0].id == 1
        assert result.citations[1].id == 2
        assert "References:" in result.references

    def test_generate_citations_inherits_relevance_from_results(self) -> None:
        generator = CitationGenerator()
        sources = [
            Source(file_path="/path/to/doc1.md", snippet="alpha beta gamma"),
            Source(file_path="/path/to/doc2.md", snippet="delta epsilon zeta"),
        ]
        results = [
            SearchResult(
                source=Source(file_path="/path/to/doc1.md"),
                content="alpha",
                score=0.9,
            ),
            SearchResult(
                source=Source(file_path="/path/to/doc2.md"),
                content="delta",
                score=0.4,
            ),
        ]

        cited = generator.generate_citations("Answer text", sources, results=results)

        assert cited.citations[0].relevance == 0.9
        assert cited.citations[1].relevance == 0.4

    def test_generate_citations_adds_inline_markers_from_snippet_overlap(self) -> None:
        generator = CitationGenerator()
        sources = [
            Source(
                file_path="/path/to/doc1.md",
                snippet="machine learning system design",
            )
        ]

        cited = generator.generate_citations(
            "This paragraph discusses machine learning system design choices.",
            sources,
        )

        assert cited.text.endswith("[1]")

    def test_citation_dataclass(self) -> None:
        source = Source(file_path="/test.md")
        citation = Citation(id=1, source=source, snippet="test", relevance=0.9)
        assert citation.id == 1
        assert citation.relevance == 0.9

    def test_cited_answer_to_markdown(self) -> None:
        answer = CitedAnswer(
            text="Test answer",
            references="**References:**\n[1] doc.md",
        )
        markdown = answer.to_markdown()
        assert "Test answer" in markdown
        assert "References" in markdown

    def test_extract_cited_sources(self) -> None:
        generator = CitationGenerator()
        sources = [
            Source(file_path="/doc1.md"),
            Source(file_path="/doc2.md"),
            Source(file_path="/doc3.md"),
        ]
        text = "This is from source [1] and also [3]."
        cited = generator.extract_cited_sources(text, sources)

        assert len(cited) == 2
        assert cited[0].file_path == "/doc1.md"
        assert cited[1].file_path == "/doc3.md"
