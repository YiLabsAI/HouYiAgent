"""Unit tests for ReportGenerator."""

from __future__ import annotations

import json

from houyi.application.research.report import ReportGenerator
from houyi.application.research.types import (
    AggregatedSources,
    OutlineSection,
    ReportChunkType,
    ResearchPlan,
    SourceReference,
)

from .conftest import MockLLM

_SECTION_JSON = json.dumps(
    {
        "content": "This section covers AI frameworks [ref_001].",
        "citations": [
            {"reference_id": "ref_001", "text_span": "AI frameworks", "context": "overview"}
        ],
    }
)


def _plan_with_outline() -> ResearchPlan:
    return ResearchPlan(
        query="AI frameworks",
        outline=[
            OutlineSection(title="Overview", objective="Landscape", related_question_ids=[]),
            OutlineSection(title="Details", objective="Deep dive", related_question_ids=[]),
        ],
    )


def _sources() -> AggregatedSources:
    return AggregatedSources(
        sources=[
            SourceReference(
                reference_id="ref_001", url="https://a.com", title="Source A", snippet="snip"
            ),
        ],
        grouped_by_question={},
    )


class TestGenerate:
    async def test_produces_report(self):
        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON, "Summary text."])
        gen = ReportGenerator(llm)
        report = await gen.generate(_plan_with_outline(), _sources())
        assert len(report.sections) == 2
        assert report.summary == "Summary text."
        assert report.metadata.section_count == 2

    async def test_citations_parsed(self):
        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON, "Summary."])
        gen = ReportGenerator(llm)
        report = await gen.generate(_plan_with_outline(), _sources())
        assert len(report.sections[0].citations) == 1
        assert report.sections[0].citations[0].reference_id == "ref_001"

    async def test_malformed_section_fallback(self):
        llm = MockLLM(responses=["Just plain text.", "Also plain.", "Summary."])
        gen = ReportGenerator(llm)
        report = await gen.generate(_plan_with_outline(), _sources())
        assert report.sections[0].content == "Just plain text."
        assert report.sections[0].citations == []


class TestGenerateStream:
    async def test_stream_yields_chunks(self):
        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON])
        gen = ReportGenerator(llm)
        chunks = []
        async for chunk in gen.generate_stream(_plan_with_outline(), _sources()):
            chunks.append(chunk)
        starts = [c for c in chunks if c.chunk_type == ReportChunkType.SECTION_START]
        completes = [c for c in chunks if c.chunk_type == ReportChunkType.SECTION_COMPLETE]
        done = [c for c in chunks if c.chunk_type == ReportChunkType.COMPLETE]
        assert len(starts) == 2
        assert len(completes) == 2
        assert len(done) == 1

    async def test_stream_sequence_monotonic(self):
        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON])
        gen = ReportGenerator(llm)
        seqs = []
        async for chunk in gen.generate_stream(_plan_with_outline(), _sources()):
            seqs.append(chunk.sequence)
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)


class TestBoundaryAndInteraction:
    async def test_empty_sources_report(self):
        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON, "Summary."])
        gen = ReportGenerator(llm)
        plan = _plan_with_outline()
        empty_src = AggregatedSources(sources=[], grouped_by_question={})
        report = await gen.generate(plan, empty_src)
        assert len(report.sections) == 2
        assert report.metadata.source_count == 0

    async def test_all_sections_invalid_json(self):
        llm = MockLLM(responses=["<<<bad>>>", "<<<bad>>>", "Summary."])
        gen = ReportGenerator(llm)
        report = await gen.generate(_plan_with_outline(), _sources())
        for sec in report.sections:
            assert sec.citations == []
            assert sec.content != ""

    async def test_section_count_matches_outline(self):
        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON, "Summary."])
        gen = ReportGenerator(llm)
        plan = _plan_with_outline()
        report = await gen.generate(plan, _sources())
        assert len(report.sections) == len(plan.outline)
        assert report.metadata.section_count == len(plan.outline)


class TestIntermediateContext:
    async def test_section_with_intermediate_context(self):
        from houyi.application.research.intermediate import IntermediateReport

        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON, "Summary."])
        gen = ReportGenerator(llm)
        plan = _plan_with_outline()
        plan.outline[0].related_question_ids = ["q1"]
        plan.outline[1].related_question_ids = ["q2"]

        intermediates = [
            IntermediateReport(
                question_id="q1",
                question="What are current frameworks?",
                analysis="Detailed analysis of frameworks [ref_001].",
                key_findings=["Finding A"],
                confidence=0.85,
            ),
            IntermediateReport(
                question_id="q2",
                question="How do they compare?",
                analysis="Comparison analysis reveals [ref_001].",
                key_findings=["Finding B"],
                confidence=0.75,
            ),
        ]

        report = await gen.generate(plan, _sources(), intermediate_reports=intermediates)
        assert len(report.sections) == 2
        assert report.summary == "Summary."

    async def test_intermediate_context_builds_analysis(self):
        from houyi.application.research.intermediate import IntermediateReport
        from houyi.application.research.report import _intermediate_context

        ir = IntermediateReport(
            question_id="q1",
            question="Test question?",
            analysis="Deep analysis text with citations.",
            confidence=0.9,
        )
        result = _intermediate_context(["q1"], {"q1": ir})
        assert "Test question?" in result
        assert "Deep analysis text" in result
        assert "90%" in result


class TestParallelGeneration:
    async def test_sections_generated_concurrently(self):
        """Verify all outline sections are produced by the parallelized generate()."""
        plan = ResearchPlan(
            query="AI",
            outline=[
                OutlineSection(title=f"Section {i}", objective=f"obj {i}", related_question_ids=[])
                for i in range(5)
            ],
        )
        # 5 sections + 1 summary = 6 LLM calls
        llm = MockLLM(responses=[_SECTION_JSON] * 5 + ["Summary."])
        gen = ReportGenerator(llm)
        report = await gen.generate(plan, _sources())
        assert len(report.sections) == 5
        assert report.summary == "Summary."
        titles = [s.title for s in report.sections]
        for i in range(5):
            assert f"Section {i}" in titles


class TestParseSectionEdgeCases:
    async def test_strips_code_fence(self):
        from houyi.application.research.report import _parse_section

        fenced = "```json\n" + _SECTION_JSON + "\n```"
        section = _parse_section("Test Section", fenced)
        assert section.content != ""
        assert len(section.citations) == 1
        assert section.citations[0].reference_id == "ref_001"

    async def test_extracts_json_from_prefix_text(self):
        """LLM sometimes prepends prose before the JSON object."""
        from houyi.application.research.report import _parse_section

        raw = 'Here is the section:\n' + _SECTION_JSON
        section = _parse_section("Test Section", raw)
        assert "AI frameworks" in section.content
        assert len(section.citations) == 1

    async def test_raw_json_not_shown_as_content(self):
        """Ensure {\"content\":...} is parsed, not displayed as-is."""
        from houyi.application.research.report import _parse_section

        section = _parse_section("Title", _SECTION_JSON)
        assert not section.content.startswith("{")
        assert '"content"' not in section.content


class TestStripLeadingHeadingEdgeCases:
    async def test_blank_lines_before_heading(self):
        from houyi.application.research.report import _strip_leading_heading

        content = "\n\n## Overview\nContent here."
        result = _strip_leading_heading("Overview", content)
        assert "Content here." in result
        assert "## Overview" not in result
