from __future__ import annotations

import json

from houyi.application.research.report import ReportGenerator, SectionEvidencePolicy
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
        report, timings = await gen.generate(_plan_with_outline(), _sources())
        assert len(report.sections) == 2
        assert report.summary == "Summary text."
        assert "report_sections_ms" in timings
        assert "report_summary_ms" in timings
        assert report.metadata.section_count == 2

    async def test_citations_parsed(self):
        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON, "Summary."])
        gen = ReportGenerator(llm)
        report, _ = await gen.generate(_plan_with_outline(), _sources())
        assert len(report.sections[0].citations) == 1
        assert report.sections[0].citations[0].reference_id == "ref_001"

    async def test_malformed_section_fallback(self):
        llm = MockLLM(responses=["Just plain text.", "Also plain.", "Summary."])
        gen = ReportGenerator(llm)
        report, _ = await gen.generate(_plan_with_outline(), _sources())
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
        report, _ = await gen.generate(plan, empty_src)
        assert len(report.sections) == 2
        assert report.metadata.source_count == 0

    async def test_all_sections_invalid_json(self):
        llm = MockLLM(responses=["<<<bad>>>", "<<<bad>>>", "Summary."])
        gen = ReportGenerator(llm)
        report, _ = await gen.generate(_plan_with_outline(), _sources())
        for sec in report.sections:
            assert sec.citations == []
            assert sec.content != ""

    async def test_section_count_matches_outline(self):
        llm = MockLLM(responses=[_SECTION_JSON, _SECTION_JSON, "Summary."])
        gen = ReportGenerator(llm)
        plan = _plan_with_outline()
        report, _ = await gen.generate(plan, _sources())
        assert len(report.sections) == len(plan.outline)
        assert report.metadata.section_count == len(plan.outline)

    async def test_section_source_selection(self):
        from houyi.application.research.report import (
            _DEFAULT_MAX_SECTION_SOURCES as _MAX_SECTION_SOURCES,
        )

        llm = MockLLM(responses=[])
        gen = ReportGenerator(
            llm,
            section_evidence_policy=SectionEvidencePolicy(
                candidate_pool_size=10,
                min_domain_diversity=3,
                require_content_usable=True,
            ),
        )
        many_sources = AggregatedSources(
            sources=[
                SourceReference(
                    reference_id="ref_best",
                    url="https://stats.gov/official-annual-report-2026",
                    title="Official Annual Report 2026",
                    snippet="official growth statistics and middle class income data",
                    reliability_score=0.95,
                ),
                SourceReference(
                    reference_id="ref_peer",
                    url="https://research.example.com/compare",
                    title="Comparison Research 2026",
                    snippet="detailed comparison analysis across providers and cost structures",
                    reliability_score=0.72,
                ),
                SourceReference(
                    reference_id="ref_news",
                    url="https://news.example.org/latest",
                    title="Latest Market Update",
                    snippet="recent market update with industry context and trend details",
                    reliability_score=0.68,
                ),
                SourceReference(
                    reference_id="ref_thin",
                    url="https://official.example.com/login",
                    title="Official Login Required",
                    snippet="tiny",
                    reliability_score=0.91,
                ),
                *[
                    SourceReference(
                        reference_id=f"ref_{idx}",
                        url=f"https://example.com/{idx}",
                        title=f"Generic commentary {idx}",
                        snippet="misc text",
                        reliability_score=0.2 + (idx * 0.01),
                    )
                    for idx in range(10)
                ],
            ],
            grouped_by_question={
                "q1": [
                    "ref_best",
                    "ref_peer",
                    "ref_news",
                    "ref_thin",
                    *[f"ref_{idx}" for idx in range(10)],
                ],
                "q2": ["ref_best", "ref_peer"],
            },
        )

        selected, total, metrics = gen.select_section_sources(
            ["q1"],
            many_sources,
            section_title="Income Growth Analysis",
            objective="Use official annual statistics to compare growth and income",
        )

        assert total == 14
        assert len(selected) == _MAX_SECTION_SOURCES
        assert selected[0].reference_id == "ref_best"
        assert "ref_0" not in {src.reference_id for src in selected}
        assert "ref_thin" not in {src.reference_id for src in selected}
        assert metrics["selected_domain_count"] >= 3
        assert metrics["cross_question_source_count"] >= 1
        assert metrics["content_usable_source_count"] == len(selected)

    async def test_selection_metrics(self):
        llm = MockLLM(responses=[])
        gen = ReportGenerator(
            llm,
            max_section_sources=2,
            section_evidence_policy=SectionEvidencePolicy(candidate_pool_size=4),
        )
        sources = AggregatedSources(
            sources=[
                SourceReference(
                    reference_id="ref_a",
                    url="https://alpha.example.com/a",
                    title="Alpha Report",
                    snippet="long enough alpha evidence snippet for section selection",
                    reliability_score=0.9,
                ),
                SourceReference(
                    reference_id="ref_b",
                    url="https://beta.example.com/b",
                    title="Beta Analysis",
                    snippet="long enough beta evidence snippet for broader coverage",
                    reliability_score=0.7,
                ),
            ],
            grouped_by_question={"q1": ["ref_a", "ref_b"], "q2": ["ref_a"]},
        )

        selected, total, metrics = gen.select_section_sources(
            ["q1"],
            sources,
            section_title="Section",
            objective="Compare providers",
        )

        assert total == 2
        assert len(selected) == 2
        assert metrics == {
            "selected_domain_count": 2,
            "authority_source_count": 1,
            "cross_question_source_count": 1,
            "content_usable_source_count": 2,
        }


class TestIntermediateContext:
    async def test_section_with_intermediate_context(self):
        from houyi.application.research.runtime.intermediate import IntermediateReport

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

        report, _ = await gen.generate(plan, _sources(), intermediate_reports=intermediates)
        assert len(report.sections) == 2
        assert report.summary == "Summary."

    async def test_intermediate_context_builds_analysis(self):
        from houyi.application.research.report import _intermediate_context
        from houyi.application.research.runtime.intermediate import IntermediateReport

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

    async def test_intermediate_context_is_capped(self):
        from houyi.application.research.report import (
            _DEFAULT_INTERMEDIATE_CONTEXT_MAX_CHARS as _INTERMEDIATE_CONTEXT_MAX_CHARS,
        )
        from houyi.application.research.report import (
            _intermediate_context,
        )
        from houyi.application.research.runtime.intermediate import IntermediateReport

        reports = {
            f"q{idx}": IntermediateReport(
                question_id=f"q{idx}",
                question=f"Question {idx}",
                analysis="A" * 1200,
                confidence=0.8,
            )
            for idx in range(5)
        }

        result = _intermediate_context(list(reports), reports)
        assert len(result) <= _INTERMEDIATE_CONTEXT_MAX_CHARS


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
        report, _ = await gen.generate(plan, _sources())
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

    async def test_extracts_from_prefix_text(self):
        """LLM sometimes prepends prose before the JSON object."""
        from houyi.application.research.report import _parse_section

        raw = "Here is the section:\n" + _SECTION_JSON
        section = _parse_section("Test Section", raw)
        assert "AI frameworks" in section.content
        assert len(section.citations) == 1

    async def test_raw_json_not_shown(self):
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
