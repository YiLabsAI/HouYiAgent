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
