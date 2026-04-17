from __future__ import annotations

import asyncio
from types import SimpleNamespace

from houyi.application.research.report import SectionEvidenceBundle
from houyi.application.research.runtime.report_pipeline import (
    ReportPipeline,
    _collect_sections_to_repair,
    _merge_validation_reports,
    _passes_retention_guard,
    _run_completeness_guard,
)
from houyi.application.research.types import (
    AggregatedSources,
    AnswerCoverageContract,
    Citation,
    CoverageFacet,
    ReportSection,
    ResearchReport,
    SourceReference,
)
from houyi.application.research.validation import SectionValidation, ValidationReport


def _section(
    *,
    title: str = "Overview",
    content: str = "Documented finding with support. " * 6,
    citations: bool = True,
) -> ReportSection:
    refs = [Citation(reference_id="ref_001")] if citations else []
    return ReportSection(title=title, content=content, citations=refs)


def _report(*sections: ReportSection) -> ResearchReport:
    return ResearchReport(title="Demo", sections=list(sections), references=[])


class TestCompletenessGuard:
    def test_flags_missing_citations(self):
        report = _report(_section(content="Useful but uncited content." * 6, citations=False))
        result = _run_completeness_guard(report)
        assert result.sections_needing_rewrite == 1
        assert "section has no citations" in result.sections[0].issues

    def test_skips_complete_sections(self):
        report = _report(_section())
        result = _run_completeness_guard(report)
        assert result.sections == []
        assert result.sections_needing_rewrite == 0


class TestValidationMerge:
    def test_merges_supplemental_issues(self):
        primary = ValidationReport(
            sections=[
                SectionValidation(
                    section_id="sec1",
                    title="Overview",
                    quality_score=60,
                    has_citations=True,
                    issues=["thin analysis"],
                    needs_rewrite=False,
                    reasoning="primary",
                )
            ],
            overall_score=60.0,
            sections_needing_rewrite=0,
            total_issues=1,
        )
        supplemental = ValidationReport(
            sections=[
                SectionValidation(
                    section_id="sec1",
                    title="Overview",
                    quality_score=25,
                    has_citations=False,
                    issues=["section has no citations"],
                    needs_rewrite=True,
                    reasoning="supplemental",
                )
            ],
            overall_score=0.0,
            sections_needing_rewrite=1,
            total_issues=1,
        )
        merged = _merge_validation_reports(primary, supplemental)
        assert merged is not None
        assert merged.sections_needing_rewrite == 1
        assert merged.sections[0].quality_score == 25
        assert "section has no citations" in merged.sections[0].issues


class TestRetentionGuard:
    def test_blocks_over_compression(self):
        previous = _section(content="Evidence rich section. " * 20)
        rewritten = _section(content="Too short.", citations=True)
        assert _passes_retention_guard(previous, rewritten) is False

    def test_blocks_citation_drop(self):
        previous = _section(content="Evidence rich section. " * 10, citations=True)
        rewritten = _section(content="Evidence rich section. " * 10, citations=False)
        assert _passes_retention_guard(previous, rewritten) is False

    def test_accepts_stable_rewrite(self):
        previous = _section(content="Evidence rich section. " * 10, citations=True)
        rewritten = _section(content="Evidence rich section updated. " * 9, citations=True)
        assert _passes_retention_guard(previous, rewritten) is True


class _StubReporter:
    def __init__(self) -> None:
        self.seen_contract = None
        self.seen_bundle = None

    def build_section_evidence_bundle(
        self,
        question_ids,
        aggregated,
        *,
        section_title,
        objective,
        coverage_contract=None,
    ):
        self.seen_contract = coverage_contract
        bundle = SectionEvidenceBundle(
            selected_sources=list(aggregated.sources),
            primary_evidence=list(aggregated.sources[:1]),
            counter_evidence=[],
            reserve_evidence=[],
            unresolved_gaps=[],
            caveat_obligations=list(
                coverage_contract.required_caveats if coverage_contract else []
            ),
            coverage_facets=[facet.name for facet in coverage_contract.must_cover_facets]
            if coverage_contract
            else [],
            comparison_axes=list(coverage_contract.comparison_axes if coverage_contract else []),
            evidence_expectations=list(
                coverage_contract.evidence_expectations if coverage_contract else []
            ),
            time_scope=coverage_contract.time_scope if coverage_contract else "",
            geo_scope=coverage_contract.geo_scope if coverage_contract else "",
        )
        return bundle, len(aggregated.sources), {}

    async def _generate_section(
        self,
        query,
        title,
        objective,
        sources,
        intermediate_context="",
        evidence_bundle=None,
    ):
        self.seen_bundle = evidence_bundle
        return ReportSection(
            title=title,
            content="Evidence rich section updated. " * 9,
            citations=[Citation(reference_id="ref_001")],
        )


class TestRepairUsesCoverageBundle:
    async def test_rewrite_reuses_section_bundle(self):
        reporter = _StubReporter()
        pipeline = ReportPipeline(
            reporter=reporter,
            validator=SimpleNamespace(),
            evaluator=SimpleNamespace(),
            url_validator=SimpleNamespace(),
            conflict_resolver=SimpleNamespace(),
            intermediate_gen=SimpleNamespace(),
            web_search=SimpleNamespace(),
            emit=lambda *args, **kwargs: None,
        )
        contract = AnswerCoverageContract(
            must_cover_facets=[CoverageFacet(name="income distribution")],
            required_caveats=["note sampling limits"],
        )
        outline_section = SimpleNamespace(
            title="Overview",
            objective="Explain the topic",
            related_question_ids=["q1"],
            coverage_contract=contract,
        )
        aggregated = AggregatedSources(
            sources=[
                SourceReference(
                    reference_id="ref_001",
                    url="https://stats.gov/report",
                    title="Official report",
                    snippet="income distribution evidence",
                    reliability_score=0.9,
                )
            ],
            grouped_by_question={"q1": ["ref_001"]},
        )
        result = await pipeline._rewrite_single_section(
            index=0,
            section_validation=SimpleNamespace(title="Overview", suggested_queries=[]),
            outline_section=outline_section,
            current_section=_section(content="Evidence rich section. " * 10, citations=True),
            plan_query="demo query",
            aggregated=aggregated,
            section_semaphore=asyncio.Semaphore(1),
            query_semaphore=asyncio.Semaphore(1),
        )
        assert result is not None
        assert reporter.seen_contract is contract
        assert reporter.seen_bundle is not None
        assert reporter.seen_bundle.coverage_facets == ["income distribution"]
        assert reporter.seen_bundle.caveat_obligations == ["note sampling limits"]


class TestRepairPriority:
    def test_collect_prioritizes_severity(self):
        report = _report(
            _section(title="Overview"),
            _section(title="Risks"),
        )
        plan = SimpleNamespace(
            outline=[
                SimpleNamespace(title="Overview"),
                SimpleNamespace(title="Risks"),
            ]
        )
        validation = ValidationReport(
            sections=[
                SectionValidation(
                    title="Overview",
                    quality_score=20,
                    has_citations=True,
                    issues=["thin analysis"],
                    needs_rewrite=True,
                    suggested_queries=[],
                ),
                SectionValidation(
                    title="Risks",
                    quality_score=20,
                    has_citations=False,
                    issues=["missing citations", "missing caveats"],
                    needs_rewrite=True,
                    suggested_queries=["risk evidence"],
                ),
            ]
        )
        repairs = _collect_sections_to_repair(validation, report, plan, limit=2)
        assert [item[1].title for item in repairs] == ["Risks", "Overview"]
