from __future__ import annotations

import json

from houyi.application.research.report import (
    ReportGenerator,
    SectionEvidenceBundle,
    SectionEvidencePolicy,
    _analyse_critical_analysis,
    _analyse_visualization_gaps,
    _build_soft_checklist,
    _cjk_char_ratio,
    _compute_section_sidecar_metrics,
    _consolidate_short_paragraphs,
    _count_sentences,
    _detect_noisy_paragraphs,
    _query_is_english,
)
from houyi.application.research.types import (
    AggregatedSources,
    CoverageFacet,
    OutlineSection,
    ReportChunkType,
    ResearchPlan,
    SourceReference,
    SubQuestion,
)

from .conftest import MockLLM

# Content is intentionally padded above ``_SHORT_SECTION_WORD_THRESHOLD``
# (350 words) so legacy tests do not accidentally trigger the EN-only
# post-generation expansion pass added in the short-section guard.
_SECTION_JSON = json.dumps(
    {
        "content": (
            "This section covers AI frameworks [ref_001]. "
            + " ".join(f"Expanded narrative sentence {i} anchored to [ref_001]." for i in range(60))
        ),
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

    async def test_prompt_includes_section_context(self):
        class _CapturingLLM(MockLLM):
            def __init__(self, responses: list[str]) -> None:
                super().__init__(responses)
                self.prompts: list[str] = []

            async def chat(self, messages: list, **kwargs):
                self.prompts.append(str(messages[0]["content"]))
                return await super().chat(messages, **kwargs)

        llm = _CapturingLLM(responses=[_SECTION_JSON, _SECTION_JSON, "Summary text."])
        gen = ReportGenerator(llm)
        plan = _plan_with_outline()
        plan.outline[0].related_question_ids = ["q1"]
        plan.outline[1].related_question_ids = ["q2"]
        plan.sub_questions = [
            SubQuestion(question_id="q1", question="What frameworks lead the market?"),
            SubQuestion(question_id="q2", question="How do the leading frameworks differ?"),
        ]
        await gen.generate(plan, _sources())
        assert "Section position in report: 1 of 2" in llm.prompts[0]
        assert "Next section: Details" in llm.prompts[0]
        assert "Related sub-question focus:" in llm.prompts[0]
        assert "What frameworks lead the market?" in llm.prompts[0]


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
        assert metrics["primary_evidence_count"] >= 1

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
        assert metrics["selected_domain_count"] == 2
        assert metrics["authority_source_count"] == 1
        assert metrics["cross_question_source_count"] == 1
        assert metrics["content_usable_source_count"] == 2
        assert metrics["primary_evidence_count"] >= 1
        assert metrics["unresolved_gap_count"] == 0

    async def test_preserves_complementary_sources(self):
        llm = MockLLM(responses=[])
        gen = ReportGenerator(
            llm,
            max_section_sources=4,
            section_evidence_policy=SectionEvidencePolicy(candidate_pool_size=6),
        )
        sources = AggregatedSources(
            sources=[
                SourceReference(
                    reference_id="ref_cross_a",
                    url="https://stats.gov/report-a",
                    title="Official income report",
                    snippet="income statistics and middle class distribution",
                    reliability_score=0.95,
                ),
                SourceReference(
                    reference_id="ref_cross_b",
                    url="https://oecd.org/report-b",
                    title="OECD middle class outlook",
                    snippet="middle class pressure and household finance",
                    reliability_score=0.88,
                ),
                SourceReference(
                    reference_id="ref_local",
                    url="https://analysis.example.com/local",
                    title="Local analysis",
                    snippet="household balance sheet detail and debt ratio",
                    reliability_score=0.72,
                ),
                SourceReference(
                    reference_id="ref_extra",
                    url="https://news.example.com/extra",
                    title="Extra commentary",
                    snippet="recent commentary on middle income groups",
                    reliability_score=0.65,
                ),
            ],
            grouped_by_question={
                "q1": ["ref_cross_a", "ref_local"],
                "q2": ["ref_cross_a", "ref_cross_b"],
                "q3": ["ref_cross_b", "ref_extra"],
            },
        )
        selected, total, metrics = gen.select_section_sources(
            ["q1", "q2"],
            sources,
            section_title="Middle Class Size",
            objective="Compare middle class size, income, and financial pressure",
        )
        # total includes global-pool supplement when question candidates < floor
        assert total >= 3
        assert len(selected) >= 3
        assert metrics["cross_question_source_count"] >= 2
        assert metrics["authority_source_count"] >= 1
        assert {src.reference_id for src in selected} >= {"ref_cross_a", "ref_cross_b"}

    async def test_builds_evidence_bundle(self):
        llm = MockLLM(responses=[])
        gen = ReportGenerator(llm)
        plan = _plan_with_outline()
        plan.outline[0].coverage_contract.must_cover_facets = [
            CoverageFacet(name="income growth", intent="compare official statistics")
        ]
        sources = AggregatedSources(
            sources=[
                SourceReference(
                    reference_id="ref_official",
                    url="https://stats.gov/report",
                    title="Official income growth report",
                    snippet="official income growth statistics by year",
                    reliability_score=0.95,
                ),
                SourceReference(
                    reference_id="ref_risk",
                    url="https://analysis.example.com/risk",
                    title="Risk and limitation analysis",
                    snippet="however the dataset has limitations and risk of bias",
                    reliability_score=0.7,
                ),
            ],
            grouped_by_question={"q1": ["ref_official", "ref_risk"]},
        )
        bundle, total, metrics = gen.build_section_evidence_bundle(
            ["q1"],
            sources,
            section_title="Overview",
            objective="Compare income growth",
            coverage_contract=plan.outline[0].coverage_contract,
        )
        assert total == 2
        assert bundle.primary_evidence[0].reference_id == "ref_official"
        assert bundle.counter_evidence[0].reference_id == "ref_risk"
        assert bundle.coverage_facets == ["income growth"]
        assert metrics["counter_evidence_count"] == 1

    async def test_bundle_adds_reserve_evidence(self):
        llm = MockLLM(responses=[])
        gen = ReportGenerator(llm, max_section_sources=1)
        contract = _plan_with_outline().outline[0].coverage_contract
        contract.must_cover_facets = [
            CoverageFacet(name="income growth", intent="compare official statistics"),
            CoverageFacet(name="debt pressure", intent="capture household debt burden"),
        ]
        sources = AggregatedSources(
            sources=[
                SourceReference(
                    reference_id="ref_income",
                    url="https://stats.gov/income",
                    title="Official income growth report",
                    snippet="official income growth statistics by year",
                    reliability_score=0.95,
                ),
                SourceReference(
                    reference_id="ref_debt",
                    url="https://oecd.org/debt",
                    title="Household debt pressure outlook",
                    snippet="household debt pressure and repayment burden",
                    reliability_score=0.88,
                ),
            ],
            grouped_by_question={"q1": ["ref_income", "ref_debt"]},
        )
        bundle, _, metrics = gen.build_section_evidence_bundle(
            ["q1"],
            sources,
            section_title="Overview",
            objective="Compare income growth and debt pressure",
            coverage_contract=contract,
        )
        assert bundle.unresolved_gaps == ["debt pressure"]
        assert [src.reference_id for src in bundle.reserve_evidence] == ["ref_debt"]
        assert metrics["reserve_evidence_count"] == 1


class TestSourceFallback:
    """Verify _relevant_sources global-pool fallback for citation desert prevention."""

    async def test_supplements_sparse_questions(self):
        """Section with <FLOOR question sources should see global sources."""
        from houyi.application.research.report import _SECTION_SOURCE_FLOOR

        llm = MockLLM(responses=[])
        gen = ReportGenerator(llm, max_section_sources=4)

        # q1 gives only 2 sources — well below _SECTION_SOURCE_FLOOR.
        # global_only is NOT linked to q1 but is highly relevant to the title.
        sources = AggregatedSources(
            sources=[
                SourceReference(
                    reference_id="ref_q1_a",
                    url="https://a.example.com/a",
                    title="General social overview",
                    snippet="general sociology overview",
                    reliability_score=0.6,
                ),
                SourceReference(
                    reference_id="ref_q1_b",
                    url="https://b.example.com/b",
                    title="Social theory basics",
                    snippet="introduction to theory",
                    reliability_score=0.5,
                ),
                SourceReference(
                    reference_id="ref_global",
                    url="https://stats.gov/middle-class-definition",
                    title="Official middle class definition and income thresholds",
                    snippet="middle class definition income threshold statistics official",
                    reliability_score=0.95,
                ),
            ],
            grouped_by_question={"q1": ["ref_q1_a", "ref_q1_b"]},
        )

        assert _SECTION_SOURCE_FLOOR > 2  # precondition: q1 sources < floor

        selected, total, _ = gen.select_section_sources(
            ["q1"],
            sources,
            section_title="Middle Class Definition",
            objective="Define middle class income thresholds",
        )

        # The global source should appear because fallback triggered.
        selected_ids = {s.reference_id for s in selected}
        assert "ref_global" in selected_ids
        assert total == 3  # all three candidates were ranked

    async def test_no_fallback(self):
        """Section with >=FLOOR question sources should NOT pull global pool."""
        from houyi.application.research.report import _SECTION_SOURCE_FLOOR

        llm = MockLLM(responses=[])
        gen = ReportGenerator(llm, max_section_sources=4)

        # Create enough question-aligned sources to exceed the floor.
        q_sources = [
            SourceReference(
                reference_id=f"ref_q_{i}",
                url=f"https://q.example.com/{i}",
                title=f"Question-aligned source {i}",
                snippet="relevant question content for section",
                reliability_score=0.7 + i * 0.01,
            )
            for i in range(_SECTION_SOURCE_FLOOR + 2)
        ]
        unlinked = SourceReference(
            reference_id="ref_unlinked",
            url="https://unlinked.example.com",
            title="Unlinked global source",
            snippet="unlinked global source content",
            reliability_score=0.99,
        )
        sources = AggregatedSources(
            sources=[*q_sources, unlinked],
            grouped_by_question={"q1": [s.reference_id for s in q_sources]},
        )

        _, total, _ = gen.select_section_sources(
            ["q1"],
            sources,
            section_title="Overview",
            objective="General overview",
        )

        # Only question-aligned candidates should be in the ranked pool.
        assert total == len(q_sources)


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
        from houyi.application.research.report import _intermediate_context
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
        plan = ResearchPlan(
            query="AI",
            outline=[
                OutlineSection(title=f"Section {i}", objective=f"obj {i}", related_question_ids=[])
                for i in range(5)
            ],
        )
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
        from houyi.application.research.report import _parse_section

        raw = "Here is the section:\n" + _SECTION_JSON
        section = _parse_section("Test Section", raw)
        assert "AI frameworks" in section.content
        assert len(section.citations) == 1

    async def test_raw_json_not_shown(self):
        from houyi.application.research.report import _parse_section

        section = _parse_section("Title", _SECTION_JSON)
        assert not section.content.startswith("{")
        assert '"content"' not in section.content

    async def test_repair_unescaped_quote_envelope(self):
        # Writer envelope where the body contains a stray unescaped quote,
        # breaking json.loads. The repair fallback must still return clean
        # prose with the envelope tokens removed.
        from houyi.application.research.report import _parse_section

        broken = (
            '{\n  "content": "### Scope of debate\n'
            'Analysts disagree on thresholds, with "upper middle" definitions.\n'
            'Sources cluster into income-first and consumption-first camps."\n}'
        )
        section = _parse_section("Scope and controversies", broken)
        assert not section.content.lstrip().startswith("{")
        assert '"content"' not in section.content
        assert "Analysts disagree" in section.content

    async def test_repair_truncated_envelope(self):
        # Writer hit max_tokens mid-string: envelope opens but has no closing
        # quote, comma, or brace. The repair must still salvage the body.
        from houyi.application.research.report import _parse_section

        truncated = (
            '{\n  "content": "### Career status missing\n\n'
            "### Same-name disambiguation\n\n"
            "Several individuals share the target name; none match the "
            "intended subject in the technology domain [ref_abc123]. The "
            "remaining evidence is too thin to attribute"
        )
        section = _parse_section("Current status", truncated)
        assert not section.content.lstrip().startswith("{")
        assert '"content"' not in section.content
        assert "Same-name disambiguation" in section.content

    async def test_restores_orphan_mermaid_fence(self):
        # A writer that intends a Mermaid diagram sometimes emits only
        # the closing ``` and leaves the indented diagram body above it.
        # Markdown then renders the orphan fence as an unclosed code
        # block, or — once stripped — the indented body falls back to an
        # unlabelled indented code block. Parsing must restore a matching
        # opener (with a ``mermaid`` tag when arrows/keywords are present)
        # so the content renders as a proper labelled code block.
        from houyi.application.research.report import _parse_section

        content = (
            "### Architecture overview\n\n"
            "The evolution loop can be sketched as follows:\n\n"
            "    Env->>Trigger: metric change\n"
            "    Trigger->>Engine: start evolution\n"
            "    Engine->>Auditor: submit for review\n"
            "    end\n"
            "```\n\n"
            "This architecture reflects a move from reactive to proactive "
            "governance [ref_abc123]."
        )
        envelope = json.dumps({"content": content, "citations": []})
        section = _parse_section("Self-evolution mechanism", envelope)
        # A balanced ```mermaid ... ``` pair must now wrap the diagram,
        # and the prose on both sides must survive intact.
        assert "```mermaid" in section.content
        assert section.content.count("```") == 2
        assert "Env->>Trigger: metric change" in section.content
        assert "move from reactive to proactive governance" in section.content

    async def test_fence_balance_preserves_paired(self):
        from houyi.application.research.report import _parse_section

        paired = (
            "### Snippet\n\n"
            "Look at this code:\n\n"
            "```python\n"
            "print('ok')\n"
            "```\n\n"
            "Done [ref_123abcde]."
        )
        paired_env = json.dumps({"content": paired, "citations": []})
        paired_section = _parse_section("Snippet", paired_env)
        # Already balanced fences must not be disturbed.
        assert paired_section.content.count("```") == 2
        assert "```python" in paired_section.content

    async def test_drops_orphan(self):
        # When an orphan fence has no indented body above it (e.g. a
        # truncated ```json wrapper whose payload was already removed
        # upstream), the normalizer must drop the lone marker so the
        # surrounding prose stays readable.
        from houyi.application.research.report import _parse_section

        content = (
            "### Pure prose\n\n"
            "Sources converge on a single conclusion [ref_abc123].\n\n"
            "```\n\n"
            "Follow-up paragraph with no code block [ref_abc123]."
        )
        envelope = json.dumps({"content": content, "citations": []})
        section = _parse_section("Pure prose", envelope)
        assert "```" not in section.content
        assert "Sources converge" in section.content
        assert "Follow-up paragraph" in section.content

    async def test_strips_envelope_citations_trailer(self):
        # Some writers emit a malformed envelope where the citations
        # array is double-nested (escaped inside the content string AND
        # as a sibling field). ``json.loads`` still succeeds, but
        # ``data["content"]`` carries the escaped trailer verbatim and
        # it renders as visible JSON noise at the end of the section.
        # The normalizer must cut the content at the first
        # ``","citations":`` boundary.
        from houyi.application.research.report import _parse_section

        # Build content whose JSON-escaped form reproduces the trailer
        # that leaks after json.loads unescapes the outer envelope.
        leaked_content = (
            "### Real prose only\n\n"
            "Sources converge on a single conclusion [ref_abc123]. The "
            "report's core structure follows an audit\u2011governance "
            'loop.",\n  "citations": [\n    {\n      '
            '"reference_id": "ref_abc123",\n      "text_span": '
            '"audit\u2011governance loop"\n    }\n  ]'
        )
        envelope = json.dumps({"content": leaked_content, "citations": []})
        section = _parse_section("Core structure", envelope)
        assert "citations" not in section.content
        assert '"text_span"' not in section.content
        assert "Sources converge" in section.content
        # The clean prose ends at the period, not at the escaped quote.
        assert section.content.rstrip().endswith("audit\u2011governance loop.")

    async def test_comma_grouped_citations_expanded(self):
        # Writers occasionally emit [ref_a, ref_b, ref_c] as a single
        # bracketed group. Downstream resolvers only match single-ref
        # tokens, so the group previously leaked through as literal noise.
        # _parse_section must expand each group into atomic bracketed
        # tokens that each contain exactly one ref id.
        from houyi.application.research.report import _parse_section

        envelope = json.dumps(
            {
                "content": (
                    "Comma group [ref_a1b2c3d4, ref_e5f6a7b8] "
                    "and trailing triple [ref_c1c1c1c1,  ref_d2d2d2d2 , ref_e3e3e3e3]."
                ),
                "citations": [],
            }
        )
        section = _parse_section("Example", envelope)
        assert "[ref_a1b2c3d4, ref_e5f6a7b8]" not in section.content
        assert "[ref_a1b2c3d4][ref_e5f6a7b8]" in section.content
        assert "[ref_c1c1c1c1][ref_d2d2d2d2][ref_e3e3e3e3]" in section.content
        # Single-ref tokens must not be reshaped by the expansion.
        single_envelope = json.dumps(
            {"content": "Only one [ref_single01] reference.", "citations": []}
        )
        single = _parse_section("Example", single_envelope)
        assert "[ref_single01]" in single.content

    async def test_repair_citations_tail_envelope(self):
        # Envelope with an unescaped quote but with the ,"citations": tail
        # present. The separator must be the right boundary, not the tail
        # brace.
        from houyi.application.research.report import _parse_section

        broken = (
            '{"content": "Paragraph one has unescaped "quotes".\n'
            'Paragraph two follows.", "citations": []}'
        )
        section = _parse_section("Example", broken)
        assert not section.content.startswith("{")
        assert "Paragraph one" in section.content


class TestStripLeadingHeadingEdgeCases:
    async def test_blank_lines_before_heading(self):
        from houyi.application.research.report import _strip_leading_heading

        content = "\n\n## Overview\nContent here."
        result = _strip_leading_heading("Overview", content)
        assert "Content here." in result
        assert "## Overview" not in result


class TestNoiseRewrite:
    """LLM-assisted rewrite path must never leak comma-grouped citations."""

    async def test_rewrite_splits_groups(self):
        # The mock LLM returns a paragraph that contains the disallowed
        # bracketed-group citation form.  The rewrite helper must normalize
        # it before handing the text back to the caller.
        llm = MockLLM(
            responses=[
                "This is the rewritten paragraph [ref_001, ref_002] with evidence.",
            ]
        )
        gen = ReportGenerator(llm)
        result = await gen._rewrite_noisy_paragraph(
            "original noisy paragraph",
            title="Section",
            objective="Objective",
            available_refs="ref_001, ref_002",
        )
        assert "[ref_001, ref_002]" not in result
        assert "[ref_001]" in result
        assert "[ref_002]" in result

    async def test_strips_content_envelope(self):
        # Rewrite LLM occasionally responds with ``{"content": "..."}``
        # instead of plain prose. The wrapper must be unwrapped before the
        # text reaches downstream consumers; otherwise escaped fences
        # inside the JSON string classify the whole paragraph as
        # structural and bypass paragraph normalisation, producing
        # oversized monolith paragraphs.
        llm = MockLLM(
            responses=[
                '{"content": "Repaired prose with [ref_001] evidence."}',
            ]
        )
        gen = ReportGenerator(llm)
        result = await gen._rewrite_noisy_paragraph(
            "original noisy paragraph",
            title="Section",
            objective="Objective",
            available_refs="ref_001",
        )
        assert result.startswith("Repaired prose")
        assert '"content"' not in result
        assert "[ref_001]" in result

    async def test_clean_normalizes_joined(self):
        # _clean_section_noise joins rewritten paragraphs.  Even if a late
        # call site bypasses _rewrite_noisy_paragraph, the joined output
        # must still have groups normalized as a defense-in-depth step.
        llm = MockLLM(
            responses=[
                "Rewritten paragraph keeps the group format [ref_a, ref_b].",
            ]
        )
        gen = ReportGenerator(llm)
        noisy = (
            "We searched extensively across many databases during the "
            "research process and gathered thin results worth ignoring."
        )
        cleaned = await gen._clean_section_noise(
            noisy,
            title="Section",
            objective="Objective",
            available_refs=["ref_a", "ref_b"],
        )
        assert "[ref_a, ref_b]" not in cleaned
        assert "[ref_a]" in cleaned
        assert "[ref_b]" in cleaned


class TestNoiseDetection:
    def test_flags_search_narration(self):
        paragraphs = [
            "We searched for relevant information online and found limited results across multiple databases and web sources during the research process.",
            "The framework was introduced in 2020 and has been widely adopted across the industry for production use cases [ref_abc123].",
        ]
        noisy = _detect_noisy_paragraphs(paragraphs)
        assert 0 in noisy
        assert 1 not in noisy

    def test_flags_uncited_claims(self):
        noisy = _detect_noisy_paragraphs(
            [
                "This is a long enough paragraph that makes factual claims about the topic without any citation reference at all, which means it should be flagged.",
            ]
        )
        assert 0 in noisy

    def test_skips_headings(self):
        assert _detect_noisy_paragraphs(["## Section Heading", "Short line."]) == []

    def test_accepts_cited_paragraph(self):
        noisy = _detect_noisy_paragraphs(
            [
                "The framework achieved 95% accuracy in benchmark testing, outperforming all competitors [ref_abc123]. It was later adopted widely [ref_def456].",
            ]
        )
        assert noisy == []

    def test_flags_same_name_dump(self):
        noisy = _detect_noisy_paragraphs(
            [
                "Multiple people share the same name, making it difficult to determine which individual is being referenced in this research context.",
            ]
        )
        assert 0 in noisy

    def test_flags_retrieval_process(self):
        noisy = _detect_noisy_paragraphs(
            [
                "The retrieval process yielded several relevant documents that helped establish the baseline for our analysis of the topic area.",
            ]
        )
        assert 0 in noisy

    def test_skips_empty_paragraphs(self):
        assert _detect_noisy_paragraphs(["", "  ", "\n"]) == []


class TestSidecarMetrics:
    """Deterministic prompt-compliance metrics — zero LLM cost verification."""

    def test_bullet_line_ratio_high(self):
        content = "- item one\n- item two\n- item three\n- item four"
        m = _compute_section_sidecar_metrics(content, "overview_and_synthesis")
        assert m["sidecar_bullet_line_ratio"] == 1.0

    def test_bullet_line_ratio_low(self):
        content = (
            "This is a dense analytical paragraph with substantial claims "
            "supported by evidence [ref_001]. The analysis shows clear trends.\n\n"
            "Another paragraph continues the analysis with further depth "
            "and additional citations [ref_002]. Key patterns emerge."
        )
        m = _compute_section_sidecar_metrics(content, "overview_and_synthesis")
        assert m["sidecar_bullet_line_ratio"] == 0.0

    def test_citation_counts(self):
        content = (
            "Claim A is supported [ref_001]. Claim B also holds [ref_002]. "
            "And A was confirmed [ref_001]."
        )
        m = _compute_section_sidecar_metrics(content, "overview_and_synthesis")
        assert m["sidecar_citation_count"] == 3
        assert m["sidecar_unique_citation_count"] == 2

    def test_uncited_paragraph_detected(self):
        content = (
            "This paragraph has a citation supporting its claim [ref_001]. "
            "It meets the minimum length requirement.\n\n"
            "This paragraph makes claims without any citation at all and is long "
            "enough to be flagged as a substantive uncited paragraph by the detector."
        )
        m = _compute_section_sidecar_metrics(content, "overview_and_synthesis")
        assert m["sidecar_uncited_paragraph_count"] == 1

    def test_bold_heading_count(self):
        content = "### Sub-topic A\nContent.\n\n### Sub-topic B\nMore content."
        m = _compute_section_sidecar_metrics(content, "overview_and_synthesis")
        assert m["sidecar_bold_heading_count"] == 2

    def test_archetype_compliance_comparison(self):
        content = (
            "When compared to alternative approaches, the trade-off "
            "between cost and performance becomes clear [ref_001]."
        )
        m = _compute_section_sidecar_metrics(content, "comparison")
        assert m["sidecar_archetype_compliant"] == 1
        assert "compared" in m["sidecar_archetype_keywords_matched"]

    def test_archetype_compliance_false_for_mismatch(self):
        content = (
            "This section provides a general overview of the landscape "
            "and summarizes the main findings from the research [ref_001]."
        )
        m = _compute_section_sidecar_metrics(content, "comparison")
        assert m["sidecar_archetype_compliant"] == 0

    def test_archetype_risk_and_caveat(self):
        content = (
            "Despite promising results, the risk of bias remains. "
            "However, the limitation of the dataset must be noted [ref_001]."
        )
        m = _compute_section_sidecar_metrics(content, "risk_and_caveat")
        assert m["sidecar_archetype_compliant"] == 1

    def test_archetype_trend_and_state(self):
        content = (
            "The trend over the past decade shows steady growth [ref_001]. "
            "Since 2015, the trajectory has accelerated markedly."
        )
        m = _compute_section_sidecar_metrics(content, "trend_and_state")
        assert m["sidecar_archetype_compliant"] == 1

    def test_word_count(self):
        content = "one two three four five"
        m = _compute_section_sidecar_metrics(content, "overview_and_synthesis")
        assert m["sidecar_word_count"] == 5

    def test_empty_content(self):
        m = _compute_section_sidecar_metrics("", "overview_and_synthesis")
        assert m["sidecar_bullet_line_ratio"] == 0.0
        assert m["sidecar_citation_count"] == 0
        assert m["sidecar_word_count"] == 0

    def test_overview_archetype_no_keywords(self):
        content = "General overview text [ref_001]."
        m = _compute_section_sidecar_metrics(content, "overview_and_synthesis")
        assert m["sidecar_archetype_compliant"] == 0
        assert m["sidecar_archetype_keywords_matched"] == ""


class TestArchetypeSoftChecklist:
    """Verify archetype hints are injected into the soft checklist."""

    def _make_bundle(self, archetype: str = "overview_and_synthesis") -> SectionEvidenceBundle:
        return SectionEvidenceBundle(
            selected_sources=[],
            primary_evidence=[],
            counter_evidence=[],
            reserve_evidence=[],
            unresolved_gaps=[],
            caveat_obligations=[],
            coverage_facets=["topic_a"],
            comparison_axes=[],
            evidence_expectations=[],
            time_scope="",
            geo_scope="",
            section_archetype=archetype,
        )

    def test_comparison_hint_injected(self):
        bundle = self._make_bundle("comparison")
        checklist = _build_soft_checklist(bundle)
        assert "ARCHETYPE GUIDANCE (comparison)" in checklist
        assert "comparison dimensions" in checklist

    def test_risk_hint_injected(self):
        bundle = self._make_bundle("risk_and_caveat")
        checklist = _build_soft_checklist(bundle)
        assert "ARCHETYPE GUIDANCE (risk_and_caveat)" in checklist
        assert "tensions or contradictions" in checklist

    def test_trend_hint_injected(self):
        bundle = self._make_bundle("trend_and_state")
        checklist = _build_soft_checklist(bundle)
        assert "ARCHETYPE GUIDANCE (trend_and_state)" in checklist
        assert "temporal axis" in checklist

    def test_overview_no_extra_hint(self):
        bundle = self._make_bundle("overview_and_synthesis")
        checklist = _build_soft_checklist(bundle)
        assert "ARCHETYPE GUIDANCE" not in checklist

    def test_empty_bundle_returns_empty(self):
        bundle = SectionEvidenceBundle(
            selected_sources=[],
            primary_evidence=[],
            counter_evidence=[],
            reserve_evidence=[],
            unresolved_gaps=[],
            caveat_obligations=[],
            coverage_facets=[],
            comparison_axes=[],
            evidence_expectations=[],
            time_scope="",
            geo_scope="",
            section_archetype="overview_and_synthesis",
        )
        assert _build_soft_checklist(bundle) == ""

    def test_archetype_hint_uncapped(self):
        bundle = SectionEvidenceBundle(
            selected_sources=[],
            primary_evidence=[],
            counter_evidence=[],
            reserve_evidence=[],
            unresolved_gaps=["gap_a", "gap_b"],
            caveat_obligations=["caveat_a"],
            coverage_facets=["facet_a", "facet_b"],
            comparison_axes=["axis_a"],
            evidence_expectations=[],
            time_scope="",
            geo_scope="",
            section_archetype="comparison",
        )
        checklist = _build_soft_checklist(bundle)
        assert "ARCHETYPE GUIDANCE" in checklist
        assert "Topics to address" in checklist


class TestSectionMetadata:
    def test_outline_archetype_defaults(self):
        section = OutlineSection(title="Overview", objective="Survey")
        assert section.section_archetype == "overview_and_synthesis"


class TestCriticalAnalysisSignal:
    def test_detects_keyword(self):
        body = (
            "The economy grew 3.4% in 2024 [ref_001]. However, the "
            "methodology behind national accounts carries known limitations "
            "when comparing across provinces [ref_002]."
        )
        assert _analyse_critical_analysis(body) is True

    def test_misses_without_keyword(self):
        body = "Survey coverage rose to 80% in 2024 [ref_001]."
        assert _analyse_critical_analysis(body) is False


class TestVisualizationSignal:
    def test_detects_table_gap(self):
        body = (
            "Revenue was 1,234 units in 2021, 2,456 units in 2022, "
            "and 3,678 units in 2023 [ref_001]."
        )
        gaps = _analyse_visualization_gaps(body)
        assert gaps["needs_table"] is True
        assert gaps["needs_mermaid"] is False

    def test_detects_mermaid_gap(self):
        body = (
            "The proposed framework classifies the pipeline into three "
            "distinct stages with nested levels of responsibility."
        )
        gaps = _analyse_visualization_gaps(body)
        assert gaps["needs_mermaid"] is True

    def test_noop_with_table(self):
        body = (
            "| Metric | 2021 | 2022 | 2023 |\n"
            "| ---- | ---- | ---- | ---- |\n"
            "| Revenue | 1,234 | 2,456 | 3,678 |"
        )
        gaps = _analyse_visualization_gaps(body)
        assert gaps["needs_table"] is False


class TestParagraphConsolidation:
    def test_merges_short_paragraphs(self):
        body = "First claim.\n\nSecond claim.\n\nThird claim."
        out = _consolidate_short_paragraphs(body)
        # All three short paragraphs collapse into a single merged paragraph
        # because each is a single sentence and the running buffer crosses
        # the target length at the third one.
        assert out.count("\n\n") == 0
        assert out.startswith("First claim.")
        assert "Third claim." in out

    def test_preserves_structure(self):
        body = (
            "Short intro.\n\n"
            "- bullet one\n- bullet two\n\n"
            "Short mid.\n\n"
            "```python\nprint('x')\n```\n\n"
            "Short end."
        )
        out = _consolidate_short_paragraphs(body)
        # The list block and the code fence must survive untouched.
        assert "- bullet one" in out
        assert "```python" in out
        parts = out.split("\n\n")
        assert any(part.startswith("- bullet") for part in parts)
        assert any("```python" in part for part in parts)

    def test_counts_cjk_sentence_terminators(self):
        # CJK terminators "\u3002\uff01\uff1f" must register as sentence
        # boundaries; otherwise Chinese paragraphs collapse to "1 sentence"
        # and trigger unwanted merges. Guards the ``_SENTENCE_TERMINATORS``
        # regex so CJK parity with ASCII stays covered.
        assert (
            _count_sentences(
                "\u7b2c\u4e00\u53e5\u8bdd\u3002\u7b2c\u4e8c\u53e5\u8bdd\u3002\u7b2c\u4e09\u53e5\u8bdd\u3002"
            )
            == 3
        )
        assert _count_sentences("\u8b66\u544a\uff01\u9519\u8bef\uff1f\u89e3\u51b3\u3002") == 3
        # Mixed ASCII + CJK terminators.
        assert _count_sentences("First sentence. \u7b2c\u4e8c\u53e5\u3002") == 2

    def test_splits_long_english(self):
        # 10-sentence EN prose exceeds _SPLITTABLE_MAX_SENTENCES (8). Expect
        # it to be broken into >=2 chunks of ~4 sentences each. Guards the
        # regression where the LLM produced 15-20 sentence monoliths and
        # readability collapsed (~22-24 vs 42+ baseline).
        para = " ".join(f"Claim number {i}." for i in range(1, 11))
        out = _consolidate_short_paragraphs(para)
        chunks = out.split("\n\n")
        assert len(chunks) >= 2
        # No chunk may still carry more than _SPLITTABLE_MAX_SENTENCES.
        for chunk in chunks:
            assert _count_sentences(chunk) <= 8
        # No information loss: every original sentence survives.
        for i in range(1, 11):
            assert f"Claim number {i}." in out

    def test_splits_long_cjk(self):
        # 10-sentence CJK prose must split symmetrically; language-agnostic
        # by design. Uses the same _SENTENCE_TERMINATORS regex so CJK
        # terminator "\u3002" participates identically to ASCII ".".
        para = "".join(f"\u7b2c{i}\u53e5\u8bdd\u3002" for i in range(1, 11))
        out = _consolidate_short_paragraphs(para)
        chunks = out.split("\n\n")
        assert len(chunks) >= 2
        for i in range(1, 11):
            assert f"\u7b2c{i}\u53e5\u8bdd\u3002" in out

    def test_preserves_moderate_paragraph(self):
        # 6 sentences is below the split threshold; paragraph must not be
        # broken. Protects the ZH case1 52.55 baseline where most body
        # paragraphs sit in the 5-8 sentence band and already score high.
        para = " ".join(f"Sentence {i}." for i in range(1, 7))
        out = _consolidate_short_paragraphs(para)
        assert out.count("\n\n") == 0
        assert out.strip() == para

    def test_strips_per_paragraph_envelope(self):
        # Section writers sometimes emit multiple ``{"content": "..."}``
        # envelopes in a single response; ``_parse_section`` only strips
        # the first one, leaving subsequent envelopes as raw paragraphs.
        # ``_consolidate_short_paragraphs`` must unwrap them defensively
        # so the final article carries no JSON artefacts. Without this
        # path, repair / multi-envelope writer output continues to leak
        # wrapper prefixes into the body.
        envelope = '{"content": "Real prose with detail. More detail here."}'
        body = f"Intro paragraph here.\n\n{envelope}\n\nClosing paragraph."
        out = _consolidate_short_paragraphs(body)
        assert '"content"' not in out
        assert "{" not in out.replace("{#", "")  # allow hypothetical anchors
        assert "Real prose with detail." in out

    def test_splits_preserve_structure(self):
        # A structural block sitting between long prose paragraphs must
        # remain untouched (no split, no merge through the boundary).
        long_para = " ".join(f"Fact {i}." for i in range(1, 11))
        body = f"{long_para}\n\n- item one\n- item two\n\n{long_para}"
        out = _consolidate_short_paragraphs(body)
        assert "- item one\n- item two" in out

    def test_preserves_cjk_paragraph(self):
        # A Chinese prose paragraph with 4 CJK periods must register as 4
        # sentences and stay un-merged. Mirrors empirical findings on the
        # two historical bench2 case-1 articles: before the CJK terminator
        # fix the whole article was mis-counted as "all 1 sentence" and the
        # summary paragraph got merged; after the fix the 50.76 baseline has
        # zero merges and the 52.19 baseline has only one.
        well_formed = (
            "\u7b2c\u4e00\u53e5\u8bf4\u660e\u80cc\u666f\u3002"
            "\u7b2c\u4e8c\u53e5\u7ed9\u51fa\u5173\u952e\u53d1\u73b0\u3002"
            "\u7b2c\u4e09\u53e5\u8ba8\u8bba\u5c40\u9650\u3002"
            "\u7b2c\u56db\u53e5\u603b\u7ed3\u7ed3\u8bba\u3002"
        )
        heading = "## \u7ed3\u6784\u6027\u5206\u8282\u6807\u9898"
        tail = "\u53e6\u4e00\u6bb5\u5185\u5bb9\u3002"
        body = f"{well_formed}\n\n{heading}\n\n{tail}"
        out = _consolidate_short_paragraphs(body)
        # Well-formed CJK prose must survive verbatim.
        assert well_formed in out
        assert heading in out


class TestScrubArtifacts:
    """``_scrub_generation_artifacts`` is the last-mile junk-token gate.

    Protects the scored article from three failure modes: multiline
    ``{"content": "..."}`` envelope leaks, fenced blocks whose body
    devolved into ``ref_<hex>`` / ``sync`` / ``30s`` tokens, and
    orphan hex reference IDs that escape the citation renumberer.
    """

    def test_noop_on_clean_prose(self):
        from houyi.application.research.report import _scrub_generation_artifacts

        body = (
            "First paragraph with real content.\n\n"
            "Second paragraph with a valid citation [ref_abc123].\n\n"
            "Closing paragraph."
        )
        assert _scrub_generation_artifacts(body) == body

    def test_drops_junk_mermaid(self):
        from houyi.application.research.report import _scrub_generation_artifacts

        body = (
            "Intro paragraph.\n\n"
            "```mermaid\n"
            "graph TD\n"
            "A --> B ref_c13d306e ref_d66b7514 sync sync sync\n"
            "```\n\n"
            "Closing paragraph."
        )
        out = _scrub_generation_artifacts(body)
        assert "mermaid" not in out
        assert "ref_c13d306e" not in out
        assert "Intro paragraph." in out
        assert "Closing paragraph." in out

    def test_keeps_valid_mermaid(self):
        from houyi.application.research.report import _scrub_generation_artifacts

        body = "Intro.\n\n```mermaid\nflowchart LR\nA[Start] --> B[End]\n```\n\nClosing."
        out = _scrub_generation_artifacts(body)
        assert "flowchart LR" in out
        assert "A[Start]" in out

    def test_unwraps_multiline_json_envelope(self):
        # Closed envelope with a citations array: the scaffold must be
        # stripped but the inner prose preserved (the writer relies
        # on this to keep high-value sections alive on ZH Q1).
        from houyi.application.research.report import _scrub_generation_artifacts

        envelope = (
            "{\n"
            '  "content": "Inner paragraph one.\\n\\nInner paragraph two.",\n'
            '  "citations": []\n'
            "}"
        )
        body = f"Intro.\n\n{envelope}\n\n## Next Heading"
        out = _scrub_generation_artifacts(body)
        assert '"content"' not in out
        assert "Inner paragraph one." in out
        assert "Inner paragraph two." in out
        assert "## Next Heading" in out

    def test_unwraps_truncated_envelope(self):
        # Malformed (never-closed) envelope leaking into the body.
        # Observed on ZH Q1: the writer emits ``{\\n  "content": "...``
        # then flows into a Markdown section heading without closing
        # the object.  The unwrap path must recover the inner prose.
        from houyi.application.research.report import _scrub_generation_artifacts

        body = (
            "Intro paragraph.\n\n"
            "{\n"
            '  "content": "Recovered prose with detail.\\n\\n### Sub'
            "\\nMore detail here.\n\n"
            "## Next Heading\n\n"
            "Tail paragraph."
        )
        out = _scrub_generation_artifacts(body)
        assert '"content"' not in out
        assert "Recovered prose with detail." in out
        assert "More detail here." in out
        assert "## Next Heading" in out
        assert "Tail paragraph." in out

    def test_drops_orphan_ref(self):
        from houyi.application.research.report import _scrub_generation_artifacts

        body = "Statement one. ref_c13d306e ref_d66b7514 Statement two."
        out = _scrub_generation_artifacts(body)
        assert "ref_c13d306e" not in out
        assert "ref_d66b7514" not in out
        assert "Statement one." in out
        assert "Statement two." in out

    def test_keeps_valid_ref_citation(self):
        from houyi.application.research.report import _scrub_generation_artifacts

        body = "Claim with citation [ref_c13d306e]. Next claim [ref_d66b7514]."
        out = _scrub_generation_artifacts(body)
        assert "[ref_c13d306e]" in out
        assert "[ref_d66b7514]" in out

    def test_idempotent(self):
        from houyi.application.research.report import _scrub_generation_artifacts

        body = (
            "Paragraph.\n\n"
            "```mermaid\n"
            "graph TD\n"
            "A --> B ref_c13d306e sync sync sync 30s 30s\n"
            "```\n\n"
            "Tail [ref_abc123]."
        )
        once = _scrub_generation_artifacts(body)
        twice = _scrub_generation_artifacts(once)
        assert once == twice

    def test_drops_unfenced_diagram_junk(self):
        # Some runs emit an orphan "graph TD" + ref/sync/30s stanza
        # without surrounding ``` fences, so the fenced-block scrub
        # cannot reach it.  The paragraph-level detector must catch
        # the residue without touching clean prose.
        from houyi.application.research.report import _scrub_generation_artifacts

        junk = (
            "graph TD\n"
            "    A[Node] --> B[Other]\n"
            "    H - (30s)\n"
            "    (30s\n"
            " ref_c [6] monetize cognition\n"
            " sync sync sync\n"
        )
        body = f"Intro paragraph.\n\n{junk}\n\nClosing paragraph."
        out = _scrub_generation_artifacts(body)
        assert "graph TD" not in out
        assert "Intro paragraph." in out
        assert "Closing paragraph." in out

    def test_keeps_prose_with_sync_keyword(self):
        # Clean prose that mentions "synchronous" or "30 seconds" must
        # not trip the paragraph-level junk filter; it requires both a
        # diagram marker and multiple independent junk signals.
        from houyi.application.research.report import _scrub_generation_artifacts

        body = (
            "The system syncs updates across nodes every 30s to "
            "maintain consistency.  Failure to sync within the "
            "window triggers a fallback path."
        )
        assert _scrub_generation_artifacts(body) == body


class TestQueryIsEnglish:
    def test_pure_en_true(self):
        assert _query_is_english("How do the wealthiest governments invest?") is True

    def test_pure_cjk_false(self):
        # ZH: zhongguo jiuda shouru jieceng fenbu (income-class distribution query).
        assert (
            _query_is_english(
                "\u4e2d\u56fd\u4e5d\u5927\u6536\u5165\u9636\u5c42\u5206"
                "\u5e03\u60c5\u51b5\u5982\u4f55\u5212\u5206\uff1f"
            )
            is False
        )

    def test_mixed_en(self):
        # Duan Yongping (\u6bb5\u6c38\u5e73) appears as a named entity in an EN query.
        assert _query_is_english("What is \u6bb5\u6c38\u5e73 investment philosophy?") is True

    def test_mixed_cjk(self):
        # Chinese-dominant query with stray English tokens must still read as ZH.
        assert (
            _query_is_english(
                "\u6bb5\u6c38\u5e73\u7684 investment \u54f2\u5b66\u662f\u4ec0\u4e48\uff1f"
            )
            is False
        )

    def test_empty_false(self):
        assert _query_is_english("") is False

    def test_digits_false(self):
        assert _query_is_english("2020 2050") is False


class TestCjkCharRatio:
    """``_cjk_char_ratio`` powers the EN section language gate."""

    def test_pure_english_zero(self):
        assert _cjk_char_ratio("Pure English prose.") == 0.0

    def test_empty_zero(self):
        assert _cjk_char_ratio("") == 0.0

    def test_pure_cjk_one(self):
        # Report body in pure Chinese.
        body = "\u672c\u62a5\u544a\u5206\u6790\u4e86\u4e2d\u56fd\u9636\u5c42"
        assert _cjk_char_ratio(body) == 1.0

    def test_mixed_half(self):
        # Three CJK chars + three ASCII letters (one word) = 0.5.
        assert _cjk_char_ratio("abc \u4e2d\u56fd\u8fd8") == 0.5

    def test_ignores_digits_and_punct(self):
        # Digits and punctuation must not dilute the ratio.
        # Body = "GDP 2024" + 5 CJK chars → ratio = 5/(5+3)=0.625.
        body = "GDP 2024 \u4e2d\u56fd\u7ecf\u6d4e\u5e74"
        assert _cjk_char_ratio(body) == 5 / 8


class TestLanguageGate:
    """EN-query language gate: translate CJK bodies to English.

    Protects English queries from emitting Chinese-dominant reports
    when search evidence is CJK-heavy; this was observed as a major
    driver of the ``inst`` / ``comp`` gap on EN leaderboard cases.
    """

    def _source(self) -> SourceReference:
        return SourceReference(
            reference_id="ref_001",
            url="https://a.com",
            title="Source A",
            snippet="government wealth and sovereign funds overview",
        )

    def _cjk_body(self) -> str:
        # ~100 CJK chars, minimal ASCII — triggers the gate.
        return (
            "\u672c\u8282\u5206\u6790\u4e86\u653f\u5e9c\u4e3b\u6743"
            "\u8d22\u5bcc\u57fa\u91d1\u7684\u6295\u8d44\u7ed3\u6784"
            "\u4e0e\u6536\u76ca\u8d8b\u52bf [ref_001]\u3002"
            "\u8fd1\u5e74\u6765\u5728\u4e2d\u4e1c\u4e0e\u4e9a\u6d32"
            "\u4e3b\u6743\u57fa\u91d1\u6301\u7eed\u62e9\u6301\u79d1"
            "\u6280\u4e0e\u7eff\u80fd\u8d44\u4ea7 [ref_001]\u3002"
        )

    def _english_translation(self) -> str:
        return (
            "This section analyses the investment structure and "
            "yield trends of sovereign wealth funds [ref_001]. In "
            "recent years Middle Eastern and Asian sovereign funds "
            "have steadily increased allocations to technology and "
            "green-energy assets [ref_001]. "
            + " ".join(f"Extra sentence {i} elaborates on governance [ref_001]." for i in range(40))
        )

    async def test_translates_cjk_body(self):
        initial = json.dumps(
            {
                "content": self._cjk_body(),
                "citations": [
                    {
                        "reference_id": "ref_001",
                        "text_span": "sovereign funds",
                        "context": "overview",
                    }
                ],
            }
        )
        translated = json.dumps({"content": self._english_translation(), "citations": []})
        llm = MockLLM(responses=[initial, translated])
        gen = ReportGenerator(llm, expand_short_sections=False)
        section = await gen._generate_section(
            query="Researching how the world's wealthiest governments invest.",
            title="Sovereign Fund Allocation",
            objective="Survey sovereign-fund investment trends.",
            sources=[self._source()],
        )
        assert _cjk_char_ratio(section.content) < 0.15
        assert "sovereign wealth funds" in section.content
        assert llm._call_count == 2

    async def test_skips_clean_english(self):
        # Already-English section must not trigger the translator.
        english = json.dumps(
            {
                "content": (
                    "The global sovereign fund landscape is dominated by "
                    "Middle Eastern and Asian investors [ref_001]. "
                    + " ".join(f"Supporting sentence {i} adds detail [ref_001]." for i in range(30))
                ),
                "citations": [],
            }
        )
        llm = MockLLM(responses=[english])
        gen = ReportGenerator(llm, expand_short_sections=False)
        section = await gen._generate_section(
            query="How do sovereign funds allocate capital?",
            title="Allocation",
            objective="Overview.",
            sources=[self._source()],
        )
        assert llm._call_count == 1
        assert "sovereign fund landscape" in section.content

    async def test_skips_cjk_query(self):
        # CJK query: the gate must not fire, even for a CJK body.
        body = json.dumps({"content": self._cjk_body(), "citations": []})
        llm = MockLLM(responses=[body])
        gen = ReportGenerator(llm, expand_short_sections=False)
        # ZH: zhuquan caifu jijin touzi qushi.
        section = await gen._generate_section(
            query=(
                "\u4e3b\u6743\u8d22\u5bcc\u57fa\u91d1\u6295\u8d44\u8d8b\u52bf\u5982\u4f55\uff1f"
            ),
            title="Trend",
            objective="Survey.",
            sources=[self._source()],
        )
        assert llm._call_count == 1
        assert _cjk_char_ratio(section.content) > 0.5

    async def test_rejects_no_progress_translation(self):
        # Misbehaving LLM returns CJK again: original body must survive.
        initial = json.dumps({"content": self._cjk_body(), "citations": []})
        still_cjk = json.dumps({"content": self._cjk_body(), "citations": []})
        llm = MockLLM(responses=[initial, still_cjk])
        gen = ReportGenerator(llm, expand_short_sections=False)
        section = await gen._generate_section(
            query="How do sovereign funds allocate capital?",
            title="Allocation",
            objective="Overview.",
            sources=[self._source()],
        )
        # Gate fired (2 calls) but rejected the no-progress response.
        assert llm._call_count == 2
        assert _cjk_char_ratio(section.content) > 0.5


class TestShortSectionExpand:
    """EN-only post-gen expansion pass."""

    def _source(self) -> SourceReference:
        return SourceReference(
            reference_id="ref_001",
            url="https://a.com",
            title="Source A",
            snippet="concentration data 50-80 percent across 5-10 names",
        )

    def _short_en_body(self) -> str:
        # ~120 words: below the 220-word threshold so expansion triggers, but
        # large enough that Guard B (1.5x growth cap) permits a realistic
        # expanded draft rather than falsely rejecting a modest rewrite.
        return " ".join(
            [
                f"Sentence {i} about Buffett's shift from Graham cigar-butt "
                f"style to quality [ref_001]."
                for i in range(12)
            ]
        )

    def _expanded_en_body(self) -> str:
        # Short body is ~144 words (12 sentences x 12 words); this expansion
        # lands at ~200 words (20 sentences x 10 words) — about 1.39x — safely
        # under Guard B's 1.5x cap.
        return " ".join(
            [
                f"Sentence {i} deepens the analysis with quantitative evidence [ref_001]."
                for i in range(20)
            ]
        )

    async def test_expands_en_section(self):
        short = json.dumps(
            {
                "content": self._short_en_body(),
                "citations": [
                    {"reference_id": "ref_001", "text_span": "quality", "context": "shift"}
                ],
            }
        )
        expanded = json.dumps(
            {
                "content": self._expanded_en_body(),
                "citations": [
                    {
                        "reference_id": "ref_001",
                        "text_span": "analysis",
                        "context": "deep",
                    }
                ],
            }
        )
        llm = MockLLM(responses=[short, expanded])
        gen = ReportGenerator(llm)
        section = await gen._generate_section(
            query="How did Buffett evolve from Graham to quality investing?",
            title="Buffett Evolution",
            objective="Trace the shift from cigar-butt to quality-business investing.",
            sources=[self._source()],
        )
        # Expanded body contains the new marker phrase and dropped the old one.
        assert "deepens the analysis" in section.content
        # Must make a second LLM call beyond the initial section + any noise rewrite.
        assert llm._call_count >= 2

    async def test_skips_cjk_query(self):
        short = json.dumps(
            {
                "content": self._short_en_body(),
                "citations": [],
            }
        )
        llm = MockLLM(responses=[short])
        gen = ReportGenerator(llm)
        # ZH: bafeite ruhe cong gelieamu zouxiang zhiliang touzi (Buffett-quality-investing query).
        section = await gen._generate_section(
            query=(
                "\u5df4\u83f2\u7279\u5982\u4f55\u4ece\u683c\u96f7\u5384"
                "\u59c6\u8d70\u5411\u8d28\u91cf\u6295\u8d44\uff1f"
            ),
            title="Buffett Evolution",
            objective="Trace the shift from cigar-butt to quality-business investing.",
            sources=[self._source()],
        )
        # Only the initial section call should have fired for a CJK query.
        assert llm._call_count == 1
        assert section.content.startswith("Sentence 0 about Buffett")

    async def test_skips_long_body(self):
        long_body = " ".join(
            [f"Sentence {i} provides evidence and context [ref_001]." for i in range(60)]
        )
        long_json = json.dumps({"content": long_body, "citations": []})
        llm = MockLLM(responses=[long_json])
        gen = ReportGenerator(llm)
        await gen._generate_section(
            query="How did Buffett evolve from Graham to quality investing?",
            title="Buffett Evolution",
            objective="Trace the shift from cigar-butt to quality-business investing.",
            sources=[self._source()],
        )
        # Long initial body must not trigger a second LLM call.
        assert llm._call_count == 1

    async def test_flag_skips_expand(self):
        short = json.dumps(
            {
                "content": self._short_en_body(),
                "citations": [],
            }
        )
        llm = MockLLM(responses=[short])
        gen = ReportGenerator(llm, expand_short_sections=False)
        await gen._generate_section(
            query="How did Buffett evolve from Graham to quality investing?",
            title="Buffett Evolution",
            objective="Trace the shift from cigar-butt to quality-business investing.",
            sources=[self._source()],
        )
        assert llm._call_count == 1

    async def test_skips_structured_section(self):
        # Guard A: section body already carries internal ``###`` subheadings,
        # which triggered a duplication failure mode where the LLM re-emitted
        # the whole subheading tree verbatim underneath itself.
        structured = json.dumps(
            {
                "content": (
                    "Opening paragraph on the shift [ref_001].\n\n"
                    "### The Cigar-Butt Foundation\n\n"
                    "Graham era analysis [ref_001].\n\n"
                    "### The Munger Catalyst\n\n"
                    "Quality turn analysis [ref_001]."
                ),
                "citations": [{"reference_id": "ref_001", "text_span": "shift", "context": ""}],
            }
        )
        llm = MockLLM(responses=[structured])
        gen = ReportGenerator(llm)
        section = await gen._generate_section(
            query="How did Buffett evolve from Graham to quality investing?",
            title="Buffett Evolution",
            objective="Trace the shift from cigar-butt to quality-business investing.",
            sources=[self._source()],
        )
        # Guard A must fire: no second LLM call, structured body preserved.
        assert llm._call_count == 1
        assert "### The Cigar-Butt Foundation" in section.content
        assert "### The Munger Catalyst" in section.content

    async def test_rejects_overlong_expand(self):
        # Guard B: expansion grew >1.5x, must be discarded.
        short = json.dumps(
            {
                "content": self._short_en_body(),
                "citations": [],
            }
        )
        # 40 sentences x ~10 words = ~400 words.  _short_en_body is ~120 words,
        # so ratio ~3.3x, well above the 1.5x cap.
        overlong = json.dumps(
            {
                "content": " ".join(
                    [
                        f"Sentence {i} overgrown with excessive padded verbose detail [ref_001]."
                        for i in range(40)
                    ]
                ),
                "citations": [],
            }
        )
        llm = MockLLM(responses=[short, overlong])
        gen = ReportGenerator(llm)
        section = await gen._generate_section(
            query="How did Buffett evolve from Graham to quality investing?",
            title="Buffett Evolution",
            objective="Trace the shift from cigar-butt to quality-business investing.",
            sources=[self._source()],
        )
        # LLM was called for expansion, but the result was rejected.
        assert llm._call_count == 2
        # Original short body preserved; padded prose must not leak in.
        assert "overgrown" not in section.content
        assert "Sentence 0 about Buffett" in section.content

    async def test_rejects_new_subheadings(self):
        # Guard C: expansion introduced ``###`` subheadings not present in the
        # original flat body; treat as a duplication-style artifact and drop.
        short = json.dumps(
            {
                "content": self._short_en_body(),
                "citations": [],
            }
        )
        expanded_with_headings = json.dumps(
            {
                "content": (
                    self._expanded_en_body() + "\n\n### Extra Structure Injected\n\nMore [ref_001]."
                ),
                "citations": [],
            }
        )
        llm = MockLLM(responses=[short, expanded_with_headings])
        gen = ReportGenerator(llm)
        section = await gen._generate_section(
            query="How did Buffett evolve from Graham to quality investing?",
            title="Buffett Evolution",
            objective="Trace the shift from cigar-butt to quality-business investing.",
            sources=[self._source()],
        )
        assert llm._call_count == 2
        assert "Extra Structure Injected" not in section.content
        # Original short body preserved.
        assert "Sentence 0 about Buffett" in section.content

    async def test_ignores_regression(self):
        short = json.dumps(
            {
                "content": self._short_en_body(),
                "citations": [],
            }
        )
        regression = json.dumps({"content": "Tiny.", "citations": []})
        llm = MockLLM(responses=[short, regression])
        gen = ReportGenerator(llm)
        section = await gen._generate_section(
            query="How did Buffett evolve from Graham to quality investing?",
            title="Buffett Evolution",
            objective="Trace the shift from cigar-butt to quality-business investing.",
            sources=[self._source()],
        )
        # Regression must be ignored; original short body preserved.
        assert "Sentence 0 about Buffett" in section.content
        assert "Tiny" not in section.content
