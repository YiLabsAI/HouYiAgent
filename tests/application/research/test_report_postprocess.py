from __future__ import annotations

import json
from typing import Any

import pytest

from houyi.application.research.report_postprocess import (
    DEFAULT_SECTION_POSTPROCESS_STEPS,
    SectionPostProcessContext,
    SectionPostProcessStep,
    clean_noise_step,
    consolidate_paragraphs_step,
    deduplicate_paragraphs_step,
    deduplicate_subheadings_step,
    enforce_query_language_step,
    postprocess_section,
    prune_empty_subheadings_step,
    strip_writer_leaks_step,
)
from houyi.application.research.types import ReportSection

from .conftest import MockLLM


def _make_ctx(
    *,
    query: str = "How do sovereign funds allocate capital?",
    title: str = "Allocation",
    objective: str = "Overview.",
    available_refs: list[str] | None = None,
    llm: Any | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    section_max_tokens: int = 2000,
) -> SectionPostProcessContext:
    return SectionPostProcessContext(
        query=query,
        title=title,
        objective=objective,
        available_refs=list(available_refs or ["ref_001", "ref_002"]),
        llm=llm or MockLLM(),
        llm_kwargs=dict(llm_kwargs or {}),
        section_max_tokens=section_max_tokens,
    )


def _make_section(content: str, *, title: str = "Allocation") -> ReportSection:
    return ReportSection(title=title, content=content, citations=[])


# ---------------------------------------------------------------------------
# Default pipeline shape
# ---------------------------------------------------------------------------


class TestDefaultPipeline:
    def test_matches_expected_order(self):
        # Order is load-bearing: clean_noise runs first so downstream
        # structural passes see reasonably clean paragraphs; prune
        # runs after both dedup passes so it sees the final set of
        # surviving ``###`` blocks; language enforcement runs last so
        # it can translate the already-structurally-clean body.
        expected = (
            clean_noise_step,
            consolidate_paragraphs_step,
            deduplicate_subheadings_step,
            deduplicate_paragraphs_step,
            prune_empty_subheadings_step,
            strip_writer_leaks_step,
            enforce_query_language_step,
        )
        assert expected == DEFAULT_SECTION_POSTPROCESS_STEPS

    def test_is_immutable_tuple(self):
        # Tuple prevents accidental mutation by importers.  The length
        # check guards against a future edit accidentally dropping or
        # duplicating a step: whenever the pipeline genuinely grows,
        # this count updates in the same patch as the addition.
        assert isinstance(DEFAULT_SECTION_POSTPROCESS_STEPS, tuple)
        assert len(DEFAULT_SECTION_POSTPROCESS_STEPS) == 7


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestPostprocessSection:
    async def test_runs_steps_in_order(self):
        # A custom pipeline that records invocation order so the runner's
        # contract is verified independent of real step bodies.
        order: list[str] = []

        async def step_a(section, ctx):  # type: ignore[no-untyped-def]
            order.append("a")
            section.content += "|a"
            return section

        async def step_b(section, ctx):  # type: ignore[no-untyped-def]
            order.append("b")
            section.content += "|b"
            return section

        async def step_c(section, ctx):  # type: ignore[no-untyped-def]
            order.append("c")
            section.content += "|c"
            return section

        ctx = _make_ctx()
        section = _make_section("start")
        result = await postprocess_section(section, ctx, steps=[step_a, step_b, step_c])
        assert order == ["a", "b", "c"]
        assert result.content == "start|a|b|c"

    async def test_empty_steps_noop(self):
        ctx = _make_ctx()
        section = _make_section("unchanged body")
        result = await postprocess_section(section, ctx, steps=[])
        assert result.content == "unchanged body"
        assert result.title == "Allocation"

    async def test_none_uses_default(self):
        # Default pipeline must run when ``steps`` is omitted.  We supply
        # a body that is already clean so no step mutates it; the key
        # assertion is that ``postprocess_section`` does not raise when
        # invoking every default step.
        ctx = _make_ctx(query="How are pensions structured?")
        # Body avoids every noise-detection trigger word (no ``search``,
        # ``query``, ``retrieval``, ``results found`` etc.) and carries
        # an inline citation so the clean_noise step short-circuits.
        section = _make_section(
            "A clean paragraph with an inline citation [ref_001] that "
            "comfortably clears the minimum-length threshold used by "
            "the noise-detection heuristic, giving the pipeline a "
            "typical well-cited body to pass through unchanged."
        )
        result = await postprocess_section(section, ctx)
        assert "[ref_001]" in result.content


# ---------------------------------------------------------------------------
# Deterministic steps (no LLM involvement)
# ---------------------------------------------------------------------------


class TestConsolidateParagraphsStep:
    async def test_merges_shorts(self):
        # The step delegates to ``_consolidate_short_paragraphs``; we
        # only need to confirm the wrapper plumbs the call through.
        body = "Short line one.\n\nShort line two.\n\nShort line three."
        result = await consolidate_paragraphs_step(_make_section(body), _make_ctx())
        # Merging collapses multiple short lines into fewer paragraphs.
        assert result.content.count("\n\n") < body.count("\n\n")


class TestDeduplicateSubheadingsStep:
    async def test_removes_duplicate_tree(self):
        duplicated = (
            "### Overview\n\nFirst pass body with [ref_001].\n\n"
            "### Overview\n\nFirst pass body with [ref_001].\n\n"
        )
        result = await deduplicate_subheadings_step(_make_section(duplicated), _make_ctx())
        # Only one ``### Overview`` should remain after dedup.
        assert result.content.count("### Overview") == 1


class TestDeduplicateParagraphsStep:
    async def test_drops_verbatim_repeat(self):
        # ``_PARAGRAPH_DUP_MIN_CHARS`` is 150 in the implementation;
        # build a paragraph comfortably above that.
        paragraph = (
            "Sovereign funds allocate across diversified asset classes "
            "including equities, fixed income, real estate, and "
            "alternatives, with growing private-market exposure in the "
            "last decade [ref_001]."
        )
        body = f"{paragraph}\n\nBridging sentence.\n\n{paragraph}"
        result = await deduplicate_paragraphs_step(_make_section(body), _make_ctx())
        # Verbatim repeat drops to a single occurrence.
        assert result.content.count(paragraph) == 1


class TestPruneEmptySubheadingsStep:
    async def test_drops_fully_empty_subheading(self):
        # The live ZH session ``rr_d79ddb66a58c`` shows section 4 ending
        # with two back-to-back empty ``###`` blocks.  The prune step
        # must delete both while preserving the substantive block that
        # precedes them.
        body = (
            "### Substantive block\n\n"
            "A paragraph with enough substance to clear every threshold "
            "that the prune step uses, carrying an inline citation "
            "[ref_001] so the prose measurement sees real content.\n\n"
            "### Empty tail block\n\n"
            "### Another empty block"
        )
        result = await prune_empty_subheadings_step(_make_section(body), _make_ctx())
        assert "### Substantive block" in result.content
        assert "### Empty tail block" not in result.content
        assert "### Another empty block" not in result.content

    async def test_drops_dangling_colon(self):
        # Live pattern from session section 5: a heading that reads
        # "Items to clarify" (26 CJK chars) with a body that is a
        # 29-CJK-char intro ending in a fullwidth colon ``U+FF1A`` but
        # never followed by the promised list.  Must be pruned as a
        # dangling-intro orphan.  Both heading and body are encoded
        # via ``\uXXXX`` escapes to satisfy the repo's no-raw-CJK rule
        # while still reproducing the observed defect exactly.
        cjk_intro = (
            "\u7efc\u5408\u4e0a\u8ff0\u5206\u6790\uff0c\u4ee5\u4e0b\u95ee\u9898"
            "\u5728\u73b0\u6709\u8bc1\u636e\u57fa\u7840\u4e0a\u65e0\u6cd5\u5f97"
            "\u51fa\u786e\u5b9a\u6027\u7ed3\u8bba\uff1a"
        )
        body = (
            "### Kept block\n\n"
            "Healthy prose with citations [ref_001] extending past the "
            "substantive threshold to anchor the test.\n\n"
            f"### Dangling intro\n\n{cjk_intro}"
        )
        result = await prune_empty_subheadings_step(_make_section(body), _make_ctx())
        assert "### Kept block" in result.content
        assert "### Dangling intro" not in result.content
        assert cjk_intro not in result.content

    async def test_drops_citation_only_body(self):
        # A body that is nothing but ``[ref_xxx]`` tokens counts as
        # empty after the citation-strip pass: there is no prose the
        # reader can hold onto under the heading.
        body = (
            "### Real block\n\n"
            "Substantive paragraph with citations [ref_001] and enough "
            "prose to exercise the keep path without ambiguity.\n\n"
            "### Citation-only block\n\n"
            "[ref_002][ref_003]"
        )
        result = await prune_empty_subheadings_step(_make_section(body), _make_ctx())
        assert "### Real block" in result.content
        assert "### Citation-only block" not in result.content

    async def test_keeps_short_body(self):
        # A short body without a trailing colon carries the writer's
        # terse-but-real analysis.  Pruning it would be a false
        # positive: the rule only drops short bodies when they ALSO
        # end in a colon (dangling-intro shape).
        body = "### Concise block\n\nBrief insight without colon [ref_001]."
        result = await prune_empty_subheadings_step(_make_section(body), _make_ctx())
        assert "### Concise block" in result.content
        assert "Brief insight" in result.content

    async def test_keeps_structural_only_body(self):
        # A body that is just a compact markdown table is substantive
        # structural content.  The structural-hint escape hatch keeps
        # it despite a very short prose count.
        body = "### Comparison\n\n| A | B |\n| - | - |\n| 1 | 2 |"
        result = await prune_empty_subheadings_step(_make_section(body), _make_ctx())
        assert "### Comparison" in result.content
        assert "| A | B |" in result.content

    async def test_preserves_leading_prose(self):
        # Section lead (prose before the first ``###``) must survive
        # unchanged regardless of what happens to the heading blocks.
        body = (
            "Leading paragraph that sets up the whole section without a "
            "subheading of its own, long enough to make the role clear.\n\n"
            "### Empty orphan\n\n"
            "### Real block\n\n"
            "Paragraph with real content and a citation [ref_001]."
        )
        result = await prune_empty_subheadings_step(_make_section(body), _make_ctx())
        assert result.content.startswith("Leading paragraph")
        assert "### Empty orphan" not in result.content
        assert "### Real block" in result.content

    async def test_noop_on_clean_body(self):
        # Every heading has a substantive body → input must come back
        # byte-for-byte identical (guaranteed no-op contract).
        body = (
            "### First block\n\n"
            "Healthy prose carrying citations [ref_001] and clearing every "
            "threshold comfortably without edge-case behaviour.\n\n"
            "### Second block\n\n"
            "Another healthy prose paragraph with citations [ref_002] and "
            "analytical content the reader can engage with."
        )
        section = _make_section(body)
        result = await prune_empty_subheadings_step(section, _make_ctx())
        assert result.content == body

    async def test_is_idempotent(self):
        # Running the prune twice produces the same result as running
        # it once — the step must not destabilise an already-pruned
        # section.
        body = (
            "### Kept\n\nHealthy prose with citation [ref_001] and enough "
            "substance to survive the prune.\n\n"
            "### Empty\n\n"
        )
        once = await prune_empty_subheadings_step(_make_section(body), _make_ctx())
        twice = await prune_empty_subheadings_step(once, _make_ctx())
        assert once.content == twice.content


class TestStripWriterLeaksStep:
    async def test_drops_orphan_ref(self):
        body = (
            "Clean prose paragraph with a legitimate citation [ref_abc123de]. "
            "An orphan token ref_def456ab leaked from the citation renumberer "
            "should be removed by the scrubber."
        )
        result = await strip_writer_leaks_step(_make_section(body), _make_ctx())
        # Bracketed reference survives; bare orphan token is stripped.
        assert "[ref_abc123de]" in result.content
        assert "ref_def456ab" not in result.content


# ---------------------------------------------------------------------------
# clean_noise_step (LLM-backed)
# ---------------------------------------------------------------------------


class TestCleanNoiseStep:
    async def test_passthrough_on_clean(self):
        # A paragraph that already carries a citation and no noise
        # patterns should skip the LLM rewrite entirely.  MockLLM has
        # an empty response queue: calling it would return ``{}`` and
        # pollute the body, so surviving unchanged confirms the step
        # correctly short-circuited.  Avoid pattern trigger words like
        # ``retrieval`` / ``query`` / ``search`` in this body.
        body = (
            "Analysts observe a persistent gap between policy targets "
            "and realised allocations [ref_001], with the widest "
            "deviation in emerging-market equities and a narrower one "
            "in developed-market fixed income."
        )
        llm = MockLLM(responses=[])
        ctx = _make_ctx(llm=llm)
        result = await clean_noise_step(_make_section(body), ctx)
        assert result.content == body
        assert llm._call_count == 0  # type: ignore[attr-defined]

    async def test_rewrites_noisy(self):
        noisy_body = (
            "We searched extensively across many databases during the "
            "research process and gathered thin results worth ignoring."
        )
        rewritten = (
            "The research drew on multiple academic databases and policy "
            "publications to build the evidence base [ref_001]."
        )
        llm = MockLLM(responses=[rewritten])
        ctx = _make_ctx(llm=llm)
        result = await clean_noise_step(_make_section(noisy_body), ctx)
        # Noise rewrite replaces the paragraph with the LLM output; the
        # citation group normaliser runs as a post-pass so the body
        # should carry the atomic ``[ref_001]`` form.
        assert "[ref_001]" in result.content
        assert "searched" not in result.content.lower()


# ---------------------------------------------------------------------------
# enforce_query_language_step (LLM-backed)
# ---------------------------------------------------------------------------


class TestEnforceQueryLanguageStep:
    # ZH sample: ``zhuquan caifu jijin touzi yu quanqiu peizhi qushi``.
    _CJK_BODY = (
        "\u4e3b\u6743\u8d22\u5bcc\u57fa\u91d1\u5728\u6295\u8d44\u4e2d\u901a\u5e38"
        "\u91c7\u7528\u591a\u5143\u5316\u8d44\u4ea7\u914d\u7f6e\u7b56\u7565"
        "[ref_001]\u3002\u8fd1\u5e74\u6765\u79c1\u52df\u80a1\u6743\u4ee5\u53ca"
        "\u53e6\u7c7b\u8d44\u4ea7\u7684\u6bd4\u91cd\u663e\u8457\u4e0a\u5347"
        "[ref_002]\u3002"
    )

    async def test_skips_non_english(self):
        # A Chinese query short-circuits the step regardless of body
        # language.  Empty MockLLM response queue means any LLM call
        # would produce noise — not reaching the LLM is the assertion.
        llm = MockLLM(responses=[])
        # ZH query: ``zhuquan caifu jijin peizhi``.
        ctx = _make_ctx(
            query=("\u4e3b\u6743\u8d22\u5bcc\u57fa\u91d1\u914d\u7f6e\u8d8b\u52bf"),
            llm=llm,
        )
        section = _make_section(self._CJK_BODY)
        result = await enforce_query_language_step(section, ctx)
        assert result.content == self._CJK_BODY
        assert llm._call_count == 0  # type: ignore[attr-defined]

    async def test_skips_english_body(self):
        llm = MockLLM(responses=[])
        ctx = _make_ctx(
            query="How do sovereign funds allocate capital?",
            title="Allocation",
            llm=llm,
        )
        body = "Sovereign funds allocate across asset classes [ref_001]."
        result = await enforce_query_language_step(_make_section(body), ctx)
        assert result.content == body
        assert llm._call_count == 0  # type: ignore[attr-defined]

    async def test_translates_cjk_body(self):
        translated = json.dumps(
            {
                "title": "Sovereign Fund Allocation",
                "content": (
                    "Sovereign wealth funds typically adopt "
                    "diversified allocation strategies [ref_001]."
                ),
                "citations": [],
            }
        )
        llm = MockLLM(responses=[translated])
        ctx = _make_ctx(
            query="How do sovereign funds allocate capital?",
            title="Allocation",
            llm=llm,
        )
        result = await enforce_query_language_step(_make_section(self._CJK_BODY), ctx)
        # Translation landed: body is English, title updated.
        assert "Sovereign wealth funds" in result.content
        assert result.title == "Sovereign Fund Allocation"

    async def test_rejects_no_progress(self):
        # LLM returns another CJK body: the defensive guard keeps the
        # original body unchanged.
        still_cjk = json.dumps({"content": self._CJK_BODY, "citations": []})
        llm = MockLLM(responses=[still_cjk])
        ctx = _make_ctx(
            query="How do sovereign funds allocate capital?",
            llm=llm,
        )
        result = await enforce_query_language_step(_make_section(self._CJK_BODY), ctx)
        assert result.content == self._CJK_BODY


# ---------------------------------------------------------------------------
# Offline-replay shape: can a ReportSection built outside the runtime
# flow be pushed through the default pipeline?  This is the fixture
# future bench-replay scripts will rely on.
# ---------------------------------------------------------------------------


class TestOfflineReplay:
    async def test_runs_on_handcrafted(self):
        body = (
            "### Overview\n\n"
            "Sovereign funds diversify across asset classes [ref_001].\n\n"
            "### Overview\n\n"
            "Sovereign funds diversify across asset classes [ref_001].\n\n"
            "### Risk profile\n\n"
            "Concentration risk remains material in commodity exporters "
            "[ref_002]."
        )
        ctx = _make_ctx(
            query="How do sovereign funds allocate capital?",
            llm=MockLLM(responses=[]),
        )
        section = _make_section(body)
        result = await postprocess_section(section, ctx)
        # Duplicate subheading tree collapses to a single occurrence.
        assert result.content.count("### Overview") == 1
        # The surviving ``### Risk profile`` is untouched.
        assert "### Risk profile" in result.content


# Type-level: the module exports a usable Callable alias.  This guards
# against a future edit that accidentally breaks the public ``Step``
# contract by, e.g., removing the ``ctx`` parameter.
def test_step_alias_is_callable():
    async def fake_step(section, ctx):  # type: ignore[no-untyped-def]
        return section

    step: SectionPostProcessStep = fake_step
    assert callable(step)


# Note: module-level ``pytest.mark.asyncio`` is intentionally omitted.
# The project's pytest config uses ``asyncio_mode = auto``, so async
# tests run under the asyncio event loop automatically, while sync
# tests (e.g. ``TestDefaultPipeline``) stay plain.

_ = pytest  # keep the import anchored; the module imports pytest for
#              fixtures in sibling helper files without using it here.
