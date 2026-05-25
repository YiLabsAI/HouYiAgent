"""Per-section post-processing pipeline for the report generation subsystem.

Owns the fixed sequence of content-mutating and language-enforcement
steps that each ReportSection goes through after the writer LLM
returns.  Factored out of report.py so steps can be unit-tested
and replayed on saved sections without invoking the full
ReportGenerator.

Design
------
* **Context** (SectionPostProcessContext) carries the per-section
  inputs plus the LLM handle and decoding kwargs the stateful steps need.
  Putting the LLM dependency into the context keeps every step a plain
  async def — no closure factories — so the pipeline is a flat list
  of functions with a uniform signature.
* **Step** (SectionPostProcessStep) is an alias for
  Callable[[ReportSection, SectionPostProcessContext], Awaitable[ReportSection]].
  A step mutates and returns the section; it is free to short-circuit.
* **Runner** (postprocess_section) iterates through the supplied
  step sequence.  Default ordering is fixed in
  DEFAULT_SECTION_POSTPROCESS_STEPS.

The default step order is load-bearing:
  1. clean_noise_step rewrites noisy paragraphs via targeted
     micro-LLM calls; all subsequent steps rely on a reasonably clean
     paragraph structure.
  2. consolidate_paragraphs_step normalises paragraph layout
     (merge shorts, split giants).
  3. deduplicate_subheadings_step collapses duplicate ### trees
     emitted by writer pass-2 regressions.
  4. deduplicate_paragraphs_step drops verbatim paragraph repeats
     that remain after subheading dedup.
  5. prune_empty_subheadings_step drops ### blocks whose body
     is empty or a dangling colon-only intro.  Must run after dedup
     so the empty-heading detector sees the final set of headings,
     and before strip_writer_leaks_step so downstream structural
     analysis never sees the orphan shells.
  6. strip_writer_leaks_step removes writer/tool-layer leaks
     (JSON envelope fragments, junk tokens, orphan reference IDs).
  7. enforce_query_language_step aligns the body + title language
     with the query language (currently implements the CJK-to-English
     direction).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import ReportSection

# The per-section helper bindings (_consolidate_short_paragraphs,
# _parse_section, _TRANSLATE_PROMPT ...) live in
# houyi.application.research.report.  Importing them at module
# load would form a circular dependency with report.py (which
# imports postprocess_section and friends from this module to run
# the pipeline).  The standard Python remedy is to defer those
# imports to the call sites that need them — every step body below
# starts with a small from ...report import ... line that lists
# exactly the helpers that step needs.  Python caches module imports
# in sys.modules so the runtime cost is a single dict lookup per
# call after the first.

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SECTION_POSTPROCESS_STEPS",
    "SectionPostProcessContext",
    "SectionPostProcessStep",
    "clean_noise_step",
    "consolidate_paragraphs_step",
    "deduplicate_paragraphs_step",
    "deduplicate_subheadings_step",
    "enforce_query_language_step",
    "postprocess_section",
    "prune_empty_subheadings_step",
    "strip_writer_leaks_step",
]


# ---------------------------------------------------------------------------
# Context and step type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectionPostProcessContext:
    """Inputs + environment shared by every step in the pipeline.

    query / title / objective / available_refs describe
    the section being processed.  llm / llm_kwargs /
    section_max_tokens are the LLM handle used by the two
    LLM-backed steps (clean_noise_step and
    enforce_query_language_step); the four deterministic text-only
    steps ignore them.
    """

    query: str
    title: str
    objective: str
    available_refs: list[str]
    llm: LLMAdapter
    llm_kwargs: dict[str, Any]
    section_max_tokens: int


SectionPostProcessStep = Callable[
    [ReportSection, SectionPostProcessContext],
    Awaitable[ReportSection],
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def postprocess_section(
    section: ReportSection,
    ctx: SectionPostProcessContext,
    steps: Sequence[SectionPostProcessStep] | None = None,
) -> ReportSection:
    """Run the post-processing pipeline on a single section.

    steps defaults to DEFAULT_SECTION_POSTPROCESS_STEPS.
    Each step receives the result of the previous step; the final
    section is returned.  The function does not catch step-level
    exceptions — callers are expected to decide whether to raise or
    degrade, matching the pre-refactor behaviour of the inline chain.
    """

    pipeline = DEFAULT_SECTION_POSTPROCESS_STEPS if steps is None else steps
    for step in pipeline:
        section = await step(section, ctx)
    return section


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


async def clean_noise_step(
    section: ReportSection,
    ctx: SectionPostProcessContext,
) -> ReportSection:
    """Detect and micro-rewrite noisy paragraphs in the section body.

    Detection is pure regex/heuristic (zero LLM cost).  Only flagged
    paragraphs trigger a targeted LLM rewrite, so latency stays bounded
    at roughly one rewrite call per noisy paragraph.  Rewritten bodies
    that come back empty remove the paragraph entirely; the joined
    output runs through _normalize_citation_groups as a
    defence-in-depth pass so any direct callers (tests, debug scripts)
    still get atomic [ref_xxx] citations.
    """

    from houyi.application.research.report import (
        _detect_noisy_paragraphs,
        _normalize_citation_groups,
    )

    content = section.content
    paragraphs = content.split("\n\n")
    if not paragraphs:
        return section
    noisy_indices = _detect_noisy_paragraphs(paragraphs)
    if not noisy_indices:
        return section
    ref_str = ", ".join(ctx.available_refs[:20])
    for idx in noisy_indices:
        paragraphs[idx] = await _rewrite_noisy_paragraph_impl(
            paragraphs[idx],
            title=ctx.title,
            objective=ctx.objective,
            available_refs=ref_str,
            llm=ctx.llm,
            llm_kwargs=ctx.llm_kwargs,
        )
    joined = "\n\n".join(p for p in paragraphs if p.strip())
    section.content = _normalize_citation_groups(joined)
    return section


async def consolidate_paragraphs_step(
    section: ReportSection,
    ctx: SectionPostProcessContext,
) -> ReportSection:
    """Normalise paragraph layout: merge shorts, split giants.

    The only content-mutating human-visible layout repair in the
    pipeline.  Stateless, deterministic, idempotent.
    """

    from houyi.application.research.report import _consolidate_short_paragraphs

    del ctx  # unused
    section.content = _consolidate_short_paragraphs(section.content)
    return section


async def deduplicate_subheadings_step(
    section: ReportSection,
    ctx: SectionPostProcessContext,
) -> ReportSection:
    """Collapse duplicate ### Subheading blocks in a single body.

    Triggered by the writer pass-2 regression where the entire
    subheading tree is emitted twice.  Must run before
    deduplicate_paragraphs_step because dropping the pass-2
    tree first lets the paragraph pass catch the bridging paragraph
    that remains attached to the pass-1 tail.
    """

    from houyi.application.research.report import _deduplicate_subheadings

    del ctx  # unused
    section.content = _deduplicate_subheadings(section.content)
    return section


async def deduplicate_paragraphs_step(
    section: ReportSection,
    ctx: SectionPostProcessContext,
) -> ReportSection:
    """Remove paragraphs that appear verbatim more than once.

    Paragraph-granularity companion to
    deduplicate_subheadings_step — drops the long verbatim
    transition paragraph that bridged the pass-1 → pass-2 trees and
    survives subheading dedup.
    """

    from houyi.application.research.report import _deduplicate_paragraphs

    del ctx  # unused
    section.content = _deduplicate_paragraphs(section.content)
    return section


async def prune_empty_subheadings_step(
    section: ReportSection,
    ctx: SectionPostProcessContext,
) -> ReportSection:
    """Drop orphan ### Subheading blocks with empty or colon-only bodies.

    The writer occasionally emits headings with no body, or with a
    single "here follows a list:" sentence and then no list.  Both
    shapes hit multiple RACE criteria at once (information depth,
    structural clarity, complete coverage) so keeping them costs
    more than dropping them.  Runs after the dedup passes so the
    empty-heading detector sees the final set of surviving ###
    blocks; runs before the writer-leak scrubber so downstream
    structural analysis never sees the orphan heading shells.
    """

    from houyi.application.research.report import _prune_empty_subheadings

    del ctx  # unused
    section.content = _prune_empty_subheadings(section.content)
    return section


async def strip_writer_leaks_step(
    section: ReportSection,
    ctx: SectionPostProcessContext,
) -> ReportSection:
    """Strip writer / tool-layer leaks that escaped upstream parsers.

    Covers three leak classes that outran the per-paragraph unwrap:
    multiline {"content": "..."} envelope residue, broken fenced
    blocks dominated by ref_<hex> / sync / 30s junk tokens,
    and orphan reference IDs in prose.  Deterministic, language-
    agnostic, idempotent on already-clean prose.
    """

    from houyi.application.research.report import _scrub_generation_artifacts

    del ctx  # unused
    section.content = _scrub_generation_artifacts(section.content)
    return section


async def enforce_query_language_step(
    section: ReportSection,
    ctx: SectionPostProcessContext,
) -> ReportSection:
    """Align the section body + title language with the query language.

    Intent is direction-agnostic: whatever language the query is in,
    the rendered section should match.  The current implementation
    covers the CJK-heavy body → English direction, which is the only
    observed regression so far.  When future query coverage expands
    (e.g. Japanese, Korean), add dispatch on ctx.query language
    here.

    Best-effort: a failed LLM call or a translated body that did not
    reduce the CJK ratio leaves the section unchanged.
    """

    from houyi.application.research.report import _query_is_english

    if not _query_is_english(ctx.query):
        return section
    return await _maybe_translate_cjk_to_english(
        section,
        title=ctx.title,
        llm=ctx.llm,
        llm_kwargs=ctx.llm_kwargs,
        section_max_tokens=ctx.section_max_tokens,
    )


# ---------------------------------------------------------------------------
# Default pipeline
# ---------------------------------------------------------------------------


DEFAULT_SECTION_POSTPROCESS_STEPS: tuple[SectionPostProcessStep, ...] = (
    clean_noise_step,
    consolidate_paragraphs_step,
    deduplicate_subheadings_step,
    deduplicate_paragraphs_step,
    prune_empty_subheadings_step,
    strip_writer_leaks_step,
    enforce_query_language_step,
)


# ---------------------------------------------------------------------------
# Internals shared across steps
# ---------------------------------------------------------------------------


async def _rewrite_noisy_paragraph_impl(
    paragraph: str,
    *,
    title: str,
    objective: str,
    available_refs: str,
    llm: LLMAdapter,
    llm_kwargs: dict[str, Any],
) -> str:
    """Targeted LLM rewrite for a single noisy paragraph.

    Free-function companion to clean_noise_step so the
    orchestration logic (split → detect → rewrite → join) stays in
    the step function without smuggling self into a step.
    ReportGenerator._rewrite_noisy_paragraph is a thin delegating
    wrapper around this function for test back-compat.
    """

    from houyi.application.research.report import (
        _NOISE_REWRITE_PROMPT,
        _normalize_citation_groups,
        _strip_content_envelope,
    )

    prompt = _NOISE_REWRITE_PROMPT.format(
        title=title,
        objective=objective,
        available_refs=available_refs,
        paragraph=paragraph,
    )
    try:
        resp = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            **llm_kwargs,
        )
        result = resp.content.strip()
        if result.lower() in ("(empty)", ""):
            return ""
        # Strip {"content": "..."} envelopes the rewrite LLM
        # occasionally emits.  Without this the wrapper leaked into
        # the paragraph and escaped mermaid fences inside the JSON
        # string classified the whole paragraph as structural,
        # bypassing downstream paragraph-structure normalisation.
        result = _strip_content_envelope(result)
        # Normalise comma-grouped citations at the earliest point so
        # every downstream consumer sees atomic [ref_x] tokens.
        return _normalize_citation_groups(result)
    except Exception:
        logger.warning("Noise rewrite failed for section '%s'", title, exc_info=True)
        return paragraph


async def _maybe_translate_cjk_to_english(
    section: ReportSection,
    *,
    title: str,
    llm: LLMAdapter,
    llm_kwargs: dict[str, Any],
    section_max_tokens: int,
) -> ReportSection:
    """Translate a CJK-heavy section into English in-place.

    Runs when either the section body or the section title carries
    CJK on an English query.  Body triggers on the
    _EN_SECTION_CJK_RATIO_MAX threshold; title triggers whenever
    any CJK character is present (titles are short enough that a
    single CJK char is a regression against leaderboard heading
    rendering).  Body and title are translated by the same LLM call
    so the rendered ## <title> and body stay language-consistent.

    Returns the original section unchanged when the translation LLM
    call fails or when the translated body still contains more CJK
    than the original (defensive guard against a misbehaving model).
    The citation list is preserved — only the body and title are
    replaced.
    """

    from houyi.application.research.report import (
        _EN_SECTION_CJK_RATIO_MAX,
        _TRANSLATE_PROMPT,
        _cjk_char_ratio,
        _extract_translated_title,
        _parse_section,
    )

    original_ratio = _cjk_char_ratio(section.content)
    title_has_cjk = _cjk_char_ratio(title) > 0.0
    if original_ratio <= _EN_SECTION_CJK_RATIO_MAX and not title_has_cjk:
        return section
    prompt = _TRANSLATE_PROMPT.format(
        title=title,
        body=section.content,
    )
    try:
        resp = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=section_max_tokens,
            **llm_kwargs,
        )
    except Exception:
        logger.warning(
            "section_translation_failed",
            extra={
                "title": title,
                "cjk_ratio": round(original_ratio, 3),
            },
        )
        return section
    translated_title = _extract_translated_title(resp.content, fallback=title)
    translated = _parse_section(translated_title, resp.content)
    if not translated.content.strip():
        return section
    new_ratio = _cjk_char_ratio(translated.content)
    # Guard: body translation must materially reduce the CJK ratio
    # when the body was the trigger.  When only the title carries
    # CJK we still accept the pass so long as the body did not
    # regress.  Without this split, a short EN body with a CJK
    # title could not be translated.
    body_regressed = new_ratio >= original_ratio and original_ratio > _EN_SECTION_CJK_RATIO_MAX
    if body_regressed:
        logger.info(
            "section_translation_rejected_no_progress",
            extra={
                "title": title,
                "cjk_before": round(original_ratio, 3),
                "cjk_after": round(new_ratio, 3),
            },
        )
        return section
    logger.info(
        "section_translation_applied",
        extra={
            "title": title,
            "translated_title": translated_title,
            "cjk_before": round(original_ratio, 3),
            "cjk_after": round(new_ratio, 3),
        },
    )
    section.content = translated.content
    section.title = translated.title
    return section
