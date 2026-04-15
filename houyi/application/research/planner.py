"""ResearchPlanner — generates and refines research plans via direct LLM calls.

Decomposes a user query into 3-8 sub-questions with priorities, search
strategies, and a report outline.  Supports interactive plan editing with
optimistic concurrency (plan version check).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import (
    AnswerCoverageContract,
    ClarificationResult,
    CoverageFacet,
    OutlineSection,
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchPlan,
    ResearchSettings,
    SearchStrategy,
    SubQuestion,
)
from houyi.rag.llm.json_utils import parse_embedded_json

logger = logging.getLogger(__name__)
_MAX_SUB_QUESTIONS_BY_DEPTH = {"quick": 3, "standard": 5, "deep": 8}
_PLAN_MAX_TOKENS_BY_DEPTH = {"quick": 1500, "standard": 2000, "deep": 3000}
_MIN_OUTLINE_SECTIONS_BY_DEPTH = {"deep": 4}
_FOCUSED_SECTION_PREFIX = "Focused"

_PLAN_SYSTEM_PROMPT = """\
You are an expert research planner specializing in PhD-level, multi-dimensional \
research decomposition. Given a user query, generate a structured research plan.

Output STRICT JSON with:
{
  "sub_questions": [
    {
      "question": "...",
      "priority": 1-5 (5=highest),
      "search_strategy": "web"|"local_file"|"rag"|"mixed",
      "expected_sources": <int>,
      "depends_on": [],
      "coverage_contract": {
        "must_cover_facets": [
          {
            "name": "...",
            "intent": "...",
            "evidence_hint": "...",
            "bilingual_terms": ["..."]
          }
        ],
        "required_caveats": ["..."],
        "evidence_expectations": ["..."]
      }
    }
  ],
  "outline": [
    {
      "title": "Section Title",
      "objective": "What this section covers",
      "related_question_ids": []
    }
  ],
  "plan_contract": {
    "must_cover_facets": [
      {
        "name": "...",
        "intent": "...",
        "evidence_hint": "...",
        "bilingual_terms": ["..."]
      }
    ],
    "comparison_axes": ["..."],
    "time_scope": "...",
    "geo_scope": "...",
    "required_caveats": ["..."],
    "evidence_expectations": ["..."]
  },
  "estimated_duration_min": <int>,
  "clarification": {
    "needs_clarification": true/false,
    "confidence": 0.0 to 1.0,
    "issues": ["issue 1", "issue 2"],
    "suggested_questions": ["question 1"],
    "refined_query": "optional improved query"
  }
}

Rules:
- Decompose the query into sub-questions that are MECE (mutually exclusive, \
collectively exhaustive). Generate NO MORE than the max specified in the user message. Cover distinct analytical dimensions: background/context, \
mechanisms/methodology, empirical evidence, comparative analysis, limitations/debate, \
and future directions — as applicable to the query.
- Each sub-question should be SPECIFIC and SEARCHABLE (not vague). Bad: "What are \
the implications?" Good: "What empirical studies have measured the economic impact \
of X on Y in the period 2020-2025?"
- Assign DISTINCT priorities: each question gets a unique value 1-5 (5=highest). \
Foundational/definitional questions get higher priority.
- Dependencies: if question B requires information from question A to formulate \
effective searches, set depends_on=[A's 0-based index]. Use dependencies to model \
logical prerequisite relationships.
- Add one shared coverage contract for the whole plan and one local delta contract for each sub-question.
- Do NOT repeat the same global scope/caveat fields in every sub-question. Put shared scope, axes, and shared caveats in plan_contract first.
- Sub-question coverage_contract should contain only the local must-cover facets and any local caveats or evidence expectations that are unique to that sub-question.
- Outline sections should NOT include coverage_contract in the JSON. The application derives section contracts from related sub-questions.
- Coverage contracts must make downstream search/writing executable: define the must-cover facets, caveats,
  comparison axes, and evidence expectations that separate a good answer from a noisy answer.
- Facets should be answer obligations, not generic search themes. Bad facet: "search web". Good facet:
  "current role and employer", "recent open-source output", "disputed identity claims".
- Keep contracts compact: plan_contract should usually have at most 3 facets; each sub-question should usually have at most 2 local facets; each facet should have at most 2 bilingual_terms.
- Keep each intent, evidence_hint, caveat, comparison axis, and expectation to a short phrase, not a paragraph.
- Outline sections should form a coherent narrative arc. Map each section to its \
contributing sub-questions via related_question_ids (0-indexed).
- Default search_strategy is "web" unless context suggests otherwise.
- Set expected_sources realistically: 3-5 for focused factual queries, 5-10 for \
broad analytical questions.
- estimated_duration_min: rough estimate based on depth and question count.
- Set clarification.needs_clarification=true only when missing constraints or competing interpretations would materially change the research plan.
- Keep clarification.issues and clarification.suggested_questions short and concrete.
- If the query is already clear enough to plan, set clarification.needs_clarification=false and clarification.refined_query=null.
"""

_PLAN_WITH_MEMORY_ADDENDUM = """
The following memories from past interactions may be relevant:
{memory_text}

Incorporate known facts to avoid redundant research.
"""

_PLAN_RETRY_PROMPT = """
Your previous response could not be used as a research plan.

Requirements:
- Respond with exactly one valid JSON object.
- Include at least one sub-question.
- Include at least one outline section with a non-empty title and objective.
- Preserve the user's research query, depth, and max sub-question limit.

Original request:
{user_msg}

Previous response:
{previous_response}
"""


@dataclass(slots=True)
class PlannerDraft:
    plan: ResearchPlan
    clarification: ClarificationResult | None = None


class ResearchPlanner:
    """Generates research plans by decomposing queries via LLM.

    Uses ``LLMAdapter.chat()`` calls (streaming by default, no tool-loop).
    """

    def __init__(self, llm_adapter: LLMAdapter, **llm_kwargs: Any) -> None:
        self._llm = llm_adapter
        self._llm_kwargs = llm_kwargs

    async def generate_plan(
        self,
        query: str,
        settings: ResearchSettings | None = None,
        memory_context: str | None = None,
    ) -> ResearchPlan:
        """Generate a research plan from a user query.

        Args:
            query: The user's research question.
            settings: Optional overrides for depth, mode, etc.
            memory_context: Pre-formatted memory text to inject as prior knowledge.

        Returns:
            A ``ResearchPlan`` in DRAFT status.
        """
        return (
            await self.generate_plan_draft(query, settings=settings, memory_context=memory_context)
        ).plan

    async def generate_plan_draft(
        self,
        query: str,
        settings: ResearchSettings | None = None,
        memory_context: str | None = None,
    ) -> PlannerDraft:
        """Generate a research plan plus internal clarification metadata."""
        settings = settings or ResearchSettings()
        system = _PLAN_SYSTEM_PROMPT
        if memory_context:
            system += _PLAN_WITH_MEMORY_ADDENDUM.format(memory_text=memory_context)

        max_qs = _MAX_SUB_QUESTIONS_BY_DEPTH.get(settings.depth, 5)
        user_msg = (
            f"Research query: {query}\n"
            f"Depth: {settings.depth.value}\n"
            f"Max sub-questions: {max_qs}\n"
            f"Respond ONLY with the JSON object."
        )

        plan_max_tokens = _PLAN_MAX_TOKENS_BY_DEPTH.get(settings.depth, 2000)
        prompt = user_msg
        last_error = "Planner returned an invalid research plan"
        for _attempt in range(2):
            resp = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=plan_max_tokens,
                **self._llm_kwargs,
            )

            plan_data = _parse_json_response(resp.content)
            if plan_data is not None:
                plan = _build_plan(query, plan_data, settings)
                validation_error = validate_research_plan(plan)
                if validation_error is None:
                    return PlannerDraft(
                        plan=plan,
                        clarification=_extract_clarification(plan_data),
                    )
                last_error = validation_error
            else:
                last_error = "Planner returned invalid JSON"

            logger.warning("Planner response unusable: %s", last_error)
            prompt = _PLAN_RETRY_PROMPT.format(
                user_msg=user_msg,
                previous_response=resp.content[:4000],
            )

        raise ValueError(last_error)

    async def refine_plan(
        self,
        plan: ResearchPlan,
        edits: list[PlanEdit],
        expected_version: int | None = None,
    ) -> ResearchPlan:
        """Apply user edits to an existing plan and bump the version.

        Args:
            plan: Current plan to edit.
            edits: List of edit operations.
            expected_version: If provided, must match ``plan.version`` for
                optimistic concurrency control. Raises ``ValueError`` on mismatch.

        Raises:
            ValueError: Plan is not editable or version conflict detected.
        """
        if plan.status not in (PlanStatus.DRAFT, PlanStatus.CONFIRMED):
            msg = f"Cannot edit plan in status {plan.status.value}"
            raise ValueError(msg)

        if expected_version is not None and expected_version != plan.version:
            msg = f"Plan version conflict: expected {expected_version}, actual {plan.version}"
            raise ValueError(msg)

        updated = plan.model_copy(deep=True)
        for edit in edits:
            _apply_edit(updated, edit)

        updated.version += 1
        updated.status = PlanStatus.DRAFT
        return updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_json_response(content: str) -> dict | None:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = _normalize_plan_json_text(content)
    try:
        data = parse_embedded_json(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        repaired = _remove_trailing_commas(text)
        if repaired != text:
            try:
                data = parse_embedded_json(repaired)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass

    logger.warning(
        "Failed to parse plan JSON len=%d preview=%r",
        len(text),
        _truncate_plan_preview(text),
    )
    return None


def _normalize_plan_json_text(content: str) -> str:
    text = content.strip()
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _remove_trailing_commas(content: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", content)


def _truncate_plan_preview(content: str, limit: int = 2000) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "...<truncated>"


def _build_plan(
    query: str,
    data: dict,
    settings: ResearchSettings,
) -> ResearchPlan:
    """Convert raw LLM JSON into a typed ResearchPlan."""
    plan_contract = _parse_coverage_contract(data.get("plan_contract"))
    sub_questions: list[SubQuestion] = []
    local_contracts: list[AnswerCoverageContract] = []
    for i, sq in enumerate(data.get("sub_questions", [])):
        strategy = sq.get("search_strategy", "web")
        if strategy not in SearchStrategy.__members__.values():
            strategy = "web"
        deps = sq.get("depends_on", [])
        dep_ids = [sub_questions[d].question_id for d in deps if d < len(sub_questions)]
        local_contract = _parse_coverage_contract(sq.get("coverage_contract"))
        local_contracts.append(local_contract)
        sub_questions.append(
            SubQuestion(
                question=sq.get("question", f"Sub-question {i + 1}"),
                priority=max(1, min(5, int(sq.get("priority", 3)))),
                search_strategy=SearchStrategy(strategy),
                expected_sources=int(sq.get("expected_sources", 5)),
                depends_on=dep_ids,
                coverage_contract=_materialize_sub_question_contract(plan_contract, local_contract),
            )
        )

    outline: list[OutlineSection] = []
    for sec in data.get("outline", []):
        related = sec.get("related_question_ids", [])
        qids = [sub_questions[r].question_id for r in related if r < len(sub_questions)]
        related_local_contracts = [local_contracts[r] for r in related if r < len(local_contracts)]
        outline.append(
            OutlineSection(
                title=sec.get("title", ""),
                objective=sec.get("objective", ""),
                related_question_ids=qids,
                coverage_contract=_derive_section_coverage_contract(
                    plan_contract,
                    related_local_contracts,
                    section_title=sec.get("title", ""),
                    objective=sec.get("objective", ""),
                ),
            )
        )
    outline = _expand_outline_for_depth(
        outline,
        sub_questions=sub_questions,
        local_contracts=local_contracts,
        plan_contract=plan_contract,
        settings=settings,
    )

    return ResearchPlan(
        query=query,
        sub_questions=sub_questions,
        outline=outline,
        settings=settings,
        estimated_duration_min=int(data.get("estimated_duration_min", 5)),
        created_at=time.time(),
        status=PlanStatus.DRAFT,
        plan_contract=plan_contract,
    )


def _parse_coverage_contract(raw: Any) -> AnswerCoverageContract:
    if not isinstance(raw, dict):
        return AnswerCoverageContract()

    facets: list[CoverageFacet] = []
    for item in raw.get("must_cover_facets", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        bilingual_terms = [
            str(term).strip() for term in item.get("bilingual_terms", []) if str(term).strip()
        ]
        facets.append(
            CoverageFacet(
                name=name,
                intent=str(item.get("intent", "")).strip(),
                evidence_hint=str(item.get("evidence_hint", "")).strip(),
                bilingual_terms=bilingual_terms[:6],
            )
        )

    return AnswerCoverageContract(
        must_cover_facets=facets,
        comparison_axes=_compact_text_list(raw.get("comparison_axes"), limit=6),
        time_scope=str(raw.get("time_scope", "")).strip(),
        geo_scope=str(raw.get("geo_scope", "")).strip(),
        required_caveats=_compact_text_list(raw.get("required_caveats"), limit=6),
        evidence_expectations=_compact_text_list(raw.get("evidence_expectations"), limit=6),
    )


def _compact_text_list(raw: Any, *, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    items = [str(item).strip() for item in raw if str(item).strip()]
    return items[:limit]


def _materialize_sub_question_contract(
    shared: AnswerCoverageContract,
    local: AnswerCoverageContract,
) -> AnswerCoverageContract:
    return AnswerCoverageContract(
        must_cover_facets=_merge_facets(local.must_cover_facets, shared.must_cover_facets),
        comparison_axes=_merge_text_lists(shared.comparison_axes, local.comparison_axes, limit=6),
        time_scope=local.time_scope or shared.time_scope,
        geo_scope=local.geo_scope or shared.geo_scope,
        required_caveats=_merge_text_lists(
            shared.required_caveats,
            local.required_caveats,
            limit=6,
        ),
        evidence_expectations=_merge_text_lists(
            shared.evidence_expectations,
            local.evidence_expectations,
            limit=6,
        ),
    )


def _derive_section_coverage_contract(
    plan_contract: AnswerCoverageContract,
    related_contracts: list[AnswerCoverageContract],
    *,
    section_title: str,
    objective: str,
) -> AnswerCoverageContract:
    local_facets: list[CoverageFacet] = []
    merged = AnswerCoverageContract(
        must_cover_facets=[],
        comparison_axes=list(plan_contract.comparison_axes),
        time_scope=plan_contract.time_scope,
        geo_scope=plan_contract.geo_scope,
        required_caveats=list(plan_contract.required_caveats),
        evidence_expectations=list(plan_contract.evidence_expectations),
    )
    for contract in related_contracts:
        local_facets = _merge_facets(local_facets, contract.must_cover_facets)
        merged.comparison_axes = _merge_text_lists(
            merged.comparison_axes,
            contract.comparison_axes,
            limit=6,
        )
        merged.required_caveats = _merge_text_lists(
            merged.required_caveats,
            contract.required_caveats,
            limit=6,
        )
        merged.evidence_expectations = _merge_text_lists(
            merged.evidence_expectations,
            contract.evidence_expectations,
            limit=6,
        )
        merged.time_scope = contract.time_scope or merged.time_scope
        merged.geo_scope = contract.geo_scope or merged.geo_scope
    shared_facets = _select_relevant_shared_facets(
        plan_contract,
        related_contracts,
        section_title=section_title,
        objective=objective,
    )
    merged.must_cover_facets = _merge_facets(local_facets, shared_facets)
    if not merged.must_cover_facets:
        merged.must_cover_facets = list(plan_contract.must_cover_facets[:2])
    merged.must_cover_facets = merged.must_cover_facets[:6]
    return merged


def _select_relevant_shared_facets(
    plan_contract: AnswerCoverageContract,
    related_contracts: list[AnswerCoverageContract],
    *,
    section_title: str,
    objective: str,
    limit: int = 2,
) -> list[CoverageFacet]:
    if not plan_contract.must_cover_facets:
        return []
    section_keywords = _extract_contract_keywords(f"{section_title} {objective}")
    for contract in related_contracts:
        for facet in contract.must_cover_facets:
            section_keywords |= _extract_contract_keywords(
                f"{facet.name} {facet.intent} {facet.evidence_hint} {' '.join(facet.bilingual_terms)}"
            )
    scored: list[tuple[int, int, CoverageFacet]] = []
    for index, facet in enumerate(plan_contract.must_cover_facets):
        facet_keywords = _extract_contract_keywords(
            f"{facet.name} {facet.intent} {facet.evidence_hint} {' '.join(facet.bilingual_terms)}"
        )
        overlap = len(section_keywords & facet_keywords)
        scored.append((overlap, -index, facet))
    scored.sort(reverse=True)
    selected = [facet for overlap, _, facet in scored if overlap > 0][:limit]
    if selected:
        return selected
    return list(plan_contract.must_cover_facets[:1])


@dataclass(slots=True)
class _OutlineExpansionContext:
    qid_to_question: dict[str, SubQuestion]
    qid_to_local: dict[str, AnswerCoverageContract]
    plan_contract: AnswerCoverageContract


@dataclass(slots=True)
class _OutlineExpansionState:
    expanded: list[OutlineSection]
    seen_titles: set[str]
    individually_covered: set[str]


def _expand_outline_for_depth(
    outline: list[OutlineSection],
    *,
    sub_questions: list[SubQuestion],
    local_contracts: list[AnswerCoverageContract],
    plan_contract: AnswerCoverageContract,
    settings: ResearchSettings,
) -> list[OutlineSection]:
    """Expand compressed deep outlines into focused sections without another LLM round.

    Planner output stays compact, but deep-mode execution needs enough section
    breadth to preserve a coherent narrative arc and keep distinct question
    obligations visible during report generation.
    """

    depth = _depth_key(settings)
    if depth not in _MIN_OUTLINE_SECTIONS_BY_DEPTH or not sub_questions:
        return outline
    target = _outline_section_target(depth, sub_questions)
    if len(outline) >= target:
        return outline

    context = _build_outline_expansion_context(sub_questions, local_contracts, plan_contract)
    state = _build_outline_expansion_state(outline)
    _expand_outline_from_related_sections(outline, target=target, context=context, state=state)
    _expand_outline_from_remaining_questions(
        sub_questions, target=target, context=context, state=state
    )
    return state.expanded


def _depth_key(settings: ResearchSettings) -> str:
    return settings.depth.value if hasattr(settings.depth, "value") else str(settings.depth)


def _outline_section_target(depth: str, sub_questions: list[SubQuestion]) -> int:
    return min(
        max(len(sub_questions) + 1, _MIN_OUTLINE_SECTIONS_BY_DEPTH[depth]),
        _MAX_SUB_QUESTIONS_BY_DEPTH.get(depth, len(sub_questions) + 1),
    )


def _build_outline_expansion_context(
    sub_questions: list[SubQuestion],
    local_contracts: list[AnswerCoverageContract],
    plan_contract: AnswerCoverageContract,
) -> _OutlineExpansionContext:
    return _OutlineExpansionContext(
        qid_to_question={sq.question_id: sq for sq in sub_questions},
        qid_to_local={
            sq.question_id: local_contracts[idx]
            for idx, sq in enumerate(sub_questions)
            if idx < len(local_contracts)
        },
        plan_contract=plan_contract,
    )


def _build_outline_expansion_state(outline: list[OutlineSection]) -> _OutlineExpansionState:
    return _OutlineExpansionState(
        expanded=list(outline),
        seen_titles={sec.title.strip().lower() for sec in outline if sec.title.strip()},
        individually_covered={
            qid
            for sec in outline
            if len(sec.related_question_ids) == 1
            for qid in sec.related_question_ids
        },
    )


def _expand_outline_from_related_sections(
    outline: list[OutlineSection],
    *,
    target: int,
    context: _OutlineExpansionContext,
    state: _OutlineExpansionState,
) -> None:
    for sec in outline:
        if len(state.expanded) >= target or len(sec.related_question_ids) <= 1:
            continue
        for qid in sec.related_question_ids:
            if len(state.expanded) >= target:
                return
            _try_add_focused_section(qid, context=context, state=state)


def _expand_outline_from_remaining_questions(
    sub_questions: list[SubQuestion],
    *,
    target: int,
    context: _OutlineExpansionContext,
    state: _OutlineExpansionState,
) -> None:
    for sq in sub_questions:
        if len(state.expanded) >= target:
            return
        _try_add_focused_section(sq.question_id, context=context, state=state)


def _try_add_focused_section(
    question_id: str,
    *,
    context: _OutlineExpansionContext,
    state: _OutlineExpansionState,
) -> None:
    # Promote one question into its own section when the planner collapsed
    # several questions into a single broad outline node.
    if question_id in state.individually_covered:
        return
    question = context.qid_to_question.get(question_id)
    local_contract = context.qid_to_local.get(question_id)
    if question is None or local_contract is None:
        return
    title = _resolve_focus_section_title(question.question, state.seen_titles)
    if title is None:
        return
    objective = _derive_focus_section_objective(question.question)
    state.expanded.append(
        OutlineSection(
            title=title,
            objective=objective,
            related_question_ids=[question_id],
            coverage_contract=_derive_section_coverage_contract(
                context.plan_contract,
                [local_contract],
                section_title=title,
                objective=objective,
            ),
        )
    )
    state.seen_titles.add(title.strip().lower())
    state.individually_covered.add(question_id)


def _resolve_focus_section_title(question: str, seen_titles: set[str]) -> str | None:
    title = _derive_focus_section_title(question)
    title_key = title.strip().lower()
    if title_key not in seen_titles:
        return title
    deduped = _dedupe_focus_section_title(title)
    if deduped.strip().lower() in seen_titles:
        return None
    return deduped


def _derive_focus_section_title(question: str) -> str:
    """Turn a sub-question into a short section title."""

    cleaned = re.sub(r"\s+", " ", question).strip().rstrip("?？。！!")
    if len(cleaned) <= 48:
        return cleaned
    return cleaned[:45].rstrip(" ,;:，；：") + "..."


def _dedupe_focus_section_title(title: str) -> str:
    """Add a stable English prefix when a derived title would collide."""

    return f"{_FOCUSED_SECTION_PREFIX}: {title}"


def _derive_focus_section_objective(question: str) -> str:
    """Turn a sub-question into a focused English section objective."""

    cleaned = re.sub(r"\s+", " ", question).strip().rstrip("?？。！!")
    return f"Focused analysis of {cleaned}"


def _extract_contract_keywords(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w\-]{3,}", text.lower()) if not token.isdigit()}


def _merge_text_lists(base: list[str], extra: list[str], *, limit: int) -> list[str]:
    merged: list[str] = []
    for item in [*base, *extra]:
        normalized = item.strip()
        if not normalized or normalized in merged:
            continue
        merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def _merge_facets(base: list[CoverageFacet], extra: list[CoverageFacet]) -> list[CoverageFacet]:
    merged: list[CoverageFacet] = []
    seen: set[str] = set()
    for facet in [*base, *extra]:
        key = facet.name.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(facet)
        if len(merged) >= 6:
            break
    return merged


def _extract_clarification(data: dict[str, Any]) -> ClarificationResult | None:
    raw = data.get("clarification")
    if not isinstance(raw, dict):
        return None
    confidence_raw = raw.get("confidence", 0.8)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.8
    issues = [str(item).strip() for item in raw.get("issues", []) if str(item).strip()]
    questions = [
        str(item).strip() for item in raw.get("suggested_questions", []) if str(item).strip()
    ]
    refined_query = raw.get("refined_query")
    if not isinstance(refined_query, str) or not refined_query.strip():
        refined_query = None
    return ClarificationResult(
        needs_clarification=bool(raw.get("needs_clarification", False)),
        confidence=confidence,
        issues=issues[:3],
        suggested_questions=questions[:3],
        refined_query=refined_query,
    )


def validate_research_plan(plan: ResearchPlan) -> str | None:
    if not plan.sub_questions:
        return "Planner returned no sub-questions"
    if not plan.outline:
        return "Planner returned no outline sections"
    if any(not sq.question.strip() for sq in plan.sub_questions):
        return "Planner returned a blank sub-question"
    if any(not section.title.strip() or not section.objective.strip() for section in plan.outline):
        return "Planner returned an incomplete outline section"
    return None


def _apply_edit(plan: ResearchPlan, edit: PlanEdit) -> None:
    """Mutate *plan* in-place according to a single PlanEdit."""
    handler = _EDIT_HANDLERS.get(edit.op)
    if handler:
        handler(plan, edit)


def apply_plan_edits(plan: ResearchPlan, edits: list[PlanEdit]) -> ResearchPlan:
    """Return a new plan with all edits applied and version incremented."""
    updated = plan.model_copy(deep=True)
    for edit in edits:
        _apply_edit(updated, edit)
    updated.version += 1
    updated.status = PlanStatus.DRAFT
    return updated


def _edit_add(plan: ResearchPlan, edit: PlanEdit) -> None:
    plan.sub_questions.append(
        SubQuestion(
            question=edit.target_question or "",
            priority=edit.new_priority or 3,
            search_strategy=edit.new_search_strategy or SearchStrategy.WEB,
        )
    )


def _edit_delete(plan: ResearchPlan, edit: PlanEdit) -> None:
    plan.sub_questions = [sq for sq in plan.sub_questions if sq.question_id != edit.question_id]


def _edit_update(plan: ResearchPlan, edit: PlanEdit) -> None:
    for sq in plan.sub_questions:
        if sq.question_id == edit.question_id and edit.target_question:
            sq.question = edit.target_question


def _edit_set_priority(plan: ResearchPlan, edit: PlanEdit) -> None:
    for sq in plan.sub_questions:
        if sq.question_id == edit.question_id and edit.new_priority is not None:
            sq.priority = edit.new_priority


def _edit_set_strategy(plan: ResearchPlan, edit: PlanEdit) -> None:
    for sq in plan.sub_questions:
        if sq.question_id == edit.question_id and edit.new_search_strategy is not None:
            sq.search_strategy = edit.new_search_strategy


def _edit_move(plan: ResearchPlan, edit: PlanEdit) -> None:
    sq_map = {sq.question_id: sq for sq in plan.sub_questions}
    if not (edit.question_id and edit.question_id in sq_map):
        return
    moved = sq_map[edit.question_id]
    remaining = [sq for sq in plan.sub_questions if sq.question_id != edit.question_id]
    insert_idx = len(remaining)
    if edit.after_question_id:
        for i, sq in enumerate(remaining):
            if sq.question_id == edit.after_question_id:
                insert_idx = i + 1
                break
    remaining.insert(insert_idx, moved)
    plan.sub_questions = remaining


_EDIT_HANDLERS = {
    PlanEditOperation.ADD: _edit_add,
    PlanEditOperation.DELETE: _edit_delete,
    PlanEditOperation.UPDATE: _edit_update,
    PlanEditOperation.SET_PRIORITY: _edit_set_priority,
    PlanEditOperation.SET_STRATEGY: _edit_set_strategy,
    PlanEditOperation.MOVE: _edit_move,
}
