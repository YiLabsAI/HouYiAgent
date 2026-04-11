"""ResearchPlanner — generates and refines research plans via direct LLM calls.

Decomposes a user query into 3-8 sub-questions with priorities, search
strategies, and a report outline.  Supports interactive plan editing with
optimistic concurrency (plan version check).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import (
    ClarificationResult,
    OutlineSection,
    PlanEdit,
    PlanEditOperation,
    PlanStatus,
    ResearchPlan,
    ResearchSettings,
    SearchStrategy,
    SubQuestion,
)

logger = logging.getLogger(__name__)

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
      "depends_on": []
    }
  ],
  "outline": [
    {
      "title": "Section Title",
      "objective": "What this section covers",
      "related_question_ids": []
    }
  ],
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

        max_qs = {"quick": 3, "standard": 5, "deep": 8}.get(settings.depth, 5)
        user_msg = (
            f"Research query: {query}\n"
            f"Depth: {settings.depth.value}\n"
            f"Max sub-questions: {max_qs}\n"
            f"Respond ONLY with the JSON object."
        )

        plan_max_tokens = {"quick": 1500, "standard": 2000, "deep": 3000}.get(settings.depth, 2000)
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
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse plan JSON")
    return None


def _build_plan(
    query: str,
    data: dict,
    settings: ResearchSettings,
) -> ResearchPlan:
    """Convert raw LLM JSON into a typed ResearchPlan."""
    sub_questions: list[SubQuestion] = []
    for i, sq in enumerate(data.get("sub_questions", [])):
        strategy = sq.get("search_strategy", "web")
        if strategy not in SearchStrategy.__members__.values():
            strategy = "web"
        deps = sq.get("depends_on", [])
        dep_ids = [sub_questions[d].question_id for d in deps if d < len(sub_questions)]
        sub_questions.append(
            SubQuestion(
                question=sq.get("question", f"Sub-question {i + 1}"),
                priority=max(1, min(5, int(sq.get("priority", 3)))),
                search_strategy=SearchStrategy(strategy),
                expected_sources=int(sq.get("expected_sources", 5)),
                depends_on=dep_ids,
            )
        )

    outline: list[OutlineSection] = []
    for sec in data.get("outline", []):
        related = sec.get("related_question_ids", [])
        qids = [sub_questions[r].question_id for r in related if r < len(sub_questions)]
        outline.append(
            OutlineSection(
                title=sec.get("title", ""),
                objective=sec.get("objective", ""),
                related_question_ids=qids,
            )
        )

    return ResearchPlan(
        query=query,
        sub_questions=sub_questions,
        outline=outline,
        settings=settings,
        estimated_duration_min=int(data.get("estimated_duration_min", 5)),
        created_at=time.time(),
        status=PlanStatus.DRAFT,
    )


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
