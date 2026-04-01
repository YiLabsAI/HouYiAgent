"""ResearchPlanner — generates and refines research plans via direct LLM calls.

Decomposes a user query into 3-8 sub-questions with priorities, search
strategies, and a report outline.  Supports interactive plan editing with
optimistic concurrency (plan version check).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.types import (
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
You are a research planner. Given a user query, generate a structured research plan.

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
  "estimated_duration_min": <int>
}

Rules:
- Generate 3-5 sub-questions covering key dimensions (keep focused, avoid redundancy).
- Assign DISTINCT priorities: each question gets a unique value 1-5 (5=highest).
  If there are fewer than 5 questions, spread across the 1-5 range.
- Dependencies: if question B needs results from question A, set depends_on=[A's index].
- Outline sections map to sub-questions via related_question_ids (0-indexed).
- Default search_strategy is "web" unless context suggests otherwise.
- estimated_duration_min: rough estimate based on depth and question count.
"""

_PLAN_WITH_MEMORY_ADDENDUM = """
The following memories from past interactions may be relevant:
{memory_text}

Incorporate known facts to avoid redundant research.
"""


class ResearchPlanner:
    """Generates research plans by decomposing queries via LLM.

    Uses direct ``LLMAdapter.chat()`` calls (no tool-loop).
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
        settings = settings or ResearchSettings()
        system = _PLAN_SYSTEM_PROMPT
        if memory_context:
            system += _PLAN_WITH_MEMORY_ADDENDUM.format(memory_text=memory_context)

        user_msg = (
            f"Research query: {query}\n"
            f"Depth: {settings.depth.value}\n"
            f"Max sub-questions: 8\n"
            f"Respond ONLY with the JSON object."
        )

        resp = await self._llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            **self._llm_kwargs,
        )

        plan_data = _parse_json_response(resp.content)
        return _build_plan(query, plan_data, settings)

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


def _parse_json_response(content: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse plan JSON, returning empty structure")
        return {"sub_questions": [], "outline": [], "estimated_duration_min": 5}


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


def _apply_edit(plan: ResearchPlan, edit: PlanEdit) -> None:
    """Mutate *plan* in-place according to a single PlanEdit."""
    handler = _EDIT_HANDLERS.get(edit.op)
    if handler:
        handler(plan, edit)


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
