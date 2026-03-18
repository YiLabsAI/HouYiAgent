from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from houyi.application.context.context_lifecycle import (
    ContextLifecycleHookService,
    stage_span,
)
from houyi.application.context.context_planner import ContextPlanner
from houyi.application.context.context_recovery import (
    ContextRecoveryPolicy,
    RenderRecoveryPolicy,
)
from houyi.application.context.context_renderer import ContextRenderer
from houyi.application.context.context_selection import (
    build_default_context_selection_policy,
)
from houyi.application.context.context_sources import (
    assemble_context_candidates,
    build_pinned_context_candidates,
    build_tool_summary_candidates,
    extract_latest_compaction_summary,
)
from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.context.types import ContextSelectionPolicy


@dataclass(slots=True)
class ContextRequestBuildInput:
    model: str
    system_instructions: str
    history_messages: list[dict[str, Any]]
    conversation_messages: list[Any]
    conversation_metadata: dict[str, Any] | None
    memory_text: str | None
    span: Any
    input_budget: int | None = None
    truncation_log_label: str | None = "chat_send"


@dataclass(slots=True)
class ContextRequestBuildResult:
    llm_messages: list[dict[str, Any]]
    context_usage: dict[str, Any]
    history_message_count: int


@dataclass(slots=True)
class ContextRequestSourceInput:
    source: Any
    model: str
    system_instructions: str
    span: Any
    input_budget: int | None = None
    truncation_log_label: str | None = None


@dataclass(slots=True)
class SummaryEligibilityDecision:
    include: bool
    reason: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class SummarySemanticJudgeInput:
    summary_text: str
    latest_user_text: str
    history_messages: list[dict[str, Any]]
    base_score: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class SummarySemanticJudgeDecision:
    include: bool
    reason: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class _PreparedContextRequest:
    estimator: TokenEstimator
    history_messages: list[dict[str, Any]]
    candidates: list[Any]
    boundary_id: str
    summary_decision: SummaryEligibilityDecision


class ContextRequestBuilder:
    """Builds request-scoped prompt context from neutral inputs.

    This object owns the reusable context assembly / planning / rendering /
    recovery flow for a single request. Product adapters are expected to
    project chat-specific objects into `ContextRequestBuildInput` before
    calling it.
    """

    def __init__(
        self,
        *,
        hook_service: ContextLifecycleHookService | None = None,
        recovery_policy: ContextRecoveryPolicy | None = None,
        render_recovery_policy: RenderRecoveryPolicy | None = None,
        selection_policy: ContextSelectionPolicy | None = None,
        build_history_messages: Callable[[Any, str], list[dict[str, Any]]] | None = None,
        get_conversation_messages: Callable[[Any], list[Any]] | None = None,
        get_conversation_metadata: Callable[[Any], dict[str, Any] | None] | None = None,
        get_memory_text: Callable[[Any], str | None] | None = None,
        sanitize_history_messages: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
        | None = None,
        summary_semantic_judge: Callable[
            [SummarySemanticJudgeInput], SummarySemanticJudgeDecision | None
        ]
        | None = None,
    ) -> None:
        self._hook_service = hook_service or ContextLifecycleHookService()
        self._recovery_policy = recovery_policy or ContextRecoveryPolicy()
        self._render_recovery_policy = render_recovery_policy or RenderRecoveryPolicy()
        self._selection_policy = selection_policy or build_default_context_selection_policy()
        self._build_history_messages = build_history_messages
        self._get_conversation_messages = get_conversation_messages or (
            lambda source: getattr(source, "messages", [])
        )
        self._get_conversation_metadata = get_conversation_metadata or (
            lambda source: getattr(source, "metadata", None)
        )
        self._get_memory_text = get_memory_text
        self._sanitize_history_messages = sanitize_history_messages
        self._summary_semantic_judge = summary_semantic_judge

    def build(self, request: ContextRequestBuildInput) -> ContextRequestBuildResult:
        prepared = self._prepare_request(request)
        planner = ContextPlanner(
            token_estimator=prepared.estimator,
            system_instructions=request.system_instructions,
        )
        renderer = ContextRenderer()
        history_messages = prepared.history_messages
        request.span.set_attribute("chat.context.history_message_count", len(history_messages))

        if request.input_budget is not None and request.input_budget <= 0 and history_messages:
            fallback = self._recovery_policy.build_latest_message_recovery(
                history_messages,
                estimator=prepared.estimator,
            )
            llm_messages, context_usage = fallback or ([], {})
            return ContextRequestBuildResult(
                llm_messages=llm_messages,
                context_usage=context_usage,
                history_message_count=len(history_messages),
            )

        with stage_span(
            request.span,
            "context.plan_assembled",
            {
                "chat.context.candidate_count": len(prepared.candidates),
                "chat.context.history_messages": len(history_messages),
            },
        ) as plan_span:
            plan = planner.plan(
                messages=history_messages,
                system_instructions=request.system_instructions,
                memory_context=request.memory_text,
                input_budget=request.input_budget,
                candidates=prepared.candidates,
                selection_policy=self._selection_policy,
                boundary_id=prepared.boundary_id,
                truncation_log_label=request.truncation_log_label,
            )
            plan = self._hook_service.run_plan_assembled(
                plan,
                span=plan_span or request.span,
            )
            usage = getattr(plan, "usage", None)
            if usage is not None:
                dropped_blocks = getattr(usage, "dropped_blocks", None)
                if isinstance(dropped_blocks, list):
                    (plan_span or request.span).set_attribute(
                        "chat.context.blocks_dropped_count",
                        len(dropped_blocks),
                    )

        with stage_span(request.span, "context.render") as render_span:
            llm_messages = renderer.render(plan)
            llm_messages = self._hook_service.run_render(
                llm_messages,
                span=render_span or request.span,
            )
            (render_span or request.span).set_attribute(
                "chat.context.rendered_message_count",
                len(llm_messages),
            )
        if not llm_messages and history_messages:
            fallback = self._render_recovery_policy.apply_empty_render_recovery(
                history_messages,
                estimator=prepared.estimator,
                usage=plan.usage,
            )
            llm_messages, context_usage = fallback or ([], {})
        else:
            context_usage = self._build_context_usage(
                plan_usage=plan.usage.model_dump(mode="json"),
                estimator=prepared.estimator,
                llm_messages=llm_messages,
                summary_decision=prepared.summary_decision,
            )
            self._apply_usage_observability(
                span=request.span,
                context_usage=context_usage,
            )

        return ContextRequestBuildResult(
            llm_messages=llm_messages,
            context_usage=context_usage,
            history_message_count=len(history_messages),
        )

    def get_usage(self, request: ContextRequestBuildInput) -> dict[str, Any] | None:
        if not request.history_messages:
            return None
        prepared = self._prepare_request(request)
        if not prepared.history_messages:
            return None
        planner = ContextPlanner(
            token_estimator=prepared.estimator,
            system_instructions=request.system_instructions,
        )
        plan = planner.plan(
            messages=prepared.history_messages,
            system_instructions=request.system_instructions,
            memory_context=request.memory_text,
            candidates=prepared.candidates,
            selection_policy=self._selection_policy,
            boundary_id=prepared.boundary_id,
            truncation_log_label=request.truncation_log_label,
        )
        renderer = ContextRenderer()
        llm_messages = renderer.render(plan)
        return self._build_context_usage(
            plan_usage=plan.usage.model_dump(mode="json"),
            estimator=prepared.estimator,
            llm_messages=llm_messages,
            summary_decision=prepared.summary_decision,
        )

    def build_from_source(self, request: ContextRequestSourceInput) -> ContextRequestBuildResult:
        return self.build(self._build_input(request))

    def get_usage_from_source(self, request: ContextRequestSourceInput) -> dict[str, Any] | None:
        return self.get_usage(self._build_input(request))

    def _build_context_usage(
        self,
        *,
        plan_usage: dict[str, Any],
        estimator: TokenEstimator,
        llm_messages: list[dict[str, Any]],
        summary_decision: SummaryEligibilityDecision,
    ) -> dict[str, Any]:
        usage = dict(plan_usage)
        # Rendered prompt accounting catches cases where plan-level savings are
        # eaten back by summary / memory / tool-summary rendering overhead.
        usage["assembled_prompt_tokens"] = estimator.count_messages(llm_messages)
        usage["assembled_message_count"] = len(llm_messages)
        usage["summary_eligible"] = summary_decision.include
        usage["summary_reason"] = summary_decision.reason
        usage["summary_metadata"] = dict(summary_decision.metadata)
        return usage

    @staticmethod
    def _apply_usage_observability(
        *,
        span: Any,
        context_usage: dict[str, Any],
    ) -> None:
        assembled_prompt_tokens = context_usage.get("assembled_prompt_tokens")
        if isinstance(assembled_prompt_tokens, int):
            span.set_attribute("chat.context.assembled_prompt_tokens", assembled_prompt_tokens)
        assembled_message_count = context_usage.get("assembled_message_count")
        if isinstance(assembled_message_count, int):
            span.set_attribute("chat.context.assembled_message_count", assembled_message_count)
        span.set_attribute(
            "chat.context.summary_eligible",
            bool(context_usage.get("summary_eligible")),
        )
        summary_reason = context_usage.get("summary_reason")
        if isinstance(summary_reason, str) and summary_reason:
            span.set_attribute("chat.context.summary_reason", summary_reason)

    def _build_input(self, request: ContextRequestSourceInput) -> ContextRequestBuildInput:
        if self._build_history_messages is None:
            raise RuntimeError("ContextRequestBuilder requires source input callbacks")
        history_messages = self._build_history_messages(request.source, request.model)
        if self._sanitize_history_messages is not None:
            history_messages = self._sanitize_history_messages(history_messages)
        memory_text = self._get_memory_text(request.span) if self._get_memory_text else None
        return ContextRequestBuildInput(
            model=request.model,
            system_instructions=request.system_instructions,
            history_messages=history_messages,
            conversation_messages=self._get_conversation_messages(request.source),
            conversation_metadata=self._get_conversation_metadata(request.source),
            memory_text=memory_text if memory_text else None,
            span=request.span,
            input_budget=request.input_budget,
            truncation_log_label=request.truncation_log_label,
        )

    def _prepare_request(self, request: ContextRequestBuildInput) -> _PreparedContextRequest:
        estimator = TokenEstimator(model=request.model)
        summary_text, summary_metadata = extract_latest_compaction_summary(
            request.conversation_metadata
        )
        boundary_id = uuid.uuid4().hex[:12]
        tool_summary_candidates, summarized_tool_ids = build_tool_summary_candidates(
            request.conversation_messages,
            boundary_id=boundary_id,
        )
        history_messages = self._filter_history_messages(
            request.history_messages,
            summarized_tool_ids,
        )
        summary_decision = self._evaluate_summary_eligibility(
            summary_text=summary_text,
            summary_metadata=summary_metadata,
            history_messages=history_messages,
            boundary_id=boundary_id,
            selection_policy=self._selection_policy,
        )
        candidates = assemble_context_candidates(
            messages=history_messages,
            system_instructions=request.system_instructions,
            memory_context=request.memory_text,
            summary_context=summary_text if summary_decision.include else None,
            summary_metadata=(
                {
                    **summary_metadata,
                    "eligibility_reason": summary_decision.reason,
                    **summary_decision.metadata,
                }
                if summary_decision.include
                else None
            ),
            boundary_id=boundary_id,
        )
        candidates.extend(build_pinned_context_candidates(request.conversation_metadata))
        candidates.extend(tool_summary_candidates)
        return _PreparedContextRequest(
            estimator=estimator,
            history_messages=history_messages,
            candidates=candidates,
            boundary_id=boundary_id,
            summary_decision=summary_decision,
        )

    def _filter_history_messages(
        self,
        history_messages: list[dict[str, Any]],
        summarized_tool_ids: set[str],
    ) -> list[dict[str, Any]]:
        if not summarized_tool_ids:
            return list(history_messages)
        filtered: list[dict[str, Any]] = []
        for message in history_messages:
            tool_call_id = message.get("tool_call_id")
            message_id = message.get("message_id")
            if tool_call_id in summarized_tool_ids or message_id in summarized_tool_ids:
                continue
            filtered.append(message)
        return filtered

    def _evaluate_summary_eligibility(
        self,
        *,
        summary_text: str | None,
        summary_metadata: dict[str, Any] | None,
        history_messages: list[dict[str, Any]],
        boundary_id: str,
        selection_policy: ContextSelectionPolicy,
    ) -> SummaryEligibilityDecision:
        metadata = dict(summary_metadata or {})
        if not selection_policy.allow_summaries:
            return SummaryEligibilityDecision(
                include=False,
                reason="policy_disabled",
                metadata={"boundary_id": boundary_id},
            )
        summary = str(summary_text or "").strip()
        if not summary:
            return SummaryEligibilityDecision(
                include=False,
                reason="empty_summary",
                metadata={"boundary_id": boundary_id},
            )
        origin_decision = self._summary_origin_gate(metadata, boundary_id=boundary_id)
        if not origin_decision.include:
            return origin_decision
        freshness_decision = self._summary_freshness_gate(
            metadata,
            history_messages=history_messages,
            boundary_id=boundary_id,
        )
        if not freshness_decision.include:
            return freshness_decision
        relevance_decision = self._summary_relevance_gate(
            summary,
            history_messages=history_messages,
            boundary_id=boundary_id,
        )
        if not relevance_decision.include:
            return relevance_decision
        return SummaryEligibilityDecision(
            include=True,
            reason="eligible",
            metadata={
                **metadata,
                **origin_decision.metadata,
                **freshness_decision.metadata,
                **relevance_decision.metadata,
                "boundary_id": boundary_id,
            },
        )

    @staticmethod
    def _summary_origin_gate(
        metadata: dict[str, Any],
        *,
        boundary_id: str,
    ) -> SummaryEligibilityDecision:
        trigger = str(metadata.get("trigger") or "").strip().lower()
        summarization_mode = str(metadata.get("summarization_mode") or "").strip().lower()
        if trigger and trigger not in {"pre_request_pressure", "threshold", "manual"}:
            return SummaryEligibilityDecision(
                include=False,
                reason="origin_rejected",
                metadata={"boundary_id": boundary_id, "summary_trigger": trigger},
            )
        if summarization_mode == "fallback_prune_only":
            return SummaryEligibilityDecision(
                include=False,
                reason="origin_prune_only",
                metadata={
                    "boundary_id": boundary_id,
                    "summary_trigger": trigger,
                    "summarization_mode": summarization_mode,
                },
            )
        return SummaryEligibilityDecision(
            include=True,
            reason="origin_ok",
            metadata={
                "boundary_id": boundary_id,
                "summary_trigger": trigger or None,
                "summarization_mode": summarization_mode or None,
            },
        )

    @staticmethod
    def _summary_freshness_gate(
        metadata: dict[str, Any],
        *,
        history_messages: list[dict[str, Any]],
        boundary_id: str,
    ) -> SummaryEligibilityDecision:
        created_at = metadata.get("created_at")
        age_seconds: float | None = None
        if isinstance(created_at, (int, float)):
            age_seconds = max(0.0, time.time() - float(created_at))
            if age_seconds > 60 * 15:
                return SummaryEligibilityDecision(
                    include=False,
                    reason="stale_summary",
                    metadata={
                        "boundary_id": boundary_id,
                        "summary_age_seconds": round(age_seconds, 2),
                    },
                )
        source_message_ids = metadata.get("source_message_ids")
        if isinstance(source_message_ids, list) and source_message_ids:
            source_ids = {str(item) for item in source_message_ids if item}
            recent_ids = {
                str(message.get("message_id"))
                for message in history_messages[-6:]
                if isinstance(message.get("message_id"), str) and message.get("message_id")
            }
            if recent_ids.intersection(source_ids):
                return SummaryEligibilityDecision(
                    include=False,
                    reason="boundary_overlap",
                    metadata={
                        "boundary_id": boundary_id,
                        "summary_age_seconds": round(age_seconds, 2)
                        if age_seconds is not None
                        else None,
                    },
                )
        if len(history_messages) > 10:
            return SummaryEligibilityDecision(
                include=False,
                reason="too_many_new_turns",
                metadata={
                    "boundary_id": boundary_id,
                    "summary_age_seconds": round(age_seconds, 2)
                    if age_seconds is not None
                    else None,
                },
            )
        return SummaryEligibilityDecision(
            include=True,
            reason="freshness_ok",
            metadata={
                "boundary_id": boundary_id,
                "summary_age_seconds": round(age_seconds, 2) if age_seconds is not None else None,
            },
        )

    def _summary_relevance_gate(
        self,
        summary_text: str,
        *,
        history_messages: list[dict[str, Any]],
        boundary_id: str,
    ) -> SummaryEligibilityDecision:
        latest_user_text = self._latest_user_text(history_messages)
        if not latest_user_text:
            return SummaryEligibilityDecision(
                include=False,
                reason="no_latest_user_turn",
                metadata={"boundary_id": boundary_id},
            )
        score_details = self._score_summary_relevance(
            summary_text=summary_text,
            latest_user_text=latest_user_text,
            history_messages=history_messages,
        )
        relevance_score = score_details["relevance_score"]
        metadata = {
            "boundary_id": boundary_id,
            **score_details,
        }
        if relevance_score >= 2:
            return SummaryEligibilityDecision(
                include=True,
                reason="relevance_scored",
                metadata=metadata,
            )
        semantic_decision = self._run_summary_semantic_judge(
            summary_text=summary_text,
            latest_user_text=latest_user_text,
            history_messages=history_messages,
            score_details=metadata,
        )
        if semantic_decision is not None:
            return semantic_decision
        return SummaryEligibilityDecision(
            include=False,
            reason="low_relevance",
            metadata=metadata,
        )

    def _score_summary_relevance(
        self,
        *,
        summary_text: str,
        latest_user_text: str,
        history_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # This layer stays deterministic and cheap. It is intentionally richer
        # than raw keyword overlap, but still avoids a mandatory model call.
        latest_terms = self._keyword_terms(latest_user_text)
        summary_terms = self._keyword_terms(summary_text)
        overlap = latest_terms.intersection(summary_terms)
        intent_overlap = self._intent_terms(latest_user_text).intersection(
            self._intent_terms(summary_text)
        )
        entity_overlap = self._entity_terms(latest_user_text).intersection(
            self._entity_terms(summary_text)
        )
        file_overlap = self._path_terms(latest_user_text).intersection(
            self._path_terms(summary_text)
        )
        turn_overlap = self._turn_terms(latest_user_text).intersection(
            self._turn_terms(summary_text)
        )
        recent_overlap = self._recent_topic_terms(history_messages).intersection(summary_terms)
        generic_only = self._is_generic_task_turn(
            latest_user_text,
            overlap=overlap,
            intent_overlap=intent_overlap,
            entity_overlap=entity_overlap,
            file_overlap=file_overlap,
            turn_overlap=turn_overlap,
        )
        relevance_score = 0
        if len(overlap) >= 2:
            relevance_score += 2
        elif overlap:
            relevance_score += 1
        if intent_overlap:
            relevance_score += 1
        if entity_overlap:
            relevance_score += 2
        if file_overlap:
            relevance_score += 2
        if turn_overlap:
            relevance_score += 1
        if recent_overlap:
            relevance_score += 1
        if self._looks_like_repo_or_tool_turn(latest_user_text) and (
            entity_overlap or file_overlap or turn_overlap
        ):
            relevance_score += 1
        if generic_only:
            relevance_score -= 2
        return {
            "relevance_overlap": sorted(overlap),
            "relevance_intent_overlap": sorted(intent_overlap),
            "relevance_entity_overlap": sorted(entity_overlap),
            "relevance_file_overlap": sorted(file_overlap),
            "relevance_turn_overlap": sorted(turn_overlap),
            "relevance_recent_overlap": sorted(recent_overlap),
            "relevance_generic_only": generic_only,
            "relevance_score": relevance_score,
        }

    def _run_summary_semantic_judge(
        self,
        *,
        summary_text: str,
        latest_user_text: str,
        history_messages: list[dict[str, Any]],
        score_details: dict[str, Any],
    ) -> SummaryEligibilityDecision | None:
        if self._summary_semantic_judge is None:
            return None
        base_score = int(score_details.get("relevance_score") or 0)
        needs_judge = (
            base_score == 1
            or bool(score_details.get("relevance_generic_only"))
            or (
                self._looks_like_repo_or_tool_turn(latest_user_text)
                and not score_details.get("relevance_file_overlap")
                and not score_details.get("relevance_entity_overlap")
            )
        )
        if not needs_judge:
            return None
        result = self._summary_semantic_judge(
            SummarySemanticJudgeInput(
                summary_text=summary_text,
                latest_user_text=latest_user_text,
                history_messages=history_messages,
                base_score=base_score,
                metadata=dict(score_details),
            )
        )
        if result is None:
            return None
        return SummaryEligibilityDecision(
            include=result.include,
            reason=result.reason,
            metadata=dict(result.metadata),
        )

    @staticmethod
    def _latest_user_text(history_messages: list[dict[str, Any]]) -> str:
        for message in reversed(history_messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        return ""

    @staticmethod
    def _keyword_terms(text: str) -> set[str]:
        normalized = str(text or "").lower()
        return {
            token
            for token in re.findall(r"[a-z0-9_\-\.]{3,}", normalized)
            if token not in {"the", "and", "for", "with", "from", "that", "this", "you", "are"}
        }

    @staticmethod
    def _intent_terms(text: str) -> set[str]:
        lowered = str(text or "").lower()
        intents = {
            "fix",
            "debug",
            "trace",
            "patch",
            "refactor",
            "search",
            "find",
            "read",
            "inspect",
            "investigate",
            "compare",
            "analyze",
            "summarize",
            "context",
            "summary",
            "model",
        }
        return {intent for intent in intents if intent in lowered}

    @staticmethod
    def _entity_terms(text: str) -> set[str]:
        normalized = str(text or "")
        entities = {
            token.lower()
            for token in re.findall(r"\b[A-Z][A-Za-z0-9_.-]*\b", normalized)
            if len(token) >= 3
        }
        entities.update(
            token
            for token in re.findall(r"[a-z0-9_\-]+\.[a-z0-9_/\.-]+", normalized.lower())
            if len(token) >= 5
        )
        return entities

    @staticmethod
    def _path_terms(text: str) -> set[str]:
        normalized = str(text or "").lower()
        return {
            token
            for token in re.findall(r"(?:[a-z0-9_.-]+/)*[a-z0-9_.-]+\.[a-z0-9_.-]+", normalized)
            if len(token) >= 5
        }

    @staticmethod
    def _turn_terms(text: str) -> set[str]:
        normalized = str(text or "")
        terms = {
            token.lower() for token in re.findall(r"`([^`]{2,80})`", normalized) if len(token) >= 2
        }
        terms.update(
            token.lower()
            for token in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", normalized)
            if ("_" in token or any(char.isupper() for char in token[1:]))
        )
        return terms

    def _recent_topic_terms(self, history_messages: list[dict[str, Any]]) -> set[str]:
        recent_user_text = " ".join(
            str(message.get("content") or "")
            for message in history_messages[-4:]
            if message.get("role") == "user"
        )
        return self._keyword_terms(recent_user_text)

    def _is_generic_task_turn(
        self,
        text: str,
        *,
        overlap: set[str],
        intent_overlap: set[str],
        entity_overlap: set[str],
        file_overlap: set[str],
        turn_overlap: set[str],
    ) -> bool:
        if entity_overlap or file_overlap or turn_overlap:
            return False
        generic_terms = self._intent_terms(text)
        if not generic_terms:
            return False
        return bool(intent_overlap) and not overlap.difference(generic_terms)

    @staticmethod
    def _looks_like_repo_or_tool_turn(text: str) -> bool:
        lowered = str(text or "").lower()
        keywords = (
            "repo",
            "file",
            "code",
            "search",
            "grep",
            "find",
            "read",
            "trace",
            "bug",
            "fix",
            "patch",
            "model",
            "context",
            "summary",
        )
        return any(keyword in lowered for keyword in keywords)
