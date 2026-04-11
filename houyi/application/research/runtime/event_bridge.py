from __future__ import annotations

from typing import Any

from houyi.application.research.types import SearchResult
from houyi.application.runtime.events import AgentEvent, AgentEventType, EventEmitter
from houyi.application.runtime.message_bus import AgentMessage, AgentMessageBus, AgentMessageType

_EVENT_TO_A2A: dict[str, AgentMessageType] = {
    "research.source_found": AgentMessageType.SOURCE_DISCOVERED,
    "research.step_completed": AgentMessageType.QUESTION_COVERED,
    "research.agent_spawned": AgentMessageType.TASK_DELEGATE,
    "research.agent_completed": AgentMessageType.TASK_RESULT,
    "research.step_started": AgentMessageType.TASK_PROGRESS,
    "research.conflict_detected": AgentMessageType.CONFLICT_DETECTED,
    "research.plan_generated": AgentMessageType.FINDING_PUBLISHED,
}

_AGENT_EVENT_TO_A2A: dict[AgentEventType, AgentMessageType] = {
    AgentEventType.TEAM_AGENT_SPAWNED: AgentMessageType.TASK_DELEGATE,
    AgentEventType.TEAM_AGENT_COMPLETED: AgentMessageType.TASK_RESULT,
    AgentEventType.TOOL_COMPLETED: AgentMessageType.SOURCE_DISCOVERED,
    AgentEventType.PROGRESS: AgentMessageType.TASK_PROGRESS,
}


class ResearchEventBridge:
    def __init__(
        self,
        *,
        run_id: str,
        emitter: EventEmitter,
        message_bus: AgentMessageBus | None,
    ) -> None:
        self._run_id = run_id
        self._emitter = emitter
        self._bus = message_bus
        self._event_sequence = 0

    @property
    def event_sequence(self) -> int:
        return self._event_sequence

    async def emit(self, event_type: str, **data: Any) -> None:
        self._event_sequence += 1
        merged = {"research_event": event_type, "sequence": self._event_sequence, **data}
        await self._emitter.emit(
            AgentEvent(
                event_type=AgentEventType.PROGRESS,
                agent_id=self._run_id,
                data=merged,
            ),
        )
        if self._bus is None:
            return
        a2a_type = _EVENT_TO_A2A.get(event_type)
        if a2a_type is None:
            return
        await self._bus.publish(
            f"research.{self._run_id}",
            AgentMessage(
                sender_id=self._run_id,
                message_type=a2a_type,
                topic=f"research.{self._run_id}",
                payload=merged,
            ),
        )

    async def on_search_event(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "search.queries_generated":
            await self.emit(
                "research.search_queries",
                question_id=data.get("question_id", ""),
                round=data.get("round", 0),
                queries=data.get("queries", []),
                bilingual_expected=data.get("bilingual_expected", False),
                bilingual_fallback_applied=data.get("bilingual_fallback_applied", False),
                language_mix=data.get("language_mix", []),
            )
            return
        if event_type == "search.source_discovered":
            await self.emit(
                "research.source_found",
                title=data.get("title", ""),
                url=data.get("url", ""),
                snippet=data.get("snippet", ""),
                question_id=data.get("question_id", ""),
                query=data.get("query", ""),
            )
            return
        if event_type == "search.query_timing":
            await self.emit(
                "research.search_query_timing",
                question_id=data.get("question_id", ""),
                round=data.get("round", 0),
                query=data.get("query", ""),
                elapsed_ms=data.get("elapsed_ms", 0.0),
                hit_count=data.get("hit_count", 0),
                cancelled=data.get("cancelled", False),
                provider=data.get("provider", ""),
                reason_code=data.get("reason_code", ""),
            )
            return
        if event_type == "search.query_cancelled":
            await self.emit(
                "research.search_query_cancelled",
                question_id=data.get("question_id", ""),
                round=data.get("round", 0),
                query=data.get("query", ""),
                reason=data.get("reason", ""),
            )
            return
        if event_type == "search.round_timing":
            await self.emit(
                "research.search_round_timing",
                question_id=data.get("question_id", ""),
                round=data.get("round", 0),
                elapsed_ms=data.get("elapsed_ms", 0.0),
                query_count=data.get("query_count", 0),
                skipped_queries=data.get("skipped_queries", 0),
                cancelled_queries=data.get("cancelled_queries", 0),
                hit_count=data.get("hit_count", 0),
                source_count=data.get("source_count", 0),
                sufficient=data.get("sufficient", False),
                rationale=data.get("rationale", ""),
                decision_by=data.get("decision_by", ""),
                reason_code=data.get("reason_code", ""),
                stop_layer=data.get("stop_layer", ""),
                missing_dimensions=data.get("missing_dimensions", []),
            )
            return
        if event_type == "search.budget_consumed":
            await self.emit(
                "research.search_budget",
                question_id=data.get("question_id", ""),
                round=data.get("round", 0),
                layer=data.get("layer", ""),
                reason_code=data.get("reason_code", ""),
                budget_ms=data.get("budget_ms", 0),
                remaining_ms=data.get("remaining_ms", 0),
            )
            return
        if event_type == "search.sufficiency_features":
            await self.emit(
                "research.search_sufficiency",
                question_id=data.get("question_id", ""),
                round=data.get("round", 0),
                source_count=data.get("source_count", 0),
                relevant_source_count=data.get("relevant_source_count", 0),
                domain_count=data.get("domain_count", 0),
                provider_count=data.get("provider_count", 0),
                authority_source_count=data.get("authority_source_count", 0),
                recent_source_count=data.get("recent_source_count", 0),
                relevance_score=data.get("relevance_score", 0.0),
                diversity_score=data.get("diversity_score", 0.0),
                authority_score=data.get("authority_score", 0.0),
                recency_score=data.get("recency_score", 0.0),
                missing_dimensions=data.get("missing_dimensions", []),
            )
            return
        if event_type == "search.sufficiency_decision":
            await self.emit(
                "research.search_stop_reason",
                question_id=data.get("question_id", ""),
                round=data.get("round", 0),
                sufficient=data.get("sufficient", False),
                rationale=data.get("rationale", ""),
                decision_by=data.get("decision_by", ""),
                reason_code=data.get("reason_code", ""),
                missing_dimensions=data.get("missing_dimensions", []),
            )

    async def emit_report_generation_end(
        self,
        *,
        total_steps: int,
        elapsed_seconds: float,
        error: str | None = None,
    ) -> None:
        await self.emit(
            "research.step_completed",
            step_id="report_generation",
            step="Generating report...",
            total_steps=total_steps,
            completed_steps=total_steps,
            elapsed_seconds=elapsed_seconds,
            failed=bool(error),
            error=error or "",
        )

    async def emit_restored_search_events(
        self,
        question_id: str,
        result: SearchResult,
    ) -> None:
        for search_round in result.rounds:
            if search_round.queries:
                await self.emit(
                    "research.search_queries",
                    question_id=question_id,
                    round=search_round.round_index + 1,
                    queries=search_round.queries,
                )
        for source in result.sources[:12]:
            await self.emit(
                "research.source_found",
                question_id=question_id,
                title=source.title or "",
                url=source.url or "",
                snippet=(source.snippet or "")[:500],
                query="",
            )

    async def bridge_agent_event(self, event: AgentEvent) -> None:
        if self._bus is None:
            return
        if event.event_type == AgentEventType.PROGRESS:
            return
        a2a_type = _AGENT_EVENT_TO_A2A.get(event.event_type)
        if a2a_type is None:
            return
        await self._bus.publish(
            f"research.{self._run_id}",
            AgentMessage(
                sender_id=event.agent_id or self._run_id,
                message_type=a2a_type,
                topic=f"research.{self._run_id}",
                payload={"agent_event": event.event_type.value, **event.data},
            ),
        )
