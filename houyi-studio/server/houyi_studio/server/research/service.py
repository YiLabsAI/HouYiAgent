"""Research service — adapts server layer to SDK ResearchSession.

Manages session persistence (JSON), idempotency, and lifecycle coordination.
Sessions are persisted as JSON and rehydrated on startup so that history
(including failed/timed-out sessions) survives server restarts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from houyi_studio.server.research.sse import ResearchSSEEnvelope

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.intermediate import IntermediateReport
from houyi.application.research.session import ResearchSession
from houyi.application.research.types import (
    PlanEdit,
    PlanStatus,
    ResearchPlan,
    ResearchProgress,
    ResearchReport,
    ResearchSettings,
    ResearchStatus,
    SearchResult,
)
from houyi.application.runtime.events import EventEmitter
from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)


class _ArchivedSession:
    """Lightweight read-only session restored from persisted JSON.

    Not a real ``ResearchSession`` — only stores enough data for
    list / get / delete operations on historical sessions.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.session_id: str = data["session_id"]
        self._status_str: str = data.get("status", "failed")
        self._plan_data: dict[str, Any] | None = data.get("plan")
        self._progress_data: dict[str, Any] = data.get("progress", {})
        self._error: str | None = data.get("error")
        self._report_data: dict[str, Any] | None = data.get("report")
        self._search_results_data: list[dict[str, Any]] = data.get("search_results", [])
        self._intermediate_reports_data: list[dict[str, Any]] = data.get("intermediate_reports", [])
        self.created_at: str = data.get("created_at", "")
        self.updated_at: float = data.get("updated_at", 0)

    @property
    def status(self) -> ResearchStatus:
        try:
            return ResearchStatus(self._status_str)
        except ValueError:
            return ResearchStatus.FAILED

    @property
    def plan(self) -> ResearchPlan | None:
        if self._plan_data is None:
            return None
        try:
            return ResearchPlan.model_validate(self._plan_data)
        except Exception:
            return None

    @property
    def progress(self) -> ResearchProgress:
        try:
            return ResearchProgress.model_validate(self._progress_data)
        except Exception:
            return ResearchProgress()

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def report_data(self) -> dict[str, Any] | None:
        return self._report_data

    @property
    def search_results_data(self) -> list[dict[str, Any]]:
        return self._search_results_data

    @property
    def intermediate_reports_data(self) -> list[dict[str, Any]]:
        return self._intermediate_reports_data


class ResearchService:
    """Server-side service managing research sessions.

    Provides idempotent session creation, plan editing with version control,
    execution lifecycle, and JSON-based persistence for session recovery.
    """

    MAX_CONCURRENT_SESSIONS = 3

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        web_search: WebSearchService,
        data_dir: Path | None = None,
        memory_service: Any | None = None,
    ) -> None:
        self._llm = llm_adapter
        self._web_search = web_search
        self._data_dir = data_dir or Path.home() / ".houyi" / "research"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ResearchSession | _ArchivedSession] = {}
        self._idempotency: dict[str, str] = {}
        self._emitters: dict[str, EventEmitter] = {}
        self._event_buffers: dict[str, list] = {}
        self._memory_service = memory_service
        self._load_persisted_sessions()

    def _attach_buffer_listener(self, session_id: str, emitter: EventEmitter) -> None:
        """Attach an always-on listener that buffers events for late SSE subscribers."""
        buf = self._event_buffers.setdefault(session_id, [])
        seq_counter = {"n": 0}

        async def _buffer_handler(event: Any) -> None:
            research_event = event.data.get("research_event", "unknown")
            seq_counter["n"] += 1
            payload = {
                k: v for k, v in event.data.items() if k not in ("research_event", "sequence")
            }
            env = ResearchSSEEnvelope(
                event_type=research_event,
                session_id=session_id,
                sequence=seq_counter["n"],
                payload=payload,
            )
            buf.append(env)

        emitter.on_any(_buffer_handler)

    def running_session_count(self) -> int:
        """Count currently executing sessions."""
        return sum(
            1
            for s in self._sessions.values()
            if not isinstance(s, _ArchivedSession)
            and s.status in (ResearchStatus.EXECUTING, ResearchStatus.GENERATING_REPORT)
        )

    async def create_session(
        self,
        query: str,
        settings: ResearchSettings | None = None,
        idempotency_key: str | None = None,
        memory_context: str | None = None,
    ) -> tuple[ResearchSession, ResearchPlan]:
        """Create a new session or return existing one for idempotent requests."""
        if idempotency_key and idempotency_key in self._idempotency:
            sid = self._idempotency[idempotency_key]
            session = self._sessions[sid]
            return session, session.plan  # type: ignore[return-value]

        emitter = EventEmitter()
        session = ResearchSession(
            llm_adapter=self._llm,
            web_search=self._web_search,
            settings=settings,
            event_emitter=emitter,
            memory_context=memory_context,
        )
        plan = await session.start(query)
        self._sessions[session.session_id] = session
        self._emitters[session.session_id] = emitter
        self._attach_buffer_listener(session.session_id, emitter)
        if idempotency_key:
            self._idempotency[idempotency_key] = session.session_id

        self._persist_session(session)
        return session, plan

    def get_session(self, session_id: str) -> ResearchSession | _ArchivedSession | None:
        return self._sessions.get(session_id)

    def get_emitter(self, session_id: str) -> EventEmitter | None:
        return self._emitters.get(session_id)

    def get_event_buffer(self, session_id: str) -> list:
        return self._event_buffers.setdefault(session_id, [])

    async def edit_plan(
        self,
        session_id: str,
        edits: list[PlanEdit],
        client_plan_version: int | None = None,
    ) -> ResearchPlan:
        """Edit a session's plan with optional version check."""
        session = self._require_live_session(session_id)
        if (
            client_plan_version is not None
            and session.plan
            and client_plan_version != session.plan.version
        ):
            msg = (
                f"Plan version conflict: client={client_plan_version}, "
                f"server={session.plan.version}"
            )
            raise VersionConflictError(msg)
        plan = await session.edit_plan(edits)
        self._persist_session(session)
        return plan

    def prepare_for_execution(self, session_id: str) -> ResearchSession:
        """Ensure session is live (rehydrated if archived) before scheduling.

        Must be called synchronously before ``asyncio.create_task`` so that
        the ``EventEmitter`` is registered before the frontend connects SSE.
        """
        return self._require_live_session(session_id)

    async def confirm_and_execute(self, session_id: str) -> None:
        """Confirm plan, execute, and extract memory candidates on success."""
        session = self._require_live_session(session_id)
        if session.status == ResearchStatus.EXECUTING:
            return
        await session.confirm_plan()
        try:
            await session.execute()
        finally:
            self._persist_session(session)
        await self._extract_and_push_memories(session)

    async def _extract_and_push_memories(self, session: ResearchSession) -> None:
        """Extract memory candidates from a completed session and push to MemoryService."""
        if not self._memory_service:
            return
        if session.status != ResearchStatus.COMPLETED:
            return
        try:
            candidates = await session.extract_memories()
            if candidates:
                self._memory_service.add_candidates(candidates)
                logger.info(
                    "Pushed %d memory candidates from session %s",
                    len(candidates),
                    session.session_id,
                )
        except Exception:
            logger.warning(
                "Memory extraction failed for session %s",
                session.session_id,
                exc_info=True,
            )

    async def cancel_session(self, session_id: str, reason: str = "") -> None:
        session = self._require_live_session(session_id)
        await session.cancel(reason)
        self._persist_session(session)

    async def get_report(self, session_id: str) -> ResearchReport:
        session = self._require_session(session_id)
        if isinstance(session, _ArchivedSession):
            if session.report_data:
                return ResearchReport.model_validate(session.report_data)
            raise RuntimeError("No report available for archived session")
        return await session.get_report()

    def get_progress(self, session_id: str) -> ResearchProgress:
        session = self._require_session(session_id)
        return session.progress

    def list_sessions(
        self,
        offset: int = 0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List sessions with pagination (live + archived)."""
        items = []
        for session in self._sessions.values():
            items.append(
                {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "query": session.plan.query if session.plan else "",
                    "progress": session.progress.model_dump(),
                    "error": getattr(session, "_error", None) or getattr(session, "error", None),
                    "created_at": getattr(session, "created_at", None),
                }
            )
        items.sort(key=lambda x: x.get("created_at") or x["session_id"], reverse=True)
        return items[offset : offset + limit]

    async def delete_session(self, session_id: str) -> None:
        """Delete a terminal or archived session."""
        session = self._sessions.get(session_id)
        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")
        if session.status == ResearchStatus.EXECUTING:
            raise SessionNotTerminalError(f"Cannot delete session in status {session.status.value}")
        del self._sessions[session_id]
        self._emitters.pop(session_id, None)
        path = self._data_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()

    def _require_live_session(self, session_id: str) -> ResearchSession:
        """Return a live session, rehydrating if archived."""
        raw = self._sessions.get(session_id)
        if not raw:
            raise SessionNotFoundError(f"Session {session_id} not found")
        if isinstance(raw, _ArchivedSession):
            return self._rehydrate(raw)
        return raw

    def _require_session(self, session_id: str) -> ResearchSession:
        session = self._sessions.get(session_id)
        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return session

    def _rehydrate(self, archived: _ArchivedSession) -> ResearchSession:
        """Convert an archived session back into a live ResearchSession.

        Restores plan and search_results so that retry can checkpoint
        already-completed sub-questions and skip re-searching them.
        """
        emitter = EventEmitter()
        session = ResearchSession(
            session_id=archived.session_id,
            llm_adapter=self._llm,
            web_search=self._web_search,
            event_emitter=emitter,
        )
        if archived.plan:
            plan = archived.plan
            plan.status = PlanStatus.DRAFT
            session._plan = plan
            session._status = ResearchStatus.PLAN_READY
        session.created_at = archived.created_at or session.created_at

        for i, sr_data in enumerate(archived.search_results_data):
            try:
                session._search_results.append(SearchResult.model_validate(sr_data))
            except Exception:
                logger.warning(
                    "Failed to restore search result [%d] for session %s",
                    i,
                    archived.session_id,
                    exc_info=True,
                )
        if session._search_results:
            logger.info(
                "Restored %d search results for session %s",
                len(session._search_results),
                archived.session_id,
            )

        for i, ir_data in enumerate(archived.intermediate_reports_data):
            try:
                session._intermediate_reports.append(IntermediateReport.model_validate(ir_data))
            except Exception:
                logger.warning(
                    "Failed to restore intermediate report [%d] for session %s",
                    i,
                    archived.session_id,
                    exc_info=True,
                )
        if session._intermediate_reports:
            logger.info(
                "Restored %d intermediate reports for session %s",
                len(session._intermediate_reports),
                archived.session_id,
            )

        self._sessions[archived.session_id] = session
        self._emitters[archived.session_id] = emitter
        self._event_buffers.pop(archived.session_id, None)
        self._attach_buffer_listener(archived.session_id, emitter)
        logger.info("Rehydrated archived session %s into live session", archived.session_id)
        return session

    def _persist_session(self, session: ResearchSession) -> None:
        """Persist session state to JSON for recovery."""
        path = self._data_dir / f"{session.session_id}.json"
        error = getattr(session, "_error", None) or getattr(session, "error", None)
        report_data = None
        if hasattr(session, "_report") and session._report is not None:
            with contextlib.suppress(Exception):
                report_data = session._report.model_dump()
        search_results_data: list[dict] = []
        if hasattr(session, "_search_results"):
            with contextlib.suppress(Exception):
                search_results_data = [sr.model_dump() for sr in session._search_results]
        intermediate_data: list[dict] = []
        if hasattr(session, "_intermediate_reports"):
            with contextlib.suppress(Exception):
                intermediate_data = [ir.model_dump() for ir in session._intermediate_reports]
        data = {
            "session_id": session.session_id,
            "status": session.status.value,
            "plan": session.plan.model_dump() if session.plan else None,
            "progress": session.progress.model_dump(),
            "error": error,
            "report": report_data,
            "search_results": search_results_data,
            "intermediate_reports": intermediate_data,
            "created_at": getattr(session, "created_at", ""),
            "updated_at": time.time(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        tmp.rename(path)

    def _load_persisted_sessions(self) -> None:
        """Load archived sessions from JSON files on startup."""
        count = 0
        for path in self._data_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sid = data.get("session_id")
                if not sid or sid in self._sessions:
                    continue
                self._sessions[sid] = _ArchivedSession(data)
                count += 1
            except Exception:
                logger.warning("Failed to load session from %s", path, exc_info=True)
        if count:
            logger.info("Loaded %d archived research sessions from %s", count, self._data_dir)


class SessionNotFoundError(Exception):
    """Raised when a session ID is not found."""


class VersionConflictError(Exception):
    """Raised on plan version mismatch."""


class SessionNotTerminalError(Exception):
    """Raised when trying to delete a non-terminal session."""
