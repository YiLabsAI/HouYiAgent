"""Research service — adapts server layer to SDK ResearchRuntime.

Manages run persistence (JSON), idempotency, and lifecycle coordination.
Runs are persisted as JSON and rehydrated on startup so that history
(including failed/timed-out runs) survives server restarts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from houyi_studio.server.research.sse import ResearchSSEEnvelope

from houyi.adapters.llm.base import LLMAdapter
from houyi.application.research.runtime import ResearchRuntime
from houyi.application.research.runtime.errors import ResearchReportNotReadyError
from houyi.application.research.runtime.intermediate import IntermediateReport
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


class _ArchivedRun:
    """Lightweight read-only run restored from persisted JSON.

    Not a real ``ResearchRuntime`` — only stores enough data for
    list / get / delete operations on historical runs.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.run_id: str = data["run_id"]
        self._status_str: str = data.get("status", "failed")
        self._plan_data: dict[str, Any] | None = data.get("plan")
        self._progress_data: dict[str, Any] = data.get("progress", {})
        self._error: str | None = data.get("error")
        self._report_data: dict[str, Any] | None = data.get("report")
        self._search_results_data: list[dict[str, Any]] = data.get("search_results", [])
        self._intermediate_reports_data: list[dict[str, Any]] = data.get("intermediate_reports", [])
        self.created_at: str = data.get("created_at", "")
        self.updated_at: float = data.get("updated_at", 0)
        self.started_at: float = float(data.get("started_at", 0) or 0)

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
    """Server-side service managing research runs.

    Provides idempotent run creation, plan editing with version control,
    execution lifecycle, and JSON-based persistence for run recovery.
    """

    MAX_CONCURRENT_RUNS = 3

    @staticmethod
    def _created_at_to_epoch(created_at: str | None) -> float:
        if not created_at:
            return 0.0
        try:
            return datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S").timestamp()
        except Exception:
            return 0.0

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
        self._runs: dict[str, ResearchRuntime | _ArchivedRun] = {}
        self._idempotency: dict[str, str] = {}
        self._emitters: dict[str, EventEmitter] = {}
        self._event_buffers: dict[str, list] = {}
        self._memory_service = memory_service
        self._load_persisted_runs()

    def _attach_buffer_listener(self, run_id: str, emitter: EventEmitter) -> None:
        """Attach an always-on listener that buffers events for late SSE subscribers."""
        buf = self._event_buffers.setdefault(run_id, [])
        seq_counter = {"n": 0}

        async def _buffer_handler(event: Any) -> None:
            research_event = event.data.get("research_event", "unknown")
            seq_counter["n"] += 1
            payload = {
                k: v for k, v in event.data.items() if k not in ("research_event", "sequence")
            }
            env = ResearchSSEEnvelope(
                event_type=research_event,
                run_id=run_id,
                sequence=seq_counter["n"],
                payload=payload,
            )
            buf.append(env)

        emitter.on_any(_buffer_handler)

    def running_run_count(self) -> int:
        """Count currently executing runs."""
        return sum(
            1
            for s in self._runs.values()
            if not isinstance(s, _ArchivedRun)
            and s.status in (ResearchStatus.EXECUTING, ResearchStatus.GENERATING_REPORT)
        )

    async def create_run(
        self,
        query: str,
        settings: ResearchSettings | None = None,
        idempotency_key: str | None = None,
        memory_context: str | None = None,
    ) -> tuple[ResearchRuntime, ResearchPlan]:
        """Create a new run or return an existing one for idempotent requests."""
        if idempotency_key and idempotency_key in self._idempotency:
            run_id = self._idempotency[idempotency_key]
            runtime = self._runs[run_id]
            return runtime, runtime.plan  # type: ignore[return-value]

        emitter = EventEmitter()
        runtime = ResearchRuntime(
            llm_adapter=self._llm,
            web_search=self._web_search,
            settings=settings,
            event_emitter=emitter,
            memory_context=memory_context,
        )
        plan = await runtime.start(query)
        self._runs[runtime.run_id] = runtime
        self._emitters[runtime.run_id] = emitter
        self._attach_buffer_listener(runtime.run_id, emitter)
        if idempotency_key:
            self._idempotency[idempotency_key] = runtime.run_id

        self._persist_run(runtime)
        return runtime, plan

    def get_run(self, run_id: str) -> ResearchRuntime | _ArchivedRun | None:
        return self._runs.get(run_id)

    def get_emitter(self, run_id: str) -> EventEmitter | None:
        return self._emitters.get(run_id)

    def get_event_buffer(self, run_id: str) -> list:
        return self._event_buffers.setdefault(run_id, [])

    async def edit_plan(
        self,
        run_id: str,
        edits: list[PlanEdit],
        client_plan_version: int | None = None,
    ) -> ResearchPlan:
        """Edit a run plan with optional version check."""
        runtime = self._require_live_run(run_id)
        if (
            client_plan_version is not None
            and runtime.plan
            and client_plan_version != runtime.plan.version
        ):
            msg = (
                f"Plan version conflict: client={client_plan_version}, "
                f"server={runtime.plan.version}"
            )
            raise VersionConflictError(msg)
        plan = await runtime.edit_plan(edits)
        self._persist_run(runtime)
        return plan

    def prepare_for_execution(self, run_id: str) -> ResearchRuntime:
        """Ensure run is live (rehydrated if archived) before scheduling.

        Must be called synchronously before ``asyncio.create_task`` so that
        the ``EventEmitter`` is registered before the frontend connects SSE.
        """
        return self._require_live_run(run_id)

    async def launch_run(self, run_id: str) -> None:
        """Confirm plan, execute, and extract memory candidates on success."""
        runtime = self._require_live_run(run_id)
        if runtime.status == ResearchStatus.EXECUTING:
            return
        await runtime.confirm_plan()
        try:
            await runtime.execute()
        finally:
            self._persist_run(runtime)
        await self._extract_and_push_memories(runtime)

    async def _extract_and_push_memories(self, runtime: ResearchRuntime) -> None:
        """Extract memory candidates from a completed run and push to MemoryService."""
        if not self._memory_service:
            return
        if runtime.status != ResearchStatus.COMPLETED:
            return
        try:
            candidates = await runtime.extract_memories()
            if candidates:
                self._memory_service.add_candidates(candidates)
                logger.info(
                    "Pushed %d memory candidates from run %s",
                    len(candidates),
                    runtime.run_id,
                )
        except Exception:
            logger.warning(
                "Memory extraction failed for run %s",
                runtime.run_id,
                exc_info=True,
            )

    async def cancel_run(self, run_id: str, reason: str = "") -> None:
        runtime = self._require_live_run(run_id)
        await runtime.cancel(reason)
        self._persist_run(runtime)

    async def get_report(self, run_id: str) -> ResearchReport:
        runtime = self._require_run(run_id)
        if isinstance(runtime, _ArchivedRun):
            if runtime.report_data:
                return ResearchReport.model_validate(runtime.report_data)
            raise ResearchReportNotReadyError("No report available for archived run")
        return await runtime.get_report()

    def get_progress(self, run_id: str) -> ResearchProgress:
        runtime = self._require_run(run_id)
        return runtime.progress

    def list_runs(
        self,
        offset: int = 0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List runs with pagination (live + archived), sorted by latest execution start."""
        items = []
        for runtime in self._runs.values():
            started_at = float(getattr(runtime, "started_at", 0) or 0)
            created_at = getattr(runtime, "created_at", None)
            activity_ts = started_at or self._created_at_to_epoch(created_at)
            items.append(
                {
                    "run_id": runtime.run_id,
                    "status": runtime.status.value,
                    "query": runtime.plan.query if runtime.plan else "",
                    "progress": runtime.progress.model_dump(),
                    "error": getattr(runtime, "_error", None) or getattr(runtime, "error", None),
                    "created_at": created_at,
                    "started_at": started_at,
                    "activity_ts": activity_ts,
                }
            )
        items.sort(key=lambda x: x["activity_ts"], reverse=True)
        for item in items:
            item.pop("activity_ts", None)
        return items[offset : offset + limit]

    async def delete_run(self, run_id: str) -> None:
        """Delete a terminal or archived run."""
        runtime = self._runs.get(run_id)
        if not runtime:
            raise RunNotFoundError(f"Run {run_id} not found")
        if runtime.status == ResearchStatus.EXECUTING:
            raise RunNotTerminalError(f"Cannot delete run in status {runtime.status.value}")
        del self._runs[run_id]
        self._emitters.pop(run_id, None)
        path = self._data_dir / f"{run_id}.json"
        if path.exists():
            path.unlink()

    def _require_live_run(self, run_id: str) -> ResearchRuntime:
        """Return a live run, rehydrating if archived."""
        raw = self._runs.get(run_id)
        if not raw:
            raise RunNotFoundError(f"Run {run_id} not found")
        if isinstance(raw, _ArchivedRun):
            return self._rehydrate(raw)
        return raw

    def _require_run(self, run_id: str) -> ResearchRuntime | _ArchivedRun:
        runtime = self._runs.get(run_id)
        if not runtime:
            raise RunNotFoundError(f"Run {run_id} not found")
        return runtime

    def _rehydrate(self, archived: _ArchivedRun) -> ResearchRuntime:
        """Convert an archived run back into a live ResearchRuntime.

        Restores plan and search_results so that retry can checkpoint
        already-completed sub-questions and skip re-searching them.
        """
        emitter = EventEmitter()
        runtime = ResearchRuntime(
            run_id=archived.run_id,
            llm_adapter=self._llm,
            web_search=self._web_search,
            event_emitter=emitter,
        )
        if archived.plan:
            plan = archived.plan
            plan.status = PlanStatus.DRAFT
            runtime._plan = plan
            runtime._status = ResearchStatus.PLAN_READY
        runtime.created_at = archived.created_at or runtime.created_at
        runtime.started_at = archived.started_at or runtime.started_at

        for i, sr_data in enumerate(archived.search_results_data):
            try:
                runtime._search_results.append(SearchResult.model_validate(sr_data))
            except Exception:
                logger.warning(
                    "Failed to restore search result [%d] for run %s",
                    i,
                    archived.run_id,
                    exc_info=True,
                )
        if runtime._search_results:
            logger.info(
                "Restored %d search results for run %s",
                len(runtime._search_results),
                archived.run_id,
            )

        for i, ir_data in enumerate(archived.intermediate_reports_data):
            try:
                runtime._intermediate_reports.append(IntermediateReport.model_validate(ir_data))
            except Exception:
                logger.warning(
                    "Failed to restore intermediate report [%d] for run %s",
                    i,
                    archived.run_id,
                    exc_info=True,
                )
        if runtime._intermediate_reports:
            logger.info(
                "Restored %d intermediate reports for run %s",
                len(runtime._intermediate_reports),
                archived.run_id,
            )

        self._runs[archived.run_id] = runtime
        self._emitters[archived.run_id] = emitter
        self._event_buffers.pop(archived.run_id, None)
        self._attach_buffer_listener(archived.run_id, emitter)
        logger.info("Rehydrated archived run %s into live runtime", archived.run_id)
        return runtime

    def _persist_run(self, runtime: ResearchRuntime) -> None:
        """Persist run state to JSON for recovery."""
        path = self._data_dir / f"{runtime.run_id}.json"
        error = getattr(runtime, "_error", None) or getattr(runtime, "error", None)
        report_data = None
        if hasattr(runtime, "_report") and runtime._report is not None:
            with contextlib.suppress(Exception):
                report_data = runtime._report.model_dump()
        search_results_data: list[dict] = []
        if hasattr(runtime, "_search_results"):
            with contextlib.suppress(Exception):
                search_results_data = [sr.model_dump() for sr in runtime._search_results]
        intermediate_data: list[dict] = []
        if hasattr(runtime, "_intermediate_reports"):
            with contextlib.suppress(Exception):
                intermediate_data = [ir.model_dump() for ir in runtime._intermediate_reports]
        data = {
            "run_id": runtime.run_id,
            "status": runtime.status.value,
            "plan": runtime.plan.model_dump() if runtime.plan else None,
            "progress": runtime.progress.model_dump(),
            "error": error,
            "report": report_data,
            "search_results": search_results_data,
            "intermediate_reports": intermediate_data,
            "created_at": getattr(runtime, "created_at", ""),
            "updated_at": time.time(),
            "started_at": float(getattr(runtime, "started_at", 0) or 0),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        tmp.rename(path)

    def _load_persisted_runs(self) -> None:
        """Load archived runs from JSON files on startup."""
        count = 0
        for path in self._data_dir.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                run_id = raw.get("run_id")
                if not run_id or run_id in self._runs:
                    continue
                run_id = str(run_id)
                data = dict(raw)
                data["run_id"] = run_id
                self._runs[run_id] = _ArchivedRun(data)
                count += 1
            except Exception:
                logger.warning("Failed to load run from %s", path, exc_info=True)
        if count:
            logger.info("Loaded %d archived research runs from %s", count, self._data_dir)


class RunNotFoundError(Exception):
    """Raised when a run ID is not found."""


class VersionConflictError(Exception):
    """Raised on plan version mismatch."""


class RunNotTerminalError(Exception):
    """Raised when trying to delete a non-terminal run."""
