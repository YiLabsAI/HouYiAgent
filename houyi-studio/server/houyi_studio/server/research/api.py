"""Research API: FastAPI router for Deep Research endpoints.

Endpoints:
  POST   /api/research/runs                              — Create run
  GET    /api/research/runs                              — List runs
  GET    /api/research/runs/{id}                         — Get run
  PUT    /api/research/runs/{id}/plan                    — Edit plan
  POST   /api/research/runs/{id}/start                   — Start execution
  GET    /api/research/runs/{id}/report                  — Get report
  POST   /api/research/runs/{id}/cancel                  — Cancel run
  DELETE /api/research/runs/{id}                         — Delete run
  GET    /api/research/runs/{id}/events                  — SSE event stream
  GET    /api/agents/types                               — Agent types
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from houyi.application.research.types import (
    PlanEdit,
    ResearchSettings,
)

from .service import (
    ResearchService,
    RunNotFoundError,
    RunNotTerminalError,
    VersionConflictError,
)
from .sse import research_sse_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])
agents_router = APIRouter(prefix="/api/agents", tags=["agents"])


# ---------------------------------------------------------------------------
# Request / Response DTOs
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    query: str
    settings: ResearchSettings | None = None
    idempotency_key: str | None = None
    memory_context: str | None = None


class EditPlanRequest(BaseModel):
    edits: list[PlanEdit]
    client_plan_version: int | None = None


class ExecuteRequest(BaseModel):
    resume_if_running: bool = False
    confirm_plan_version: int | None = None


class CancelRequest(BaseModel):
    reason: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/runs", status_code=201)
async def create_run(
    body: CreateSessionRequest,
    request: Request,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    try:
        runtime, plan = await svc.create_run(
            query=body.query,
            settings=body.settings,
            idempotency_key=body.idempotency_key,
            memory_context=body.memory_context,
        )
    except Exception as exc:
        logger.error("create_run failed: %s", exc, exc_info=True)
        raise HTTPException(502, detail=f"LLM/planning error: {exc}") from exc
    return {
        "run_id": runtime.run_id,
        "plan": plan.model_dump(),
        "status": runtime.status.value,
    }


@router.get("/runs")
async def list_runs(
    request: Request,
    offset: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    items = svc.list_runs(offset=offset, limit=limit)
    return {"runs": items, "offset": offset, "limit": limit}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    runtime = svc.get_run(run_id)
    if not runtime:
        raise HTTPException(404, detail="run_not_found")
    try:
        from .service import _ArchivedRun

        search_results = None
        if isinstance(runtime, _ArchivedRun):
            sr_data = runtime.search_results_data
            if sr_data:
                search_results = sr_data
        elif hasattr(runtime, "_search_results") and runtime._search_results:
            search_results = [sr.model_dump() for sr in runtime._search_results]

        return {
            "run_id": runtime.run_id,
            "status": runtime.status.value,
            "plan": runtime.plan.model_dump() if runtime.plan else None,
            "progress": runtime.progress.model_dump(),
            "error": getattr(runtime, "_error", None) or getattr(runtime, "error", None),
            "search_results": search_results,
        }
    except Exception as exc:
        logger.error("get_run serialization failed: %s", exc, exc_info=True)
        raise HTTPException(500, detail=f"run data error: {exc}") from exc


@router.put("/runs/{run_id}/plan")
async def edit_plan(
    run_id: str,
    body: EditPlanRequest,
    request: Request,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    try:
        plan = await svc.edit_plan(
            run_id,
            body.edits,
            client_plan_version=body.client_plan_version,
        )
    except RunNotFoundError as exc:
        raise HTTPException(404, detail="run_not_found") from exc
    except VersionConflictError as exc:
        raise HTTPException(409, detail="plan_version_conflict") from exc
    return {"plan": plan.model_dump()}


@router.post("/runs/{run_id}/start", status_code=202)
async def start_run(
    run_id: str,
    body: ExecuteRequest,
    request: Request,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    runtime = svc.get_run(run_id)
    if not runtime:
        raise HTTPException(404, detail="run_not_found")

    from houyi.application.research.types import ResearchStatus

    if runtime.status == ResearchStatus.EXECUTING:
        if body.resume_if_running:
            return {"run_id": run_id, "status": runtime.status.value}
        raise HTTPException(409, detail="run_already_executing")

    running = svc.running_run_count()
    if running >= svc.MAX_CONCURRENT_RUNS:
        raise HTTPException(
            429,
            detail=f"max_concurrent_runs_reached: {running}/{svc.MAX_CONCURRENT_RUNS} running",
        )

    if (
        body.confirm_plan_version is not None
        and runtime.plan
        and body.confirm_plan_version != runtime.plan.version
    ):
        raise HTTPException(
            409,
            detail=f"plan_version_conflict: client={body.confirm_plan_version}, server={runtime.plan.version}",
        )

    import asyncio

    svc.prepare_for_execution(run_id)
    task = asyncio.create_task(svc.launch_run(run_id))
    request.app.state.research_tasks = getattr(request.app.state, "research_tasks", {})
    request.app.state.research_tasks[run_id] = task
    return {"run_id": run_id, "status": "executing"}


@router.get("/runs/{run_id}/report")
async def get_report(run_id: str, request: Request) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    try:
        report = await svc.get_report(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(404, detail="run_not_found") from exc
    except RuntimeError as exc:
        raise HTTPException(409, detail="report_not_ready") from exc
    return {"report": report.model_dump()}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    body: CancelRequest,
    request: Request,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    try:
        await svc.cancel_run(run_id, body.reason)
    except RunNotFoundError as exc:
        raise HTTPException(404, detail="run_not_found") from exc

    tasks: dict = getattr(request.app.state, "research_tasks", {})
    task = tasks.pop(run_id, None)
    if task is not None and not task.done():
        task.cancel()

    return {"status": "cancelled"}


@router.delete("/runs/{run_id}", status_code=204)
async def delete_run(run_id: str, request: Request) -> None:
    svc: ResearchService = request.app.state.research_service
    try:
        await svc.delete_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(404, detail="run_not_found") from exc
    except RunNotTerminalError as exc:
        raise HTTPException(409, detail="run_not_terminal") from exc


@router.get("/runs/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    last_event_id: str | None = None,
) -> StreamingResponse:
    svc: ResearchService = request.app.state.research_service
    emitter = svc.get_emitter(run_id)
    if not emitter:
        raise HTTPException(404, detail="run_not_found")
    event_buffer = svc.get_event_buffer(run_id)
    return StreamingResponse(
        research_sse_stream(
            emitter,
            run_id,
            last_event_id=last_event_id,
            event_buffer=event_buffer,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Agent Types (S4-8)
# ---------------------------------------------------------------------------

_AGENT_TYPES = [
    {
        "id": "deep_research",
        "name": "Deep Research",
        "description": "Multi-step research with planning, search, and structured reports.",
        "icon": "🔍",
        "available": True,
    },
    {
        "id": "code_analyst",
        "name": "Code Analyst",
        "description": "Analyze codebases, find bugs, suggest improvements, and generate documentation.",
        "icon": "💻",
        "available": False,
    },
    {
        "id": "personal_office",
        "name": "Personal Office",
        "description": "Manage schedules, draft emails, organize files, and handle daily productivity workflows.",
        "icon": "📋",
        "available": False,
    },
    {
        "id": "data_analysis",
        "name": "Data Analysis",
        "description": "Connect to databases and datasets, run queries, visualize trends, and produce insights.",
        "icon": "📊",
        "available": False,
    },
]


@agents_router.get("/types")
async def get_agent_types() -> dict[str, Any]:
    return {"types": _AGENT_TYPES}


def register_research_routes(research_service: ResearchService) -> tuple[APIRouter, APIRouter]:
    """Register the research service and return routers for app inclusion."""
    router.app = None  # type: ignore[attr-defined]

    async def _inject_service(request: Request) -> None:
        request.app.state.research_service = research_service

    return router, agents_router
