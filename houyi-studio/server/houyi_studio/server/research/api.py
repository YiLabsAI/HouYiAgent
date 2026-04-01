"""Research API: FastAPI router for Deep Research endpoints.

Endpoints:
  POST   /api/research/sessions                          — Create session
  GET    /api/research/sessions                          — List sessions
  GET    /api/research/sessions/{id}                     — Get session
  PUT    /api/research/sessions/{id}/plan                — Edit plan
  POST   /api/research/sessions/{id}/execute             — Start execution
  GET    /api/research/sessions/{id}/report              — Get report
  POST   /api/research/sessions/{id}/cancel              — Cancel session
  DELETE /api/research/sessions/{id}                     — Delete session
  GET    /api/research/sessions/{id}/events              — SSE event stream
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
    SessionNotFoundError,
    SessionNotTerminalError,
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


@router.post("/sessions", status_code=201)
async def create_session(
    body: CreateSessionRequest,
    request: Request,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    session, plan = await svc.create_session(
        query=body.query,
        settings=body.settings,
        idempotency_key=body.idempotency_key,
        memory_context=body.memory_context,
    )
    return {
        "session_id": session.session_id,
        "plan": plan.model_dump(),
        "status": session.status.value,
    }


@router.get("/sessions")
async def list_sessions(
    request: Request,
    offset: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    items = svc.list_sessions(offset=offset, limit=limit)
    return {"sessions": items, "offset": offset, "limit": limit}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    session = svc.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="session_not_found")
    return {
        "session_id": session.session_id,
        "status": session.status.value,
        "plan": session.plan.model_dump() if session.plan else None,
        "progress": session.progress.model_dump(),
        "error": getattr(session, "_error", None) or getattr(session, "error", None),
    }


@router.put("/sessions/{session_id}/plan")
async def edit_plan(
    session_id: str,
    body: EditPlanRequest,
    request: Request,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    try:
        plan = await svc.edit_plan(
            session_id,
            body.edits,
            client_plan_version=body.client_plan_version,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(404, detail="session_not_found") from exc
    except VersionConflictError as exc:
        raise HTTPException(409, detail="plan_version_conflict") from exc
    return {"plan": plan.model_dump()}


@router.post("/sessions/{session_id}/execute", status_code=202)
async def execute_session(
    session_id: str,
    body: ExecuteRequest,
    request: Request,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    session = svc.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="session_not_found")

    from houyi.application.research.types import ResearchStatus

    if session.status == ResearchStatus.EXECUTING:
        if body.resume_if_running:
            return {"session_id": session_id, "status": session.status.value}
        raise HTTPException(409, detail="session_already_executing")

    running = svc.running_session_count()
    if running >= svc.MAX_CONCURRENT_SESSIONS:
        raise HTTPException(
            429,
            detail=f"max_concurrent_sessions_reached: {running}/{svc.MAX_CONCURRENT_SESSIONS} running",
        )

    if (
        body.confirm_plan_version is not None
        and session.plan
        and body.confirm_plan_version != session.plan.version
    ):
        raise HTTPException(
            409,
            detail=f"plan_version_conflict: client={body.confirm_plan_version}, server={session.plan.version}",
        )

    import asyncio

    svc.prepare_for_execution(session_id)
    task = asyncio.create_task(svc.confirm_and_execute(session_id))
    request.app.state.research_tasks = getattr(request.app.state, "research_tasks", {})
    request.app.state.research_tasks[session_id] = task
    return {"session_id": session_id, "status": "executing"}


@router.get("/sessions/{session_id}/report")
async def get_report(session_id: str, request: Request) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    try:
        report = await svc.get_report(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(404, detail="session_not_found") from exc
    except RuntimeError as exc:
        raise HTTPException(409, detail="report_not_ready") from exc
    return {"report": report.model_dump()}


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(
    session_id: str,
    body: CancelRequest,
    request: Request,
) -> dict[str, Any]:
    svc: ResearchService = request.app.state.research_service
    try:
        await svc.cancel_session(session_id, body.reason)
    except SessionNotFoundError as exc:
        raise HTTPException(404, detail="session_not_found") from exc

    tasks: dict = getattr(request.app.state, "research_tasks", {})
    task = tasks.pop(session_id, None)
    if task is not None and not task.done():
        task.cancel()

    return {"status": "cancelled"}


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> None:
    svc: ResearchService = request.app.state.research_service
    try:
        await svc.delete_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(404, detail="session_not_found") from exc
    except SessionNotTerminalError as exc:
        raise HTTPException(409, detail="session_not_terminal") from exc


@router.get("/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    last_event_id: str | None = None,
) -> StreamingResponse:
    svc: ResearchService = request.app.state.research_service
    emitter = svc.get_emitter(session_id)
    if not emitter:
        raise HTTPException(404, detail="session_not_found")
    return StreamingResponse(
        research_sse_stream(emitter, session_id, last_event_id=last_event_id),
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
