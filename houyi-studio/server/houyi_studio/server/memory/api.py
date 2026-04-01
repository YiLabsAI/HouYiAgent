"""Memory API: FastAPI router for memory candidate and record management.

Endpoints:
  GET    /api/memory/candidates                    — List candidates
  POST   /api/memory/candidates/{id}/approve       — Approve candidate
  POST   /api/memory/candidates/{id}/reject        — Reject candidate
  PUT    /api/memory/candidates/{id}               — Edit candidate
  GET    /api/memory/records                       — List records
  PUT    /api/memory/records/{id}                  — Edit record
  DELETE /api/memory/records/{id}                  — Delete record
  GET    /api/memory/records/{id}/recalls          — Recall history
  POST   /api/memory/extract                       — Extract memories from messages
  GET    /api/memory/config                        — Get memory config
  PUT    /api/memory/config                        — Update memory config
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .service import MemoryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


class UpdateCandidateRequest(BaseModel):
    content: str | None = None
    suggested_tags: list[str] | None = None


class UpdateRecordRequest(BaseModel):
    content: str | None = None
    tags: list[str] | None = None


class ExtractMemoriesRequest(BaseModel):
    messages: list[dict[str, Any]]
    session_id: str | None = None


class MemoryConfigUpdate(BaseModel):
    enabled: bool | None = None
    auto_extract: bool | None = None


# ---------------------------------------------------------------------------
# Candidate endpoints
# ---------------------------------------------------------------------------


@router.get("/candidates")
async def list_candidates(
    request: Request,
    status: str | None = None,
) -> dict[str, Any]:
    svc: MemoryService = request.app.state.memory_service
    from houyi.adapters.memory.types import CandidateStatus

    filter_status = CandidateStatus(status) if status else None
    candidates = svc.list_candidates(status=filter_status)
    return {"candidates": [c.model_dump() for c in candidates]}


@router.post("/candidates/{candidate_id}/approve")
async def approve_candidate(
    candidate_id: str,
    request: Request,
) -> dict[str, Any]:
    svc: MemoryService = request.app.state.memory_service
    record = await svc.approve_candidate(candidate_id)
    if not record:
        raise HTTPException(404, detail="candidate_not_found")
    return {"record": record.model_dump()}


@router.post("/candidates/{candidate_id}/reject")
async def reject_candidate(
    candidate_id: str,
    request: Request,
) -> dict[str, Any]:
    svc: MemoryService = request.app.state.memory_service
    ok = await svc.reject_candidate(candidate_id)
    if not ok:
        raise HTTPException(404, detail="candidate_not_found")
    return {"status": "rejected"}


@router.put("/candidates/{candidate_id}")
async def update_candidate(
    candidate_id: str,
    body: UpdateCandidateRequest,
    request: Request,
) -> dict[str, Any]:
    svc: MemoryService = request.app.state.memory_service
    candidate = await svc.update_candidate(
        candidate_id,
        content=body.content,
        suggested_tags=body.suggested_tags,
    )
    if not candidate:
        raise HTTPException(404, detail="candidate_not_found")
    return {"candidate": candidate.model_dump()}


# ---------------------------------------------------------------------------
# Record endpoints
# ---------------------------------------------------------------------------


@router.get("/records")
async def list_records(
    request: Request,
    scope: str | None = None,
) -> dict[str, Any]:
    svc: MemoryService = request.app.state.memory_service
    from houyi.adapters.memory.types import MemoryScope

    filter_scope = MemoryScope(scope) if scope else None
    records = svc.list_records(scope=filter_scope)
    return {"records": [r.model_dump() for r in records]}


@router.put("/records/{record_id}")
async def update_record(
    record_id: str,
    body: UpdateRecordRequest,
    request: Request,
) -> dict[str, Any]:
    svc: MemoryService = request.app.state.memory_service
    record = await svc.update_record(
        record_id,
        content=body.content,
        tags=body.tags,
    )
    if not record:
        raise HTTPException(404, detail="record_not_found")
    return {"record": record.model_dump()}


@router.delete("/records/{record_id}", status_code=204)
async def delete_record(record_id: str, request: Request) -> None:
    svc: MemoryService = request.app.state.memory_service
    ok = await svc.delete_record(record_id)
    if not ok:
        raise HTTPException(404, detail="record_not_found")


@router.get("/records/{record_id}/recalls")
async def get_recall_history(
    record_id: str,
    request: Request,
) -> dict[str, Any]:
    svc: MemoryService = request.app.state.memory_service
    history = await svc.get_recall_history(record_id)
    return {"recalls": history}


# ---------------------------------------------------------------------------
# Chat → Memory extraction
# ---------------------------------------------------------------------------


@router.post("/extract")
async def extract_memories(
    body: ExtractMemoriesRequest,
    request: Request,
) -> dict[str, Any]:
    """Extract memory candidates from chat messages via MemoryEngine."""
    svc: MemoryService = request.app.state.memory_service
    engine = getattr(request.app.state, "memory_engine", None)
    if not engine:
        raise HTTPException(503, detail="memory_engine_not_available")

    from houyi.adapters.memory.types import ExtractionContext

    ctx = ExtractionContext(session_id=body.session_id) if body.session_id else None
    candidates = await engine.process_messages(body.messages, context=ctx)
    if candidates:
        svc.add_candidates(candidates)
    return {"candidates": [c.model_dump() for c in candidates], "count": len(candidates)}


# ---------------------------------------------------------------------------
# Memory configuration
# ---------------------------------------------------------------------------

_memory_config: dict[str, Any] = {"enabled": True, "auto_extract": True}


@router.get("/config")
async def get_memory_config() -> dict[str, Any]:
    return {"config": _memory_config}


@router.put("/config")
async def update_memory_config(body: MemoryConfigUpdate) -> dict[str, Any]:
    if body.enabled is not None:
        _memory_config["enabled"] = body.enabled
    if body.auto_extract is not None:
        _memory_config["auto_extract"] = body.auto_extract
    return {"config": _memory_config}
