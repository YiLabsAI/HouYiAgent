"""Chat API: FastAPI router for chat endpoints.

Provides REST endpoints for conversation CRUD and SSE streaming for messages.

API spec:
  POST   /api/chat/conversations                            — Create conversation
  GET    /api/chat/conversations                            — List conversations
  GET    /api/chat/conversations/{id}                       — Get conversation
  PATCH  /api/chat/conversations/{id}                       — Update conversation
  DELETE /api/chat/conversations/{id}                       — Delete conversation
  POST   /api/chat/conversations/{id}/messages              — Send message (SSE stream)
  PUT    /api/chat/conversations/{id}/messages/{msg_id}     — Edit message
  DELETE /api/chat/conversations/{id}/messages/{msg_id}     — Delete message
  POST   /api/chat/conversations/{id}/messages/{msg_id}/regenerate — Regenerate (SSE stream)
  GET    /api/chat/trace/{trace_id}                         — Query observability trace detail
  GET    /api/chat/export                                   — Export all conversations
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from houyi.observability.query import ObservabilityQuery

from .chat_service import ChatService
from .import_export import ChatExporter, CherryStudioImporter
from .provider_service import get_probe
from .settings_store import GlobalSettings, SettingsStore
from .types import (
    Conversation,
    CreateConversationRequest,
    EditMessageRequest,
    Message,
    SendMessageRequest,
    UpdateConversationRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Module-level service references, set by register_chat_routes()
_chat_service: ChatService | None = None
_settings_store: SettingsStore | None = None


def register_chat_routes(
    chat_service: ChatService,
    settings_store: SettingsStore | None = None,
) -> APIRouter:
    """Register chat routes with a ChatService instance.

    Args:
        chat_service: Initialized ChatService.

    Returns:
        The configured APIRouter.
    """
    global _chat_service, _settings_store
    _chat_service = chat_service
    _settings_store = settings_store
    return router


def _get_service() -> ChatService:
    """Get the chat service, raising if not initialized."""
    if _chat_service is None:
        raise RuntimeError("ChatService not initialized. Call register_chat_routes() first.")
    return _chat_service


def _get_settings_store() -> SettingsStore:
    if _settings_store is None:
        raise RuntimeError(
            "SettingsStore not initialized. Call register_chat_routes(..., settings_store=...) first."
        )
    return _settings_store


async def _iter_with_disconnect_guard(
    *,
    request: Request,
    stream: Any,
    disconnect_log: str,
) -> Any:
    """Yield SSE chunks while reacting quickly to client disconnects.

    Some upstream stream steps (e.g. tool loops) can take a long time before the
    next chunk is produced. This guard polls disconnect status while waiting for
    the next item and closes the upstream iterator as soon as the client aborts.
    """
    iterator = stream.__aiter__()
    while True:
        next_task = asyncio.create_task(iterator.__anext__())
        disconnected = False
        try:
            while True:
                done, _ = await asyncio.wait({next_task}, timeout=0.25)
                if done:
                    break
                if await request.is_disconnected():
                    disconnected = True
                    logger.info(disconnect_log)
                    next_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                        await next_task
                    aclose = getattr(iterator, "aclose", None)
                    if callable(aclose):
                        await aclose()
                    return

            try:
                chunk = next_task.result()
            except StopAsyncIteration:
                return
            yield chunk
        finally:
            if not disconnected and not next_task.done():
                next_task.cancel()


# --- Conversation CRUD ---


@router.post("/conversations", status_code=201)
async def create_conversation(req: CreateConversationRequest) -> dict[str, Any]:
    """Create a new conversation."""
    service = _get_service()
    conversation = Conversation(
        title=req.title,
        model=req.model,
        system_instructions=req.system_instructions,
        metadata=req.metadata,
    )
    created = service.json_store.create(conversation)
    return created.to_summary()


@router.post("/conversations/{conversation_id}/_seed-messages")
async def seed_messages(conversation_id: str, request: Request) -> dict[str, Any]:
    """Inject messages into a conversation (test-only endpoint).

    Body: { "messages": [{"role": "user"|"assistant", "content": "..."}] }
    """
    service = _get_service()
    conversation = service.json_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    body = await request.json()
    for msg_data in body.get("messages", []):
        conversation.messages.append(
            Message(
                role=msg_data["role"],
                content=msg_data["content"],
            )
        )
    service.json_store.update(conversation)
    return {"seeded": len(body.get("messages", []))}


@router.get("/conversations")
async def list_conversations(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List conversations with optional filtering."""
    service = _get_service()
    conversations = service.json_store.list_conversations(
        status=status,
        limit=limit,
        offset=offset,
    )
    total = service.json_store.count(status=status)
    return {
        "conversations": conversations,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Get a conversation with full message history."""
    service = _get_service()
    conversation = service.json_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    return conversation.model_dump(mode="json")


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    req: UpdateConversationRequest,
) -> dict[str, Any]:
    """Update conversation metadata (title, status, system_instructions, model)."""
    service = _get_service()
    conversation = service.json_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

    if req.title is not None:
        conversation.title = req.title
    if req.status is not None:
        conversation.status = req.status
    if req.system_instructions is not None:
        conversation.system_instructions = req.system_instructions
    if req.model is not None:
        conversation.model = req.model
    # For numeric params, distinguish between "not sent" (field absent from JSON)
    # and "explicitly set to null" (reset to global default).
    # Pydantic v2: field is None both when absent and when explicitly null,
    # so we check the raw request body to detect explicit null.
    raw_body = req.model_dump(exclude_unset=True)
    if "temperature" in raw_body:
        conversation.temperature = req.temperature
    if "max_tokens" in raw_body:
        conversation.max_tokens = req.max_tokens
    if "top_p" in raw_body:
        conversation.top_p = req.top_p
    if "stream" in raw_body:
        conversation.stream = req.stream
    if req.bookmarked is not None:
        conversation.bookmarked = req.bookmarked

    updated = service.json_store.update(conversation)
    return updated.to_summary()


@router.get("/conversations/{conversation_id}/context-usage")
async def get_context_usage(conversation_id: str) -> dict[str, Any]:
    """Get context window usage for a conversation (token counts)."""
    service = _get_service()
    usage = service.get_context_usage(conversation_id)
    if usage is None:
        return {"usage": None}
    return {"usage": usage}


def _span_to_tree_node(span: Any, children: list[dict[str, Any]]) -> dict[str, Any]:
    duration_ms = 0.0
    if span.end_time is not None:
        duration_ms = max(0.0, (span.end_time - span.start_time) * 1000)
    events = []
    for event in getattr(span, "events", []) or []:
        events.append(
            {
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": event.attributes,
            }
        )
    return {
        "name": span.name,
        "span_type": getattr(span, "span_type", None),
        "start_time_ms": span.start_time * 1000,
        "duration_ms": duration_ms,
        "status": span.status,
        "attributes": span.attributes,
        "events": events,
        "children": children,
    }


def _build_trace_tree(spans: list[Any]) -> dict[str, Any] | None:
    if not spans:
        return None

    span_by_id = {span.span_id: span for span in spans}
    children_map: dict[str, list[Any]] = {span.span_id: [] for span in spans}
    roots: list[Any] = []

    for span in spans:
        if span.parent_id and span.parent_id in span_by_id:
            children_map[span.parent_id].append(span)
        else:
            roots.append(span)

    for child_list in children_map.values():
        child_list.sort(key=lambda x: x.start_time)
    roots.sort(key=lambda x: x.start_time)

    def _to_node(span: Any) -> dict[str, Any]:
        children = [_to_node(child) for child in children_map.get(span.span_id, [])]
        return _span_to_tree_node(span, children)

    return _to_node(roots[0]) if roots else None


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """Get observability trace detail for Chat tool-calling timeline.

    Returns a tree-shaped span payload for UI sidebar rendering.
    """
    query = ObservabilityQuery()
    trace_view = query.get_trace(trace_id, include_content=False)
    if trace_view is None:
        logger.warning("Chat trace not found: trace_id=%s", trace_id)
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    root_span = _build_trace_tree(trace_view.spans)

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    llm_span_count = 0
    llm_spans_with_usage = 0
    for span in trace_view.spans:
        if getattr(span, "span_type", None) != "llm":
            continue
        llm_span_count += 1
        if span.tokens is None:
            continue
        llm_spans_with_usage += 1
        prompt_tokens += span.tokens.input
        completion_tokens += span.tokens.output
        total_tokens += span.tokens.total

    logger.info(
        "Chat trace fetched: trace_id=%s spans=%d root=%s duration_ms=%.2f",
        trace_view.trace_id,
        len(trace_view.spans),
        bool(root_span),
        trace_view.total_duration_ms or 0.0,
    )

    return {
        "trace_id": trace_view.trace_id,
        "root_span": root_span,
        "total_duration_ms": trace_view.total_duration_ms or 0.0,
        "total_tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "llm_spans": llm_span_count,
            "llm_spans_with_usage": llm_spans_with_usage,
            "is_partial": llm_span_count > llm_spans_with_usage,
        },
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, str]:
    """Delete a conversation."""
    service = _get_service()
    deleted = service.json_store.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    return {"status": "deleted", "conversation_id": conversation_id}


# --- Message streaming ---


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    req: SendMessageRequest,
    request: Request,
) -> StreamingResponse:
    """Send a message and stream the assistant response as SSE.

    The response is a Server-Sent Events stream (text/event-stream).
    Events: message.delta, message.finish, message.error, message.aborted, context.usage.

    Supports AbortController: client can abort the request to trigger message.aborted.
    """
    service = _get_service()

    # Verify conversation exists
    conversation = service.json_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

    async def event_generator():
        try:
            async for chunk in _iter_with_disconnect_guard(
                request=request,
                stream=service.send_message(conversation_id, req),
                disconnect_log=f"Client disconnected during streaming for {conversation_id}",
            ):
                yield chunk
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.error("Stream error for %s: %s", conversation_id, e, exc_info=True)
            # Error event already sent by sse_adapter; just stop

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# --- Message operations ---


@router.put("/conversations/{conversation_id}/messages/{message_id}")
async def edit_message(
    conversation_id: str,
    message_id: str,
    req: EditMessageRequest,
) -> dict[str, Any]:
    """Edit a user message's content.

    Only user messages can be edited. Returns the updated message.
    """
    service = _get_service()
    try:
        msg = await service.edit_message(conversation_id, message_id, req)
        return msg.model_dump(mode="json")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.patch("/conversations/{conversation_id}/messages/{message_id}/bookmark")
async def toggle_message_bookmark(
    conversation_id: str,
    message_id: str,
    request: Request,
) -> dict[str, Any]:
    """Toggle bookmark on a message.

    Body: { "bookmarked": true|false }
    Returns the updated message.
    """
    service = _get_service()
    conversation = service.json_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    body = await request.json()
    bookmarked = body.get("bookmarked", False)
    msg = next((m for m in conversation.messages if m.message_id == message_id), None)
    if msg is None:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
    msg.bookmarked = bookmarked
    service.json_store.update(conversation)
    return msg.model_dump(mode="json")


@router.delete("/conversations/{conversation_id}/messages/{message_id}")
async def delete_message(
    conversation_id: str,
    message_id: str,
) -> dict[str, str]:
    """Delete a single message from a conversation."""
    service = _get_service()
    try:
        await service.delete_message(conversation_id, message_id)
        return {"status": "deleted", "message_id": message_id}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/conversations/{conversation_id}/messages/{message_id}/regenerate")
async def regenerate_message(
    conversation_id: str,
    message_id: str,
    request: Request,
) -> StreamingResponse:
    """Regenerate an assistant message.

    Removes the target assistant message and all subsequent messages,
    then re-sends the preceding user message. Returns an SSE stream.
    """
    service = _get_service()

    async def event_generator():
        try:
            async for chunk in _iter_with_disconnect_guard(
                request=request,
                stream=service.regenerate_message(conversation_id, message_id),
                disconnect_log=f"Client disconnected during regenerate for {conversation_id}/{message_id}",
            ):
                yield chunk
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(
                "Regenerate error for %s/%s: %s", conversation_id, message_id, e, exc_info=True
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Bookmarks ---


@router.get("/bookmarks")
async def get_bookmarks() -> dict[str, Any]:
    """Get all bookmarked conversations and messages.

    Returns a flat list of bookmark entries sorted by created_at descending.
    Each entry has type="conversation" or type="message".
    """
    service = _get_service()
    results = service.json_store.get_bookmarks()
    return {"bookmarks": results}


# --- Search ---


@router.get("/search")
async def search_conversations(q: str = "", limit: int = 20) -> dict[str, Any]:
    """Full-text search across conversation titles and message content.

    Args:
        q: Search query string.
        limit: Max results (default 20).

    Returns:
        Dict with 'results' list of search hits.
    """
    service = _get_service()
    results = service.json_store.search(q, limit=limit)
    return {"results": results, "query": q}


# --- Settings ---


def _get_settings_store() -> SettingsStore:
    """Get the settings store, raising if not initialized."""
    if _settings_store is None:
        raise RuntimeError(
            "SettingsStore not initialized. Pass settings_store to register_chat_routes()."
        )
    return _settings_store


@router.get("/settings")
async def get_settings() -> dict[str, Any]:
    """Get global settings."""
    store = _get_settings_store()
    return store.get().model_dump(mode="json")


@router.put("/settings")
async def update_settings(settings: GlobalSettings) -> dict[str, Any]:
    """Update global settings (full replace)."""
    store = _get_settings_store()
    updated = store.update(settings)
    # Invalidate adapter cache so model routing picks up new provider config
    service = _get_service()
    service.invalidate_adapter_cache()
    return updated.model_dump(mode="json")


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """Get available models from all enabled providers."""
    store = _get_settings_store()
    models = store.get_available_models()
    return {"models": models}


# --- Provider health checks & model discovery ---
# Delegated to provider_service.py for clean abstraction and testability.
# See ProviderProbe (ABC), VertexAIProbe, OpenAICompatProbe.
@router.post("/providers/test")
async def test_provider_connection(request: Request) -> dict[str, Any]:
    """Test connection to an LLM provider.

    Body: { "base_url": str, "api_key": str, "provider_id": str? }
    Returns: { "ok": bool, "message": str, "latency_ms": int }
    """
    body = await request.json()
    base_url = body.get("base_url", "").rstrip("/")
    api_key = body.get("api_key", "")
    provider_id = body.get("provider_id", "")

    probe = get_probe(provider_id, base_url)
    return await probe.test_connection(base_url, api_key)


@router.post("/providers/fetch-models")
async def fetch_provider_models(request: Request) -> dict[str, Any]:
    """Fetch available models from an LLM provider.

    Body: { "base_url": str, "api_key": str, "provider_id": str? }
    Returns: { "models": [{ "id": str, "owned_by": str }], "error": str|null }
    """
    body = await request.json()
    base_url = body.get("base_url", "").rstrip("/")
    api_key = body.get("api_key", "")
    provider_id = body.get("provider_id", "")

    probe = get_probe(provider_id, base_url)
    return await probe.fetch_models(base_url, api_key)


# --- Import / Export ---


@router.post("/import/cherrystudio")
async def import_cherrystudio(file: UploadFile) -> dict[str, Any]:
    """Import conversations from a CherryStudio backup zip.

    Accepts a multipart file upload of a CherryStudio backup .zip file.
    The zip must contain a top-level data.json with indexedDB.topics and
    indexedDB.message_blocks.

    Returns import result with counts and any warnings/errors.
    """
    service = _get_service()

    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Expected a .zip file")

    zip_data = await file.read()
    if not zip_data:
        raise HTTPException(status_code=400, detail="Empty file")

    importer = CherryStudioImporter(service.json_store)
    result = importer.import_from_zip(zip_data)
    return result.to_dict()


@router.get("/export")
async def export_conversations() -> StreamingResponse:
    """Export all conversations as HouyiChatWorkspace JSON.

    Returns a downloadable JSON file containing all conversations,
    settings, and metadata in the HouYiChatWorkspace format.

    Response: application/json file download.
    """
    service = _get_service()
    exporter = ChatExporter(service.json_store)
    json_bytes = exporter.export_json_bytes()

    return StreamingResponse(
        iter([json_bytes]),
        media_type="application/json",
        headers={
            "Content-Disposition": "attachment; filename=houyi-chat-export.json",
            "Content-Length": str(len(json_bytes)),
        },
    )
