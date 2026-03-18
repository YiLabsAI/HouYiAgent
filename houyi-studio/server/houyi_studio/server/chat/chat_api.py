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
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from houyi.infrastructure.observability import ObservabilityQuery

from .chat_service import ChatService
from .import_export import ChatExporter, CherryStudioImporter
from .provider_service import get_probe
from .settings_store import GlobalSettings, SettingsStore
from .types import (
    Conversation,
    CreateConversationRequest,
    EditMessageRequest,
    Message,
    PinMessageRequest,
    SendMessageRequest,
    UpdateConversationRequest,
    UpdatePinnedContextRequest,
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


def _http_error_detail(
    *,
    error_code: str,
    public_message: str,
    status_code: int,
    retryable: bool,
    provider_code: str | None = None,
) -> dict[str, Any]:
    return {
        "error": public_message,
        "error_code": error_code,
        "public_message": public_message,
        "retryable": retryable,
        "status_code": status_code,
        "provider_code": provider_code,
    }


def _not_found_detail(message: str, error_code: str = "resource_not_found") -> dict[str, Any]:
    return _http_error_detail(
        error_code=error_code,
        public_message=message,
        status_code=404,
        retryable=False,
    )


def _bad_request_detail(message: str, error_code: str = "invalid_request") -> dict[str, Any]:
    return _http_error_detail(
        error_code=error_code,
        public_message=message,
        status_code=400,
        retryable=False,
    )


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
    return service.create_conversation(req)


@router.post("/conversations/{conversation_id}/_seed-messages")
async def seed_messages(conversation_id: str, request: Request) -> dict[str, Any]:
    """Inject messages into a conversation (test-only endpoint).

    Body: { "messages": [{"role": "user"|"assistant", "content": "..."}] }
    """
    service = _get_service()
    body = await request.json()
    try:
        return service.seed_messages(
            conversation_id,
            messages=body.get("messages", []),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(exc))) from exc


@router.get("/conversations")
async def list_conversations(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List conversations with optional filtering."""
    service = _get_service()
    return service.list_conversations(status=status, limit=limit, offset=offset)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Get a conversation with full message history."""
    service = _get_service()
    try:
        return service.get_conversation(conversation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(exc))) from exc


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    req: UpdateConversationRequest,
) -> dict[str, Any]:
    """Update conversation metadata (title, status, system_instructions, model)."""
    service = _get_service()
    try:
        return service.update_conversation(conversation_id, req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(exc))) from exc


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


def _parse_trace_json_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_trace_json_list(value: Any) -> list[Any]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_compaction_history(conversation: Conversation) -> list[dict[str, Any]]:
    history = conversation.metadata.get("compaction_history")
    if not isinstance(history, list):
        return []
    return [item for item in history if isinstance(item, dict)]


def _summarize_message_preview(message: Message, *, max_chars: int = 180) -> str:
    content = str(message.content or "").strip()
    if not content and isinstance(message.reasoning_content, str):
        content = message.reasoning_content.strip()
    if not content and isinstance(message.tool_calls, list) and message.tool_calls:
        content = f"[tool calls: {len(message.tool_calls)}]"
    if not content and isinstance(message.tool_call_id, str) and message.tool_call_id:
        content = f"[tool result: {message.tool_call_id}]"
    if not content and message.attachments:
        content = f"[attachments: {len(message.attachments)}]"
    if not content:
        content = "[empty]"
    normalized = " ".join(content.split())
    return normalized if len(normalized) <= max_chars else f"{normalized[:max_chars]}…"


def _build_message_previews(
    messages: list[Message],
    *,
    ordered_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if ordered_ids is None:
        selected = [
            message
            for message in messages
            if isinstance(message.message_id, str) and message.message_id
        ]
    else:
        by_id = {
            message.message_id: message
            for message in messages
            if isinstance(message.message_id, str) and message.message_id
        }
        selected = [by_id[message_id] for message_id in ordered_ids if message_id in by_id]
    return [
        {
            "message_id": message.message_id,
            "role": message.role.value,
            "name": message.name,
            "created_at": message.created_at,
            "preview": _summarize_message_preview(message),
        }
        for message in selected
    ]


def _build_compaction_diff(
    *,
    current_conversation: Conversation,
    backup_conversation: Conversation | None,
    compaction: dict[str, Any],
) -> dict[str, Any]:
    current_ids = [
        message.message_id
        for message in current_conversation.messages
        if isinstance(message.message_id, str) and message.message_id
    ]
    current_id_set = set(current_ids)
    source_message_ids = [
        message_id
        for message_id in compaction.get("source_message_ids", [])
        if isinstance(message_id, str) and message_id
    ]
    if backup_conversation is None:
        return {
            "source_message_ids": source_message_ids,
            "backup_message_count": None,
            "current_message_count": len(current_conversation.messages),
            "backup_visible_message_count": None,
            "current_visible_message_count": current_conversation.visible_message_count,
            "removed_message_ids": source_message_ids,
            "added_message_ids": [],
            "source_message_previews": [],
            "added_message_previews": [],
        }
    backup_ids = [
        message.message_id
        for message in backup_conversation.messages
        if isinstance(message.message_id, str) and message.message_id
    ]
    backup_id_set = set(backup_ids)
    removed_message_ids = [
        message_id for message_id in backup_ids if message_id not in current_id_set
    ]
    added_message_ids = [
        message_id for message_id in current_ids if message_id not in backup_id_set
    ]
    return {
        "source_message_ids": source_message_ids,
        "backup_message_count": len(backup_conversation.messages),
        "current_message_count": len(current_conversation.messages),
        "backup_visible_message_count": backup_conversation.visible_message_count,
        "current_visible_message_count": current_conversation.visible_message_count,
        "removed_message_ids": removed_message_ids,
        "added_message_ids": added_message_ids,
        "source_message_previews": _build_message_previews(
            backup_conversation.messages,
            ordered_ids=source_message_ids,
        ),
        "added_message_previews": _build_message_previews(
            current_conversation.messages,
            ordered_ids=added_message_ids,
        ),
    }


def _build_compaction_history_item(
    *,
    service: ChatService,
    conversation: Conversation,
    compaction: dict[str, Any],
) -> dict[str, Any]:
    backup_id = str(compaction.get("backup_id") or "").strip()
    backup_entry = service.json_store.get_backup(backup_id) if backup_id else None
    backup_conversation: Conversation | None = None
    if backup_id:
        with contextlib.suppress(FileNotFoundError):
            backup_conversation = service.json_store.read_backup(backup_id)
    return {
        "compaction": compaction,
        "backup": backup_entry,
        "diff": _build_compaction_diff(
            current_conversation=conversation,
            backup_conversation=backup_conversation,
            compaction=compaction,
        ),
    }


def _build_trace_request_context(root_attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": root_attrs.get("chat.request_id"),
        "conversation_id": root_attrs.get("chat.conversation_id"),
        "model": root_attrs.get("chat.model"),
        "max_context_tokens": _coerce_int(root_attrs.get("chat.context_tokens_max")),
        "llm_messages_count": _coerce_int(root_attrs.get("chat.llm_messages_count")),
    }


def _build_trace_context_plan(root_attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "used_tokens": _coerce_int(root_attrs.get("chat.context_tokens_used")),
        "planned_prompt_tokens": _coerce_int(root_attrs.get("chat.context_planned_prompt_tokens")),
        "reserved_output_tokens": _coerce_int(
            root_attrs.get("chat.context_reserved_output_tokens")
        ),
        "available_input_tokens": _coerce_int(
            root_attrs.get("chat.context_available_input_tokens")
        ),
        "block_breakdown": _parse_trace_json_mapping(root_attrs.get("chat.context_blocks")),
    }


def _build_trace_context_governance(root_attrs: dict[str, Any]) -> dict[str, Any]:
    tokens_before = _coerce_int(root_attrs.get("chat.compaction.tokens_before"))
    tokens_after = _coerce_int(root_attrs.get("chat.compaction.tokens_after"))
    saved_tokens = None
    if tokens_before is not None and tokens_after is not None:
        saved_tokens = max(0, tokens_before - tokens_after)
    return {
        "dropped_blocks": _parse_trace_json_list(root_attrs.get("chat.context_dropped_blocks")),
        "drop_reasons": _parse_trace_json_mapping(root_attrs.get("chat.context_drop_reasons")),
        "dropped_block_details": _parse_trace_json_list(
            root_attrs.get("chat.context_dropped_block_details")
        ),
        "compaction": {
            "triggered": bool(root_attrs.get("chat.compaction.triggered")),
            "trigger": root_attrs.get("chat.compaction.trigger"),
            "messages_compacted": _coerce_int(root_attrs.get("chat.compaction.messages_compacted")),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "saved_tokens": saved_tokens,
            "pin_violation_count": _coerce_int(
                root_attrs.get("chat.compaction.pin_violation_count")
            ),
        },
    }


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """Get observability trace detail for Chat tool-calling timeline.

    Returns a tree-shaped span payload for UI sidebar rendering.
    """
    query = ObservabilityQuery()
    trace_view = query.get_trace(trace_id, include_content=False)
    if trace_view is None:
        logger.warning("Chat trace not found: trace_id=%s", trace_id)
        raise HTTPException(
            status_code=404,
            detail=_not_found_detail(f"Trace {trace_id} not found", "trace_not_found"),
        )

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
    root_attrs = root_span.get("attributes", {}) if isinstance(root_span, dict) else {}

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
        "request_context": _build_trace_request_context(root_attrs),
        "context_plan": _build_trace_context_plan(root_attrs),
        "context_governance": _build_trace_context_governance(root_attrs),
    }


@router.get("/conversations/{conversation_id}/compactions")
async def list_compactions(conversation_id: str) -> dict[str, Any]:
    service = _get_service()
    conversation = service.json_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=_not_found_detail(
                f"Conversation {conversation_id} not found",
                "conversation_not_found",
            ),
        )
    history = _get_compaction_history(conversation)
    items = [
        _build_compaction_history_item(
            service=service,
            conversation=conversation,
            compaction=compaction,
        )
        for compaction in reversed(history)
    ]
    return {"items": items}


@router.post("/conversations/{conversation_id}/compactions/{compaction_id}/restore")
async def restore_compaction(conversation_id: str, compaction_id: str) -> dict[str, Any]:
    service = _get_service()
    conversation = service.json_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=_not_found_detail(
                f"Conversation {conversation_id} not found",
                "conversation_not_found",
            ),
        )
    compaction = next(
        (
            item
            for item in _get_compaction_history(conversation)
            if str(item.get("compaction_id") or "") == compaction_id
        ),
        None,
    )
    if compaction is None:
        raise HTTPException(
            status_code=404,
            detail=_not_found_detail(
                f"Compaction {compaction_id} not found",
                "compaction_not_found",
            ),
        )
    backup_id = str(compaction.get("backup_id") or "").strip()
    if not backup_id:
        raise HTTPException(
            status_code=400,
            detail=_bad_request_detail(
                f"Compaction {compaction_id} has no backup",
                "compaction_backup_missing",
            ),
        )
    try:
        restore_point = service.json_store.create_backup(
            conversation_id,
            trigger="restore_point",
            metadata={
                "kind": "restore_point",
                "reason": "before_restore_compaction",
                "restored_compaction_id": compaction_id,
                "source_backup_id": backup_id,
            },
        )
        restored = service.json_store.restore_backup(backup_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(exc))) from exc
    if restored.conversation_id != conversation_id:
        raise HTTPException(
            status_code=400,
            detail=_bad_request_detail(
                f"Backup {backup_id} does not belong to {conversation_id}",
                "backup_conversation_mismatch",
            ),
        )
    service.ensure_conversation_context_state(
        restored,
        model=restored.model or service.default_model,
        persist=True,
    )
    refreshed = service.json_store.get(conversation_id) or restored
    return {
        "status": "restored",
        "restored_compaction_id": compaction_id,
        "backup_id": backup_id,
        "restore_point_backup_id": restore_point["backup_id"],
        "conversation": refreshed.model_dump(mode="json"),
    }


@router.post("/conversations/{conversation_id}/backups/{backup_id}/restore")
async def restore_backup(conversation_id: str, backup_id: str) -> dict[str, Any]:
    service = _get_service()
    conversation = service.json_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=_not_found_detail(
                f"Conversation {conversation_id} not found",
                "conversation_not_found",
            ),
        )
    backup = service.json_store.get_backup(backup_id)
    if backup is None:
        raise HTTPException(
            status_code=404,
            detail=_not_found_detail(f"Backup {backup_id} not found", "backup_not_found"),
        )
    if str(backup.get("conversation_id") or "") != conversation_id:
        raise HTTPException(
            status_code=400,
            detail=_bad_request_detail(
                f"Backup {backup_id} does not belong to {conversation_id}",
                "backup_conversation_mismatch",
            ),
        )
    try:
        restored = service.json_store.restore_backup(backup_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(exc))) from exc
    service.ensure_conversation_context_state(
        restored,
        model=restored.model or service.default_model,
        persist=True,
    )
    refreshed = service.json_store.get(conversation_id) or restored
    return {
        "status": "restored",
        "backup_id": backup_id,
        "conversation": refreshed.model_dump(mode="json"),
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, str]:
    """Delete a conversation."""
    service = _get_service()
    try:
        return service.delete_conversation(conversation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(exc))) from exc


@router.post("/conversations/{conversation_id}/compact")
async def compact_conversation(conversation_id: str) -> dict[str, Any]:
    """Run manual conversation compaction."""
    service = _get_service()
    try:
        return await service.compact_conversation(conversation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(exc))) from exc


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
        raise HTTPException(
            status_code=404,
            detail=_not_found_detail(
                f"Conversation {conversation_id} not found",
                "conversation_not_found",
            ),
        )

    async def event_generator():
        try:
            async for chunk in _iter_with_disconnect_guard(
                request=request,
                stream=service.send_message(conversation_id, req),
                disconnect_log=f"Client disconnected during streaming for {conversation_id}",
            ):
                yield chunk
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=_not_found_detail(str(e))) from e
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
        raise HTTPException(status_code=404, detail=_not_found_detail(str(e))) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=_bad_request_detail(str(e))) from e


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
    body = await request.json()
    bookmarked = body.get("bookmarked", False)
    try:
        return service.toggle_message_bookmark(
            conversation_id=conversation_id,
            message_id=message_id,
            bookmarked=bookmarked,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(exc))) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_bad_request_detail(str(exc))) from exc


@router.get("/conversations/{conversation_id}/pins")
async def list_pinned_contexts(conversation_id: str) -> dict[str, Any]:
    service = _get_service()
    try:
        pins = await service.pinned_context_store.list_pins(
            conversation_id=conversation_id,
            include_inactive=True,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(e))) from e
    return {"pins": [pin.model_dump(mode="json") for pin in pins]}


@router.post("/conversations/{conversation_id}/messages/{message_id}/pin-context")
async def pin_message_to_context(
    conversation_id: str,
    message_id: str,
    req: PinMessageRequest,
) -> dict[str, Any]:
    service = _get_service()
    try:
        pin = await service.pinned_context_store.pin_message(
            conversation_id=conversation_id,
            message_id=message_id,
            replace_pin_id=req.replace_pin_id,
            title=req.title,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(e))) from e
    return pin.model_dump(mode="json")


@router.patch("/conversations/{conversation_id}/pins/{pin_id}")
async def update_pinned_context(
    conversation_id: str,
    pin_id: str,
    req: UpdatePinnedContextRequest,
) -> dict[str, Any]:
    service = _get_service()
    try:
        pin = await service.pinned_context_store.update_pin_status(
            conversation_id=conversation_id,
            pin_id=pin_id,
            status=req.status,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=_not_found_detail(str(e))) from e
    return pin.model_dump(mode="json")


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
        raise HTTPException(status_code=404, detail=_not_found_detail(str(e))) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=_bad_request_detail(str(e))) from e


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
            raise HTTPException(status_code=404, detail=_not_found_detail(str(e))) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=_bad_request_detail(str(e))) from e
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
        raise HTTPException(
            status_code=400,
            detail=_bad_request_detail("Expected a .zip file", "invalid_import_file"),
        )

    zip_data = await file.read()
    if not zip_data:
        raise HTTPException(
            status_code=400,
            detail=_bad_request_detail("Empty file", "empty_import_file"),
        )

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
