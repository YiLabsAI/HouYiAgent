"""FastAPI application for console WebSocket server."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time as _time
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from houyi.adapters.llm import DEFAULT_MODEL
from houyi.domain.skill.registry import DEFAULT_SKILL_REGISTRY
from houyi.infrastructure.config import (
    ENV_CHAT_DATA_DIR,
    ENV_CHAT_SETTINGS_PATH,
    ENV_CHAT_SYSTEM_PROMPT,
    ENV_DEEPSEEK_MODEL,
    ENV_DEFAULT_LLM_PROVIDER,
    ENV_EMBEDDING_MODEL,
    ENV_EMBEDDING_PROVIDER,
    ENV_GEMINI_MODEL,
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_APPLICATION_CREDENTIALS,
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_HOUYI_PORT,
    ENV_SILICONFLOW_API_KEY,
    ENV_SILICONFLOW_BASE_URL,
)
from houyi.interface.protocol.ir import ExecutionStatus, PlanIR

from ..chat.chat_api import register_chat_routes
from ..chat.chat_service import ChatService
from ..chat.json_store import JsonStore
from ..chat.settings_store import SettingsStore
from ..execution.command_handler import ExecutionCommandHandler
from ..execution.engine import ExecutionEngine
from ..logging_config import (
    build_logging_config,
    configure_logging,
    get_log_level,
    truncate_payload,
)
from ..rag import get_knowledge_service, resolve_embedding_config
from ..skill.command_handler import SkillCommandHandler
from ..skill.startup_hooks import register_console_skills
from .command_dispatcher import CommandDispatcher
from .command_handler import CommandHandler
from .command_parser import CommandParser
from .commands import (
    ClientCommand,
    PlanPatch,
)
from .events import LogLevelEvent
from .websocket import connection_manager

# Load environment variables from .env file
load_dotenv()

LOG_LEVEL = configure_logging()
logger = logging.getLogger(__name__)

execution_engine: ExecutionEngine | None = None
command_parser = CommandParser(logger=logger)
command_dispatcher = CommandDispatcher()
skill_command_handler = SkillCommandHandler(
    send_event=connection_manager.send_event,
    logger=logger,
)
command_handler = CommandHandler(
    send_event=connection_manager.send_event,
    get_execution_engine=lambda: get_execution_engine(),
    sanitize_plan_payload=lambda payload: _sanitize_plan_payload(payload),
    knowledge_service_getter=get_knowledge_service,
    logger=logger,
)
execution_command_handler = ExecutionCommandHandler(
    send_event=connection_manager.send_event,
    get_execution_engine=lambda: get_execution_engine(),
    apply_plan_patches=lambda plan, patches: _apply_plan_patches(plan, patches),
    logger=logger,
)

# Unique ID generated at server startup; sent to frontend so it can detect restarts
SERVER_BOOT_ID = uuid4().hex[:12]


def _sanitize_plan_payload(plan_payload: dict) -> dict:
    nodes = plan_payload.get("nodes")
    edges = plan_payload.get("edges")
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    sanitized_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_type = node.get("node_type")
        if isinstance(node_type, dict):
            node_type = node_type.get("value")
        if isinstance(node_type, str):
            node_type = node_type.lower()

        position = node.get("position")
        if not isinstance(position, dict):
            position = {"x": 0, "y": 0}
        x = position.get("x", 0)
        y = position.get("y", 0)
        try:
            x = float(x)
        except Exception:
            x = 0.0
        try:
            y = float(y)
        except Exception:
            y = 0.0

        outputs = node.get("outputs")
        if not isinstance(outputs, dict):
            outputs = {}
        outputs = {str(k): v for k, v in outputs.items() if isinstance(v, str)}

        sanitized_nodes.append(
            {
                "node_id": node.get("node_id") or node.get("id") or "",
                "node_type": node_type or "llm",
                "position": {"x": x, "y": y},
                "config": node.get("config") if isinstance(node.get("config"), dict) else {},
                "inputs": node.get("inputs") if isinstance(node.get("inputs"), dict) else {},
                "outputs": outputs,
                "metadata": node.get("metadata") if isinstance(node.get("metadata"), dict) else {},
            }
        )

    layout_payload = plan_payload.get("layout")
    sanitized_layout: dict | None = None
    if isinstance(layout_payload, dict):
        positions = layout_payload.get("positions")
        if isinstance(positions, dict):
            sanitized_positions: dict[str, dict[str, float]] = {}
            for node_id, pos in positions.items():
                if not isinstance(node_id, str):
                    continue
                if not isinstance(pos, dict):
                    continue
                try:
                    x = float(pos.get("x", 0.0))
                except Exception:
                    x = 0.0
                try:
                    y = float(pos.get("y", 0.0))
                except Exception:
                    y = 0.0
                sanitized_positions[node_id] = {"x": x, "y": y}
            sanitized_layout = {"positions": sanitized_positions}

    sanitized_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        sanitized_edges.append(
            {
                "edge_id": edge.get("edge_id") or edge.get("id") or "",
                "source_node_id": edge.get("source_node_id") or edge.get("source"),
                "target_node_id": edge.get("target_node_id") or edge.get("target"),
                "metadata": edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {},
            }
        )

    entry_node_id = plan_payload.get("entry_node_id")
    if not entry_node_id and sanitized_nodes:
        entry_node_id = sanitized_nodes[0].get("node_id")

    sanitized = {
        "plan_id": plan_payload.get("plan_id") or "",
        "version": plan_payload.get("version") or 1,
        "nodes": sanitized_nodes,
        "edges": sanitized_edges,
        "entry_node_id": entry_node_id or "",
        "metadata": plan_payload.get("metadata")
        if isinstance(plan_payload.get("metadata"), dict)
        else {},
    }
    if sanitized_layout is not None:
        sanitized["layout"] = sanitized_layout
    return sanitized


def get_execution_engine() -> ExecutionEngine:
    global execution_engine
    if execution_engine is None:
        execution_engine = ExecutionEngine(connection_manager)
    return execution_engine


def _apply_plan_patches(current_plan: PlanIR, patches: list[PlanPatch]) -> bool:
    """Apply plan patches and return True if plan is modified."""

    plan_modified = False
    for patch in patches:
        action = patch.action
        logger.debug("Processing patch action: %s", action)
        if action == "add_node":
            node_data = patch.node or {}
            logger.debug("Adding node with data: %s", truncate_payload(node_data))
            from houyi.application.workflow.orchestration.plan import NodeType
            from houyi.interface.protocol.ir.plan_ir import NodeIR

            node_type_str = node_data.get("node_type", "llm")
            if isinstance(node_type_str, str):
                node_type_str = node_type_str.lower()
            new_node = NodeIR(
                node_id=node_data.get("node_id", ""),
                node_type=NodeType(node_type_str),
                position=node_data.get("position", {"x": 0, "y": 0}),
                config=node_data.get("config", {}),
                inputs=node_data.get("inputs", {}),
                outputs=node_data.get("outputs", {}),
                metadata=node_data.get("metadata", {}),
            )
            current_plan.nodes.append(new_node)
            if new_node.node_id:
                current_plan.set_node_position(new_node.node_id, new_node.position)
            plan_modified = True
            logger.info(
                "✓ Added node: %s (total nodes: %d)", new_node.node_id, len(current_plan.nodes)
            )

        elif action == "update_node":
            node_id = patch.node_id
            updates = patch.node or {}
            for node in current_plan.nodes:
                if node.node_id == node_id:
                    if "config" in updates:
                        node.config = updates["config"]
                    if "position" in updates:
                        current_plan.set_node_position(node_id, updates["position"])
                    if "metadata" in updates:
                        node.metadata = updates["metadata"]
                    if "inputs" in updates:
                        node.inputs = updates["inputs"]
                    if "outputs" in updates:
                        node.outputs = updates["outputs"]
                    plan_modified = True
                    logger.debug(
                        "Updated node: %s (update keys: %s)",
                        node_id,
                        sorted(updates.keys()),
                    )
                    break

        elif action == "delete_node":
            node_id = patch.node_id
            current_plan.nodes = [n for n in current_plan.nodes if n.node_id != node_id]
            current_plan.edges = [
                e
                for e in current_plan.edges
                if e.source_node_id != node_id and e.target_node_id != node_id
            ]
            plan_modified = True
            logger.debug("Deleted node: %s", node_id)

    return plan_modified


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_execution_engine()
    register_console_skills()

    # Initialize Chat subsystem
    chat_data_dir = os.getenv(ENV_CHAT_DATA_DIR, "data/conversations")
    settings_path = os.getenv(ENV_CHAT_SETTINGS_PATH, "data/settings.json")
    json_store = JsonStore(data_dir=chat_data_dir)
    settings_store = SettingsStore(settings_path=settings_path)
    chat_service = ChatService(
        json_store=json_store,
        default_model=os.getenv(ENV_DEEPSEEK_MODEL, DEFAULT_MODEL),
        default_system_instructions=os.getenv(ENV_CHAT_SYSTEM_PROMPT, ""),
        settings_store=settings_store,
    )
    chat_router = register_chat_routes(chat_service, settings_store=settings_store)
    app.include_router(chat_router)
    logger.info(
        "Chat subsystem initialized (data_dir=%s, settings=%s)", chat_data_dir, settings_path
    )

    logger.info("=" * 60)
    logger.info("HouYi Console Server Starting - %s", datetime.now())
    logger.info("Logging level: %s", LOG_LEVEL)
    logger.info("=" * 60)

    # Log active LLM provider — only show the configured provider at INFO level.
    # Other providers are logged at DEBUG to reduce startup noise.
    active_provider = os.getenv(ENV_DEFAULT_LLM_PROVIDER, "siliconflow")
    logger.info("LLM Provider: %s", active_provider)

    if active_provider == "siliconflow":
        siliconflow_key = os.getenv(ENV_SILICONFLOW_API_KEY, "")
        logger.info("  API Key: %s", "configured" if siliconflow_key else "NOT SET")
        logger.info(
            "  Base URL: %s",
            os.getenv(ENV_SILICONFLOW_BASE_URL, "https://api.siliconflow.cn/v1"),
        )
        logger.info("  Model: %s", os.getenv(ENV_DEEPSEEK_MODEL, "deepseek-chat"))
    elif active_provider in ("vertex", "google", "vertex_ai", "gemini"):
        google_project = os.getenv(ENV_GOOGLE_CLOUD_PROJECT, "")
        google_creds = os.getenv(ENV_GOOGLE_APPLICATION_CREDENTIALS, "")
        google_api_key = os.getenv(ENV_GOOGLE_API_KEY, "")
        if google_project:
            logger.info("  Auth: Vertex AI (project=%s)", google_project)
        elif google_creds:
            logger.info("  Auth: service account (%s) — project auto-detect pending", google_creds)
        elif google_api_key:
            logger.info("  Auth: API key (Gemini Developer API — public GA models only)")
        else:
            logger.warning("  Auth: NONE — set GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY")
        logger.info("  Model: %s", os.getenv(ENV_GEMINI_MODEL, "gemini-2.5-pro"))
    else:
        logger.info("  Provider details: check environment variables")

    # Auto-detect Google project from service account credentials
    if not os.environ.get(ENV_GOOGLE_CLOUD_PROJECT):
        creds_file = os.getenv(ENV_GOOGLE_APPLICATION_CREDENTIALS, "")
        if creds_file:
            try:
                import json

                with open(creds_file, encoding="utf-8") as f:
                    project = json.load(f).get("project_id", "")
                if project:
                    os.environ[ENV_GOOGLE_CLOUD_PROJECT] = project
                    logger.debug("Auto-detected Google project: %s", project)
            except Exception:
                pass

    configured_embedding_provider = os.getenv(ENV_EMBEDDING_PROVIDER, "")
    configured_embedding_model = os.getenv(ENV_EMBEDDING_MODEL, "")
    logger.info(
        "Embedding config requested: provider=%s model=%s",
        configured_embedding_provider or "(auto)",
        configured_embedding_model or "(default)",
    )
    try:
        embedding_cfg, detected_provider = resolve_embedding_config(
            strict_explicit=configured_embedding_provider == "local"
        )
    except RuntimeError as exc:
        logger.error("Embedding config validation failed: %s", exc)
        logger.error(
            "Embedding env snapshot: EMBEDDING_PROVIDER=%s EMBEDDING_MODEL=%s "
            "OPENAI_API_KEY=%s GOOGLE_API_KEY=%s GOOGLE_CLOUD_PROJECT=%s",
            configured_embedding_provider or "(unset)",
            configured_embedding_model or "(unset)",
            "configured" if os.getenv("OPENAI_API_KEY") else "unset",
            "configured" if os.getenv(ENV_GOOGLE_API_KEY) else "unset",
            os.getenv(ENV_GOOGLE_CLOUD_PROJECT) or "(unset)",
        )
        logger.error(
            "Explicit EMBEDDING_PROVIDER=local requires fastembed. Install dependencies with: "
            "uv sync --extra studio-server --extra rag-embedding"
        )
        logger.error(
            "Quick fallback (auto-detect): unset EMBEDDING_PROVIDER && unset EMBEDDING_MODEL"
        )
        raise

    if embedding_cfg is None:
        logger.warning(
            "No embedding provider detected at startup; ingest will run in degraded mode until a provider is available"
        )
    else:
        logger.info(
            "Embedding provider active: %s (model=%s, dim=%d)",
            detected_provider,
            embedding_cfg.model,
            embedding_cfg.dimension,
        )

    logger.info("=" * 60)
    yield


# Create FastAPI app
app = FastAPI(
    title="HouYi Console Server",
    description="WebSocket server for Agent execution visualization and control",
    version="0.3.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "houyi-console"}


@app.get("/api/sessions")
async def list_sessions() -> dict[str, list[str]]:
    """List active sessions."""
    return {"sessions": list(connection_manager.active_connections.keys())}


@app.get("/api/tools")
async def list_tools() -> dict:
    """List registered skills and their tools for the Title Bar pill.

    Returns both a flat skill list (for the dropdown) and aggregate
    counts so the Header can display "N skills (M tools)".
    """

    def _classify_skill_semantics(skill: object) -> tuple[str, bool, str, list[str]]:
        has_executor = callable(getattr(skill, "executor", None))
        instructions = str(getattr(skill, "instructions", "") or "")
        description = str(getattr(skill, "description", "") or "")
        allowed_tools = bool(getattr(skill, "allowed_tools", None))
        hooks = bool(getattr(skill, "hooks", None))

        semantic_text = "\n".join((description, instructions)).lower()
        has_flow_keywords = any(
            token in semantic_text for token in ("workflow", "flow", "steps", "process", "decision")
        )
        has_command_examples = bool(re.search(r"\b(python\s+\S+|\./\S+|sh\s+\S+)", instructions))
        has_script_runner = "scripts/run.py" in semantic_text

        signals: list[str] = []
        if has_executor:
            signals.append("bound_executor")
        if has_flow_keywords:
            signals.append("flow_keywords")
        if has_command_examples:
            signals.append("command_examples")
        if has_script_runner:
            signals.append("script_runner")
        if hooks:
            signals.append("hooks")
        if allowed_tools:
            signals.append("allowed_tools")

        if has_executor:
            return (
                "executable",
                has_executor,
                "Bound runtime executor is available",
                signals,
            )

        if has_command_examples and (has_flow_keywords or has_script_runner):
            return (
                "executable",
                has_executor,
                "SKILL.md defines runnable workflow command examples",
                signals,
            )

        if instructions.strip() or hooks or allowed_tools:
            return (
                "instruction",
                has_executor,
                "SKILL.md describes instruction-first orchestration behavior",
                signals,
            )

        return (
            "instruction",
            has_executor,
            "Metadata-only skill definition (no runnable command evidence)",
            signals,
        )

    skills = DEFAULT_SKILL_REGISTRY.list()
    items = []
    total_tools = 0
    for skill in skills:
        tool_count = len(skill.tools) if hasattr(skill, "tools") and skill.tools else 1
        total_tools += tool_count
        skill_type, has_executor, classification_basis, classification_signals = (
            _classify_skill_semantics(skill)
        )

        items.append(
            {
                "name": skill.name,
                "description": skill.description,
                "type": skill_type,
                "tool_count": tool_count,
                "has_executor": has_executor,
                "classification_basis": classification_basis,
                "classification_signals": classification_signals,
            }
        )
    return {
        "tools": items,
        "skill_count": len(items),
        "tool_count": total_tools,
    }


## Heartbeat / session constants
# How often the server sends a ping to the client (seconds).
# 30s is the industry standard, well within typical proxy idle timeouts (60-120s).
_HEARTBEAT_INTERVAL_S = 30
# If no pong is received within this window, the client is considered dead.
# 3 × heartbeat interval = 90s, tolerates up to 2 missed pings.
# The client sends a pong on visibilitychange (tab foreground), so background
# tabs won't cause false positives within this window.
_CLIENT_TIMEOUT_S = 90
# Grace period after disconnect before aborting executions.
# ReconnectingWebSocket reconnects within 500ms-2s on first attempt;
# 15s covers several retry cycles with exponential back-off.
_DISCONNECT_GRACE_S = 15


@app.get("/api/knowledge/embedding-status")
async def get_embedding_status() -> JSONResponse:
    """Check which embedding provider is available.

    Returns the detected provider or instructions for installing one.
    The frontend uses this to show proactive guidance in the Knowledge panel.
    """
    from ..rag import _detect_embedding_config

    config, provider_name = _detect_embedding_config()
    if config is not None:
        return JSONResponse(
            content={
                "available": True,
                "provider": provider_name,
                "model": config.model,
                "dimension": config.dimension,
            }
        )
    return JSONResponse(
        content={
            "available": False,
            "provider": None,
            "reason": provider_name,
            "instructions": (
                "No embedding provider detected. To enable semantic search, "
                "choose one of:\n"
                "1. Set OPENAI_API_KEY environment variable (OpenAI embeddings)\n"
                "2. pip install fastembed (local embeddings, no API key needed)\n"
                "3. Set GOOGLE_CLOUD_PROJECT + install google-genai (Gemini embeddings)"
            ),
        }
    )


@app.post("/api/knowledge/{library_id}/upload")
async def upload_knowledge_files(
    library_id: str,
    files: list[UploadFile] = File(...),  # noqa: B008
) -> JSONResponse:
    """Upload files to a knowledge library.

    Files are saved to the library's dedicated storage directory:
    {STORAGE}/{library_id}/uploads/

    Args:
        library_id: Knowledge library identifier
        files: List of files to upload

    Returns:
        JSON with uploaded file paths
    """
    import aiofiles

    knowledge_service = get_knowledge_service()
    library = knowledge_service.get_library(library_id)

    if not library:
        return JSONResponse(
            status_code=404,
            content={"error": f"Library {library_id} not found"},
        )

    # Use library-specific upload directory (via service instance for isolation)
    upload_dir = knowledge_service.library_upload_dir(library_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    errors: list[str] = []

    for file in files:
        if not file.filename:
            continue

        # Sanitize filename
        safe_filename = Path(file.filename).name
        file_path = upload_dir / safe_filename

        try:
            # Read and save file content
            content = await file.read()
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)
            saved_paths.append(str(file_path.resolve()))  # Use absolute path
            logger.info("Uploaded file: %s -> %s", file.filename, file_path.resolve())
        except Exception as e:
            errors.append(f"{file.filename}: {e!s}")
            logger.error("Failed to upload %s: %s", file.filename, e)

    return JSONResponse(
        content={
            "library_id": library_id,
            "uploaded_paths": saved_paths,
            "errors": errors,
            "upload_dir": str(upload_dir.resolve()),  # Use absolute path
        }
    )


@app.websocket("/ws/session/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for console sessions.

    Session lifecycle within a single page load:
      1. Client opens WS with a fresh session_id (generated per page load).
      2. Server sends existing plan + log level + buffered spans (for reconnect).
      3. Server-driven heartbeat keeps the connection alive:
         - Server → Client: ping every 30s
         - Client → Server: pong reply (automatic on ping, plus on tab foreground)
         - Server tracks last pong; closes if 90s exceeded (3 missed pings).
      4. On disconnect, server waits 15s for reconnect before aborting executions.
    """
    engine = get_execution_engine()
    logger.debug("WebSocket connection attempt: session=%s", session_id)
    await connection_manager.connect(websocket, session_id)
    logger.debug("WebSocket connected: session=%s", session_id)

    # --- Restore session state for reconnecting clients ---
    # Wrap in try/except: the client may disconnect during this phase (e.g. during
    # hot-reload the frontend opens multiple connections rapidly and closes stale ones
    # before the server finishes sending initial state).
    try:
        # Send existing plan if available (same session reconnect)
        existing_plan = engine.plan_service.get_current_plan(session_id)
        if existing_plan:
            logger.info(
                "Restoring plan for session=%s: %d nodes, %d edges",
                session_id,
                len(existing_plan.nodes),
                len(existing_plan.edges),
            )
            from .events import PlanUpdatedEvent

            plan_event = PlanUpdatedEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                plan=existing_plan,
            )
            await connection_manager.send_event(session_id, plan_event)

        # Send server info (informational, no session logic depends on it)
        await websocket.send_json({"event_type": "server_info", "server_boot_id": SERVER_BOOT_ID})

        log_level_event = LogLevelEvent(
            event_id=f"evt_{uuid4().hex[:8]}",
            session_id=session_id,
            level=get_log_level().lower(),
        )
        await connection_manager.send_event(session_id, log_level_event)

        # Replay buffered span events so reconnecting clients recover timeline data
        buffered_spans = engine.observation_service.get_buffered_spans(session_id)
        if buffered_spans:
            logger.info(
                "Replaying %d buffered span events for session=%s",
                len(buffered_spans),
                session_id,
            )
            for span_event in buffered_spans:
                await connection_manager.send_event(session_id, span_event)
    except (RuntimeError, WebSocketDisconnect):
        # Client already gone — clean up and exit (heartbeat not started yet)
        logger.debug("Client disconnected during session restore: session=%s", session_id)
        connection_manager.disconnect(websocket)
        return

    # --- Bidirectional heartbeat ---

    last_pong = _time.monotonic()

    async def _heartbeat() -> None:
        """Server→client ping + client liveness check."""
        nonlocal last_pong
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            # Check client liveness
            elapsed = _time.monotonic() - last_pong
            if elapsed > _CLIENT_TIMEOUT_S:
                logger.warning(
                    "Client timeout (no pong for %.0fs): session=%s",
                    elapsed,
                    session_id,
                )
                # Close the connection; the finally block handles cleanup
                with suppress(Exception):
                    await websocket.close(code=4000, reason="client_timeout")
                break
            try:
                await websocket.send_json({"event_type": "ping"})
            except Exception:
                break

    heartbeat_task = asyncio.create_task(_heartbeat())

    # --- Command receive loop ---
    try:
        while True:
            command_data = await connection_manager.receive_command(websocket)

            if command_data is None:
                break

            # Handle pong from client (heartbeat reply, not a real command)
            if isinstance(command_data, dict) and command_data.get("command_type") == "pong":
                last_pong = _time.monotonic()
                continue

            logger.debug("Received command: %s", truncate_payload(command_data))

            try:
                command = parse_command(command_data)
                if command:
                    await handle_command(command, session_id)
                else:
                    logger.warning("Failed to parse command: %s", command_data)
            except Exception as e:
                logger.error("Error handling command: %s", e, exc_info=True)

    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected: session=%s", session_id)

    finally:
        heartbeat_task.cancel()
        connection_manager.disconnect(websocket)
        if connection_manager.get_session_count(session_id) == 0:
            # Check if this session has any running executions worth waiting for
            has_running = any(
                e.status == ExecutionStatus.RUNNING and e.metadata.get("session_id") == session_id
                for e in engine.execution_store.executions.values()
            )
            if has_running:
                # Grace period: allow ReconnectingWebSocket to reconnect
                logger.info(
                    "Session disconnected with running executions, waiting %ds for reconnect: session=%s",
                    _DISCONNECT_GRACE_S,
                    session_id,
                )
                await asyncio.sleep(_DISCONNECT_GRACE_S)
                if connection_manager.get_session_count(session_id) == 0:
                    logger.info("No reconnection, aborting executions: session=%s", session_id)
                    await execution_engine.abort_session_executions(session_id)
                else:
                    logger.info("Client reconnected during grace period: session=%s", session_id)
            else:
                # No running executions — clean up immediately, no grace period needed
                logger.debug("Session disconnected (no running executions): session=%s", session_id)
            engine.observation_service.clear_session_buffer(session_id)


def parse_command(data: dict) -> ClientCommand | dict | None:
    """Parse command data into appropriate command object.

    Args:
        data: Raw command data from client

    Returns:
        Parsed command object or None if parsing fails
    """
    parsed = command_parser.parse(data)
    if parsed is None:
        return None
    return parsed


async def _dispatch_skill_command(command: ClientCommand | dict, session_id: str) -> None:
    if skill_command_handler.can_handle(command):
        await skill_command_handler.handle(command, session_id)


async def _dispatch_command(command: ClientCommand | dict, session_id: str) -> None:
    if command_handler.can_handle(command):
        await command_handler.handle(command, session_id)


for _skill_command_type in (
    "list_skills",
    "get_skill_detail",
    "get_skill_metrics",
    "load_skill",
    "unload_skill",
    "remove_skill_from_disk",
    "configure_skill",
    "dry_run_skill",
    "consent_response",
):
    command_dispatcher.register(_skill_command_type, _dispatch_skill_command)

for _command_type in (
    "save_workflow",
    "load_workflow",
    "list_workflows",
    "list_knowledge_libraries",
    "create_knowledge_library",
    "delete_knowledge_library",
    "update_knowledge_library",
    "search_knowledge",
    "ingest_knowledge_files",
    "rebuild_knowledge_index",
    "cancel_ingest",
    "list_documents",
    "get_document",
    "delete_document",
    "disable_document",
    "enable_document",
    "list_chunks",
    "preview_chunks",
    "frontend_log",
):
    command_dispatcher.register(_command_type, _dispatch_command)


async def handle_command(command: ClientCommand | dict, session_id: str) -> None:
    """Handle a client command.

    Args:
        command: Parsed command (ClientCommand object or dict for workflow commands)
        session_id: Session identifier
    """
    # Ensure execution engine is initialized for command handling.
    get_execution_engine()
    # Get command type safely for both dict and object types
    if isinstance(command, dict):
        command_type = command.get("command_type")
        command_id = command.get("command_id")
    else:
        command_type = command.command_type
        command_id = command.command_id

    logger.debug(
        "Handling command: type=%s session=%s command_id=%s",
        command_type,
        session_id,
        command_id,
    )

    try:
        if await command_dispatcher.dispatch(command, session_id):
            return

        # Delegate typed execution-lifecycle commands
        if execution_command_handler.can_handle(command):
            await execution_command_handler.handle(command, session_id)

    except Exception as e:
        logger.error("Error handling command: %s", e, exc_info=True)


def main() -> None:
    """Launch the uvicorn dev server with hot-reload."""
    import uvicorn

    log_config = build_logging_config(LOG_LEVEL)

    server_dir = Path(__file__).resolve().parent.parent.parent.parent  # houyi-studio/server/
    project_root = server_dir.parent.parent  # project root
    reload_dirs = [
        str(server_dir / "houyi_studio"),
        str(project_root / "houyi"),
    ]

    uvicorn.run(
        "houyi_studio.server.gateway.app:app",
        host="0.0.0.0",
        port=int(os.environ.get(ENV_HOUYI_PORT, "8000")),
        reload=True,
        reload_dirs=reload_dirs,
        log_level=LOG_LEVEL.lower(),
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
