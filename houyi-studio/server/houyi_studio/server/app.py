"""FastAPI application for console WebSocket server."""

from __future__ import annotations

import asyncio
import logging
import os
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY
from houyi.protocol.ir import ExecutionStatus, PlanIR

from .commands import (
    AbortCommand,
    ClientCommand,
    CommandType,
    PatchPlanCommand,
    PauseCommand,
    PlanPatch,
    RestoreCheckpointCommand,
    ResumeCommand,
    RetryNodeCommand,
    SetLogLevelCommand,
    StartExecutionCommand,
)
from .events import ExecutionStatusEvent, LogLevelEvent
from .execution_engine import ExecutionEngine
from .logging_config import (
    build_logging_config,
    configure_logging,
    get_log_level,
    set_log_level,
    truncate_payload,
)
from .startup_hooks import register_console_skills
from .websocket import connection_manager

# Load environment variables from .env file
load_dotenv()

LOG_LEVEL = configure_logging()
logger = logging.getLogger(__name__)

execution_engine: ExecutionEngine | None = None

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
            from houyi.orchestration.plan import NodeType
            from houyi.protocol.ir.plan_ir import NodeIR

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
    logger.info("=" * 60)
    logger.info("HouYi Console Server Starting - %s", datetime.now())
    logger.info("Logging level: %s", LOG_LEVEL)
    logger.info("=" * 60)

    # Log environment configuration
    logger.info("Environment Configuration:")
    logger.info("  LLM Provider: %s", os.getenv("DEFAULT_LLM_PROVIDER", "siliconflow"))
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY", "")
    logger.info("  SiliconFlow API Key: %s", "✓ Configured" if siliconflow_key else "✗ Not Set")
    logger.info(
        "  SiliconFlow Base URL: %s",
        os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    )
    logger.info("  DeepSeek Model: %s", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    google_project = os.getenv("GOOGLE_PROJECT_ID", "")
    logger.info("  Google Project ID: %s", "✓ Configured" if google_project else "✗ Not Set")
    logger.info("  Gemini Model: %s", os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
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
async def list_tools() -> dict[str, list[dict[str, str]]]:
    """List registered tools for console debugging."""
    return {
        "tools": [
            {"name": skill.name, "description": skill.description}
            for skill in DEFAULT_SKILL_REGISTRY.list()
        ]
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
    logger.info("WebSocket connection attempt: session=%s", session_id)
    await connection_manager.connect(websocket, session_id)
    logger.info("WebSocket connected: session=%s", session_id)

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
        logger.info("Client disconnected during session restore: session=%s", session_id)
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
                try:
                    await websocket.close(code=4000, reason="client_timeout")
                except Exception:
                    pass
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
        logger.info("WebSocket disconnected: session=%s", session_id)

    finally:
        heartbeat_task.cancel()
        connection_manager.disconnect(websocket)
        if connection_manager.get_session_count(session_id) == 0:
            # Grace period: allow ReconnectingWebSocket to reconnect with same session ID
            logger.info(
                "No active connections for session=%s, waiting %ds for reconnect",
                session_id,
                _DISCONNECT_GRACE_S,
            )
            await asyncio.sleep(_DISCONNECT_GRACE_S)
            if connection_manager.get_session_count(session_id) == 0:
                logger.info(
                    "No reconnection after grace period, aborting executions: session=%s",
                    session_id,
                )
                await execution_engine.abort_session_executions(session_id)
                engine.observation_service.clear_session_buffer(session_id)
            else:
                logger.info("Client reconnected during grace period: session=%s", session_id)


def parse_command(data: dict) -> ClientCommand | None:
    """Parse command data into appropriate command object.

    Args:
        data: Raw command data from client

    Returns:
        Parsed command object or None if parsing fails
    """
    try:
        command_type = data.get("command_type")
        logger.debug("Parsing command type: %s", command_type)

        if command_type == CommandType.START_EXECUTION:
            return StartExecutionCommand(**data)
        elif command_type == CommandType.PAUSE:
            return PauseCommand(**data)
        elif command_type == CommandType.RESUME:
            return ResumeCommand(**data)
        elif command_type == CommandType.ABORT:
            return AbortCommand(**data)
        elif command_type == CommandType.RETRY_NODE:
            return RetryNodeCommand(**data)
        elif command_type == CommandType.PATCH_PLAN:
            return PatchPlanCommand(**data)
        elif command_type == CommandType.RESTORE_CHECKPOINT:
            return RestoreCheckpointCommand(**data)
        elif command_type == CommandType.SET_LOG_LEVEL:
            return SetLogLevelCommand(**data)
        elif command_type == "save_workflow":
            # Simple dict-based command for workflow operations
            return data
        elif command_type == "load_workflow":
            return data
        elif command_type == "list_workflows":
            return data
        else:
            logger.warning("Unknown command type: %s", command_type)
            return None
    except Exception as e:
        logger.error("Error parsing command: %s", e)
        return None


async def handle_command(command: ClientCommand | dict, session_id: str) -> None:
    """Handle a client command.

    Args:
        command: Parsed command (ClientCommand object or dict for workflow commands)
        session_id: Session identifier
    """
    # Ensure execution engine is initialized for command handling.
    engine = get_execution_engine()
    # Get command type safely for both dict and object types
    if isinstance(command, dict):
        command_type = command.get("command_type")
        command_id = command.get("command_id")
    else:
        command_type = command.command_type
        command_id = command.command_id

    logger.info(
        "Handling command: type=%s session=%s command_id=%s",
        command_type,
        session_id,
        command_id,
    )

    try:
        # Handle workflow commands first (dict-based)
        if isinstance(command, dict) and command.get("command_type") == "save_workflow":
            workflow_name = command.get("workflow_name")
            logger.info("=== Save workflow command received ===")
            logger.info("Workflow name: %s", workflow_name)
            logger.info("Session ID: %s", session_id)

            # Prefer plan payload from client if provided
            plan_payload = command.get("plan") if isinstance(command, dict) else None
            current_plan = None
            if plan_payload:
                try:
                    sanitized_payload = _sanitize_plan_payload(plan_payload)
                    current_plan = PlanIR.model_validate(sanitized_payload)
                    logger.info("Using plan payload from client for workflow save")
                except Exception:
                    logger.warning(
                        "Failed to parse plan payload for workflow save; falling back to session plan",
                        exc_info=True,
                    )
            if current_plan is None:
                current_plan = engine.plan_service.get_current_plan(session_id)
            logger.info("Current plan exists: %s", current_plan is not None)

            if current_plan:
                logger.info(
                    "Plan has %d nodes, %d edges", len(current_plan.nodes), len(current_plan.edges)
                )
                success = engine.workflow_service.save_workflow(workflow_name, current_plan)
                if success:
                    logger.info("✓ Workflow '%s' saved successfully", workflow_name)
                else:
                    logger.error("✗ Failed to save workflow '%s'", workflow_name)
            else:
                logger.warning("✗ No plan to save for session: %s", session_id)
                logger.warning("Available sessions: %s", list(engine.plans.keys()))

        elif isinstance(command, dict) and command.get("command_type") == "load_workflow":
            workflow_name = command.get("workflow_name")
            logger.info("=== Load workflow command received ===")
            logger.info("Workflow name: %s", workflow_name)
            logger.info("Session ID: %s", session_id)

            # Load workflow
            plan = engine.workflow_service.load_workflow(workflow_name)
            if plan:
                # Store plan for this session
                engine.plan_service.set_current_plan(session_id, plan, persist=True)

                # Send plan to frontend
                from .events import PlanUpdatedEvent

                plan_event = PlanUpdatedEvent(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    plan=plan,
                )
                await connection_manager.send_event(session_id, plan_event)
                logger.info("✓ Workflow '%s' loaded and sent to frontend", workflow_name)
            else:
                logger.error("✗ Failed to load workflow '%s'", workflow_name)

        elif isinstance(command, dict) and command.get("command_type") == "list_workflows":
            logger.info("=== List workflows command received ===")
            logger.info("Session ID: %s", session_id)

            # Get workflow list
            workflows = engine.workflow_service.list_workflows()
            logger.info("Found %d workflows", len(workflows))
            from .events import WorkflowListEvent

            workflow_event = WorkflowListEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                workflows=workflows,
            )
            await connection_manager.send_event(session_id, workflow_event)
            logger.info("Sent workflow list to frontend")

        # Handle standard ClientCommand objects
        if isinstance(command, StartExecutionCommand):
            # Get plan from command inputs or use stored plan
            plan_data = command.inputs.get("plan") if command.inputs else None
            run_settings = command.inputs.get("run_settings") if command.inputs else None

            if plan_data:
                # Convert frontend plan data to PlanIR
                from houyi.orchestration.plan import NodeType
                from houyi.protocol.ir.plan_ir import EdgeIR, NodeIR, PlanIR

                # Build nodes list
                nodes = []
                for node_data in plan_data.get("nodes", []):
                    # Frontend sends node_id and node_type, not id and type
                    node = NodeIR(
                        node_id=node_data.get("node_id", node_data.get("id", "")),
                        node_type=NodeType(
                            node_data.get("node_type", node_data.get("type", "llm"))
                        ),
                        position=node_data.get("position", {"x": 0, "y": 0}),
                        config=node_data.get("config", {}),
                        inputs=node_data.get("inputs", {}),
                        outputs=node_data.get("outputs", {}),
                        metadata=node_data.get("metadata", {}),
                    )
                    nodes.append(node)

                # Build edges list
                edges = []
                for edge_data in plan_data.get("edges", []):
                    edge = EdgeIR(
                        edge_id=edge_data["id"],
                        source_node_id=edge_data["source"],
                        target_node_id=edge_data["target"],
                        metadata=edge_data.get("metadata", {}),
                    )
                    edges.append(edge)

                # Create PlanIR
                plan = PlanIR(
                    plan_id=command.plan_id,
                    nodes=nodes,
                    edges=edges,
                    entry_node_id=nodes[0].node_id if nodes else None,
                    metadata={"source": "console_ui"},
                    version=0,
                )
                logger.info(
                    "Converted frontend plan to PlanIR: %d nodes, %d edges", len(nodes), len(edges)
                )
            else:
                # No plan data provided
                logger.error("No plan data provided in start_execution command")
                return

            validation_errors = []
            for node in plan.nodes:
                if node.node_type != NodeType.TOOL:
                    continue
                tool_name = None
                if isinstance(node.config, dict):
                    tool_name = node.config.get("tool_name")
                if not tool_name and isinstance(node.metadata, dict):
                    tool_name = node.metadata.get("tool_name") or node.metadata.get("skill_name")
                if not tool_name:
                    validation_errors.append(f"tool node {node.node_id} missing tool_name")

            if validation_errors:
                logger.debug("Start execution validation failed: %s", "; ".join(validation_errors))
                validation_execution_id = f"exec_validation_{uuid4().hex[:8]}"
                event = ExecutionStatusEvent(
                    event_id=f"evt_{uuid4().hex[:8]}",
                    session_id=session_id,
                    execution_id=validation_execution_id,
                    status=ExecutionStatus.FAILED,
                    message=f"Execution failed: {'; '.join(validation_errors)}",
                )
                await connection_manager.send_event(session_id, event)
                return

            logger.info("Starting execution with plan: %s", plan.plan_id)
            await execution_engine.start_execution(session_id, plan, run_settings=run_settings)
            logger.info("Execution started successfully")

        elif isinstance(command, PauseCommand):
            await execution_engine.pause_execution(command.execution_id)

        elif isinstance(command, ResumeCommand):
            await execution_engine.resume_execution(command.execution_id)

        elif isinstance(command, AbortCommand):
            await execution_engine.abort_execution(command.execution_id)

        elif isinstance(command, RetryNodeCommand):
            logger.info(
                "Retry node: execution_id=%s, node_id=%s",
                command.execution_id,
                command.node_id,
            )
            await execution_engine.retry_node(
                session_id=session_id,
                execution_id=command.execution_id,
                node_id=command.node_id,
                new_inputs=command.new_inputs,
            )

        elif isinstance(command, PatchPlanCommand):
            logger.info(
                "Patch plan: execution_id=%s, base_version=%s, patches=%d",
                command.execution_id,
                command.base_version,
                len(command.patches),
            )
            # Apply patches to current plan
            current_plan = execution_engine.plan_service.get_current_plan(session_id)

            # Create empty plan if not exists (for new sessions)
            if not current_plan:
                from houyi.protocol.ir.plan_ir import PlanIR

                current_plan = PlanIR(
                    plan_id=f"plan_{int(datetime.now().timestamp() * 1000)}",
                    version=0,
                    nodes=[],
                    edges=[],
                    entry_node_id="",  # Empty string for plans without entry node yet
                    metadata={"source": "console_ui"},
                )
                execution_engine.plan_service.set_current_plan(
                    session_id, current_plan, persist=False
                )
                logger.info("Created new empty plan for session: %s", session_id)

            if current_plan:
                logger.info("Processing %d patches", len(command.patches))
                plan_modified = _apply_plan_patches(current_plan, command.patches)

                # Increment plan version if modified
                if plan_modified:
                    current_plan.version += 1
                    logger.info("Plan patched successfully")
                    # Save plan to file for persistence
                    execution_engine.plan_service.save_plan_to_file(session_id, current_plan)

                    # Send updated plan to frontend only if modified
                    from .events import PlanUpdatedEvent

                    plan_event = PlanUpdatedEvent(
                        event_id=f"evt_{uuid4().hex[:8]}",
                        session_id=session_id,
                        plan=current_plan,
                    )
                    await connection_manager.send_event(session_id, plan_event)
                    logger.info("Plan updated and sent to frontend")

        elif isinstance(command, RestoreCheckpointCommand):
            logger.info(
                "Restore checkpoint: execution_id=%s, checkpoint_id=%s, mode=%s",
                command.execution_id,
                command.checkpoint_id,
                command.replay_mode,
            )
            await execution_engine.restore_checkpoint(
                session_id=session_id,
                checkpoint_id=command.checkpoint_id,
                replay_mode=getattr(command.replay_mode, "value", command.replay_mode),
                execution_id=command.execution_id,
            )

        elif isinstance(command, SetLogLevelCommand):
            resolved_level = set_log_level(command.level)
            logger.info(
                "Log level updated: requested=%s resolved=%s effective=%s",
                command.level,
                resolved_level,
                get_log_level(),
            )
            log_level_event = LogLevelEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                session_id=session_id,
                level=resolved_level.lower(),
                requested_level=command.level,
            )
            await connection_manager.send_event(session_id, log_level_event)

    except Exception as e:
        logger.error("Error handling command: %s", e, exc_info=True)


if __name__ == "__main__":
    import uvicorn

    log_config = build_logging_config(LOG_LEVEL)

    uvicorn.run(
        "houyi_studio.server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("HOUYI_PORT", "8000")),
        reload=True,
        log_level=LOG_LEVEL.lower(),
        log_config=log_config,
    )
