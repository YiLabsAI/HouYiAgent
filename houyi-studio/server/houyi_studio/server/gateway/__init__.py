"""API gateway subsystem.

Public API:
    - app: FastAPI application instance
    - connection_manager: WebSocket connection manager singleton
    - CommandDispatcher: Routes commands to appropriate handlers
    - CommandParser: Parses raw WebSocket JSON into typed commands
    - CommandHandler: Resource CRUD command handler (workflow/knowledge/document)
    - ServerEvent: Base class for all server-sent events
    - ClientCommand: Base class for all client commands

Internal:
    - event_bus: In-process pub/sub (used by execution layer)
    - Individual event/command Pydantic models (import directly from .events / .commands)
"""
