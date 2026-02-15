"""Central command dispatcher — the Open/Closed backbone of the handler hierarchy.

This module implements the **Strategy + Registry** pattern that decouples
WebSocket command routing from command handling:

::

    app.py (startup)
        │
        │  register("list_skills", skill_handler.handle)
        │  register("save_workflow", command_handler.handle)
        │  ...
        ▼
    CommandDispatcher          ◄── this module
        │
        │  dispatch(command, session_id) → look up command_type → call handler
        ▼
    SkillCommandHandler | CommandHandler | ExecutionCommandHandler

**OCP in action**: Adding a new command requires only *registering* a new
``(command_type, handler)`` pair — no ``if/elif`` chains, no modifications to
existing handlers or dispatcher logic.

**Liskov substitution**: All handlers share the same ``async (command, session_id) -> None``
signature (the ``CommandHandler`` callable alias), so the dispatcher treats
them uniformly regardless of their concrete type.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .commands import ClientCommand

CommandLike = ClientCommand | dict[str, Any]
CommandHandler = Callable[[CommandLike, str], Awaitable[None]]


def get_command_type_and_id(command: CommandLike) -> tuple[str | None, str | None]:
    """Read command_type and command_id from dict or typed command."""
    if isinstance(command, dict):
        command_type = command.get("command_type")
        command_id = command.get("command_id")
        return (
            command_type if isinstance(command_type, str) else None,
            command_id if isinstance(command_id, str) else None,
        )

    command_type = command.command_type
    if hasattr(command_type, "value"):
        command_type = command_type.value
    return (
        command_type if isinstance(command_type, str) else None,
        command.command_id,
    )


class CommandDispatcher:
    """Registry-based command router following the Open/Closed Principle.

    Handlers are registered at startup via ``register(command_type, handler)``.
    At runtime, ``dispatch()`` performs an O(1) lookup by ``command_type`` and
    delegates to the matching handler, returning ``True`` if a handler was
    found or ``False`` to let the caller try alternative dispatch paths
    (e.g., isinstance-based routing for typed Pydantic commands).
    """

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}

    def register(self, command_type: str, handler: CommandHandler) -> None:
        self._handlers[command_type] = handler

    async def dispatch(self, command: CommandLike, session_id: str) -> bool:
        command_type, _ = get_command_type_and_id(command)
        if not command_type:
            return False
        handler = self._handlers.get(command_type)
        if handler is None:
            return False
        await handler(command, session_id)
        return True
