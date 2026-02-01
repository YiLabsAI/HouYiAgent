"""Context bundle structures for execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContextBundle:
    system_context: dict[str, Any] = field(default_factory=dict)
    session_context: dict[str, Any] = field(default_factory=dict)
    run_settings: dict[str, Any] = field(default_factory=dict)
    memory_items: list[Any] = field(default_factory=list)
    rag_chunks: list[Any] = field(default_factory=list)
    tool_state: dict[str, Any] = field(default_factory=dict)
    conflicts: list[Any] = field(default_factory=list)
