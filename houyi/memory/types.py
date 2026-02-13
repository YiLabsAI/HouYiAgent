"""Memory Engine data types.

Phase 1: Basic MemoryRecord and MemoryScope.
Phase 3: MemoryCandidate, scoring metadata, multi-scope retrieval.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryScope(str, Enum):
    """Scope of a memory record."""

    SESSION = "session"
    USER = "user"
    WORKSPACE = "workspace"


class MemoryRecord(BaseModel):
    """A single memory entry.

    Phase 1: Simple key-value with scope and timestamps.
    Phase 3: Adds embedding, relevance score, source trace.
    """

    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    scope: MemoryScope = MemoryScope.SESSION
    key: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    ttl: float | None = None

    @property
    def is_expired(self) -> bool:
        """Check if this record has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl
