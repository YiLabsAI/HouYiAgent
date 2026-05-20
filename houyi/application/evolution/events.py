from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvolutionEventType(str, Enum):
    MEMORY_WRITE = "memory_write"
    RECALL_RESULT = "recall_result"
    RECALL_FAILURE = "recall_failure"
    USER_CORRECTION = "user_correction"
    IDK_DECISION = "idk_decision"
    EXTRACTOR_LOW_CERTAINTY = "extractor_low_certainty"
    RETRACTION_FIRED = "retraction_fired"
    BENCHMARK_RESULT = "benchmark_result"
    TOOL_TRACE = "tool_trace"
    AGENT_TRAJECTORY = "agent_trajectory"
    ROLLBACK_PERFORMED = "rollback_performed"


@dataclass(frozen=True, slots=True)
class EvolutionEvent:
    event_type: EvolutionEventType
    target: str
    payload: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    namespace: str = "default"
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")


@dataclass(frozen=True, slots=True)
class EvolutionSignal:
    signal_type: str
    target: str
    severity: float
    event_ids: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
