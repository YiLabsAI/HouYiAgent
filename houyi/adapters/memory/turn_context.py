"""Cross-turn context recovery for the memory answerer.

Atomic extraction attributes each fact to a single turn, so a fact whose
meaning depends on a prior turn's context (e.g. a destination reached on
a trip mentioned earlier) loses that qualifier. This module recovers the
surrounding dialogue so the answerer sees the conversation context.

Neighbor selection is format-agnostic across bench and production:
- dia_id priority: when the turn_id encodes a dialogue position
  (D{session}:N), gather same-prefix turns whose N is within +/-window.
  This is conversation-true regardless of session_id granularity or
  ingestion order, because the dia_id IS the conversation position.
- turn_index fallback: for opaque turn_ids (e.g. uuid), fall back to
  per-(namespace, session_id) turn_index +/-window. This is
  conversation-true ONLY when the caller ingests with session-level
  session_id in conversation order -- a deployment prerequisite, not a
  code guarantee.
"""

from __future__ import annotations

import re
from typing import Any


def fetch_turn_context(backend: Any, turn_id: str | None, window: int = 3) -> str:
    """Return conversation-adjacent turn text for a fact's source turn."""
    if not turn_id or not backend:
        return ""
    raw_log = getattr(backend, "_raw_turn_log", None)
    if raw_log is None:
        return ""
    turn = raw_log.get_raw_turn(turn_id)
    if turn is None:
        return ""
    try:
        turns = raw_log.list_raw_turns(turn.namespace, turn.session_id, limit=500)
    except Exception:
        return ""
    ctx = _dia_id_neighbours(turn_id, turns, window)
    if ctx:
        return ctx
    return _turn_index_neighbours(turn_id, turns, window)


def _dia_id_neighbours(turn_id: str, turns: list[Any], window: int) -> str:
    """Gather same-prefix turns whose dia_id N is within +/-window."""
    m = re.search(r"^(.+?D\d+):(\d+)(?::|$)", turn_id)
    if not m:
        return ""
    prefix, n = m.group(1), int(m.group(2))
    adj: list[tuple[int, Any]] = []
    for t in turns:
        tm = re.search(r"^(.+?D\d+):(\d+)(?::|$)", t.turn_id)
        if tm and tm.group(1) == prefix:
            tn = int(tm.group(2))
            if n - window <= tn <= n + window:
                adj.append((tn, t))
    if not adj:
        return ""
    adj.sort(key=lambda x: x[0])
    return " | ".join(f"{t.role}: {t.content[:140]}" for _, t in adj)


def _turn_index_neighbours(turn_id: str, turns: list[Any], window: int) -> str:
    """Fallback: per-(namespace, session_id) turn_index +/-window."""
    idx = next((i for i, t in enumerate(turns) if t.turn_id == turn_id), None)
    if idx is None:
        return ""
    lo, hi = max(0, idx - window), min(len(turns), idx + window + 1)
    return " | ".join(f"{t.role}: {t.content[:140]}" for t in turns[lo:hi])


def reformat_recall_content(
    content: str,
    quals: dict[str, str] | None,
    internal_keys: frozenset[str],
) -> str:
    """Render a memory record's content with time qualifiers for the answerer."""
    m = re.search(r"\((?:time|date):\s*([^)]+)\)", content)
    cal_time = m.group(1).strip() if m else None
    if not quals and not cal_time:
        return content

    approximate = bool(
        quals and str(quals.get("date_certainty", "")).strip().lower() == "approximate"
    )
    parts = []
    if cal_time:
        if approximate:
            parts.append(f"reported on {cal_time}, exact date earlier/uncertain")
        else:
            parts.append(f"time: {cal_time}")
    for qk, qv in sorted((quals or {}).items()):
        if not qv or (qk == "date" and cal_time) or qk == "date_certainty":
            continue
        if qk in internal_keys:
            continue
        if qk == "date" and approximate:
            parts.append(f"reported on {qv}, exact date earlier/uncertain")
            continue
        lbl = "time" if qk == "date" else qk
        parts.append(f"{lbl}: {qv}")

    if parts:
        content_stripped = re.sub(r"\s*\([^)]*\)\s*$", "", content).strip()
        return f"{content_stripped} ({', '.join(parts)})"
    return content
