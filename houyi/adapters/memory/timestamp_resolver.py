"""Deterministic relative-time resolution for event timestamps.

The extractor delegates relative-to-absolute time conversion to the LLM,
which is non-deterministic: the same source turn sometimes yields a
resolved year and sometimes the verbatim relative phrase. The prompt also
contains a contradiction (one rule says preserve the phrase verbatim, the
mapping table says resolve it), so the LLM picks either path at random.

This module removes that non-determinism by resolving relative time
expressions in code, anchored on the per-turn observation_date that is
deterministically available at extraction time. Storing the resolved
absolute value means the event carries its own time context and no longer
depends on the LLM mood or on a query-time anchor (which would be the
wrong reference date for an event from an earlier turn).

Resolution is conservative: only expressions that match a generic
relative-time grammar are rewritten. Absolute timestamps and unrecognized
strings pass through unchanged, so recall embeddings and non-temporal
facts are unaffected.
"""

from __future__ import annotations

import calendar
import datetime
import json
import re

from houyi.adapters.memory.reasoner import _normalize_observation_date


def _parse_observation_date(raw: str | None) -> datetime.date | None:
    iso = _normalize_observation_date(raw)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
        return None
    try:
        return datetime.date.fromisoformat(iso)
    except ValueError:
        return None


def _subtract_months(d: datetime.date, n: int) -> datetime.date:
    month = d.month - n
    year = d.year
    while month <= 0:
        month += 12
        year -= 1
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year, month, min(d.day, last_day))


def _add_months(d: datetime.date, n: int) -> datetime.date:
    month = d.month + n
    year = d.year
    while month > 12:
        month -= 12
        year += 1
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year, month, min(d.day, last_day))


def _most_recent_weekend_day(d: datetime.date, weekday: int) -> datetime.date:
    # weekday(): Monday=0 .. Sunday=6. Walk backwards to the most recent
    # occurrence of the target weekday strictly before d.
    delta = (d.weekday() - weekday) % 7
    if delta == 0:
        delta = 7
    return d - datetime.timedelta(days=delta)


def extract_observation_date(text: str | None) -> str | None:
    """Pull the observation_date out of an extractor input JSON blob.

    The bench and production ingest paths format each turn as a JSON object
    containing observation_date, system_date, text, and speaker_name. This
    recovers the per-turn observation_date so the resolver can anchor on the
    date the conversation actually occurred, not the query-time date. Plain
    (non-JSON) text and blobs without the field yield None, which makes the
    resolver a no-op.
    """
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("observation_date")
    return str(raw).strip() if raw else None


def _resolve_years(low: str, obs: datetime.date) -> str | None:
    # N years ago (with optional hedge: around / about / approximately).
    m = re.search(r"(?:around|about|approximately)?\s*(?:a\s+)?(\d+)\s+years?\s+ago", low)
    if m:
        return str(obs.year - int(m.group(1)))
    # A few / several years ago -> conventional 3-year span.
    if re.search(r"(?:a\s+few|few|several)\s+years?\s+ago", low):
        return str(obs.year - 3)
    if "last year" in low:
        return str(obs.year - 1)
    if "next year" in low:
        return str(obs.year + 1)
    return None


def _resolve_months(low: str, obs: datetime.date) -> str | None:
    m = re.search(r"(?:around|about|approximately)?\s*(\d+)\s+months?\s+ago", low)
    if m:
        target = _subtract_months(obs, int(m.group(1)))
        return f"{target.year:04d}-{target.month:02d}"
    if "last month" in low:
        target = _subtract_months(obs, 1)
        return f"{target.year:04d}-{target.month:02d}"
    if "next month" in low:
        target = _add_months(obs, 1)
        return f"{target.year:04d}-{target.month:02d}"
    return None


def _resolve_days_and_weeks(low: str, obs: datetime.date) -> str | None:
    m = re.search(r"(\d+)\s+weeks?\s+ago", low)
    if m:
        return (obs - datetime.timedelta(weeks=int(m.group(1)))).isoformat()
    m = re.search(r"(\d+)\s+days?\s+ago", low)
    if m:
        return (obs - datetime.timedelta(days=int(m.group(1)))).isoformat()
    if "last weekend" in low:
        return _most_recent_weekend_day(obs, 5).isoformat()
    if "last week" in low:
        return (obs - datetime.timedelta(days=7)).isoformat()
    if "next week" in low:
        return (obs + datetime.timedelta(days=7)).isoformat()
    if "yesterday" in low:
        return (obs - datetime.timedelta(days=1)).isoformat()
    if "tomorrow" in low:
        return (obs + datetime.timedelta(days=1)).isoformat()
    if low == "today" or "today" in low:
        return obs.isoformat()
    return None


def resolve_relative_timestamp(raw_ts: str, observation_date: str | None) -> str:
    """Resolve a relative time string against an observation date.

    Returns the original string unchanged when it is already absolute, is
    not a recognized relative expression, or when the observation date is
    unavailable or unparseable.
    """
    if not raw_ts:
        return raw_ts
    obs = _parse_observation_date(observation_date)
    if obs is None:
        return raw_ts
    text = raw_ts.strip()
    if _looks_absolute(text):
        return raw_ts
    low = text.lower()
    return (
        _resolve_years(low, obs)
        or _resolve_months(low, obs)
        or _resolve_days_and_weeks(low, obs)
        or raw_ts
    )


_ABSOLUTE_PATTERNS = (
    re.compile(r"^\d{4}$"),
    re.compile(r"^\d{4}-\d{2}(-\d{2})?$"),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}\s+(?:spring|summer|autumn|fall|winter)\b", re.IGNORECASE),
)


def _looks_absolute(text: str) -> bool:
    return any(p.search(text) for p in _ABSOLUTE_PATTERNS)


__all__ = ["extract_observation_date", "resolve_relative_timestamp"]
