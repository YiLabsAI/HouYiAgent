"""Span storage with SQLite backend.

Provides persistent storage for observability spans with:
- SQLite as default local storage
- Extensible interface for other backends (PostgreSQL, ClickHouse, OTLP)
- Efficient querying by trace_id, node_id, time range
"""

from __future__ import annotations

import atexit
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from houyi.infrastructure.observability.types import SpanSchema, SpanType


@dataclass
class SpanFilter:
    """Filter criteria for span queries."""

    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    node_id: str | None = None
    span_type: SpanType | None = None
    status: str | None = None
    start_time_gte: float | None = None
    start_time_lte: float | None = None
    end_time_gte: float | None = None
    end_time_lte: float | None = None
    limit: int = 1000
    offset: int = 0


@dataclass
class SpanStorageConfig:
    """Configuration for span storage."""

    db_path: str | Path = ".houyi/spans.db"
    max_spans_per_trace: int = 10000
    retention_days: int = 30
    auto_vacuum: bool = True


class SpanStorage(ABC):
    """Abstract base class for span storage backends."""

    @abstractmethod
    def save(self, span: SpanSchema) -> None:
        """Save a span to storage."""
        pass

    @abstractmethod
    def save_batch(self, spans: list[SpanSchema]) -> None:
        """Save multiple spans in a batch."""
        pass

    @abstractmethod
    def get(self, span_id: str) -> SpanSchema | None:
        """Get a span by ID."""
        pass

    @abstractmethod
    def query(self, filter: SpanFilter) -> list[SpanSchema]:
        """Query spans with filter criteria."""
        pass

    @abstractmethod
    def get_trace(self, trace_id: str) -> list[SpanSchema]:
        """Get all spans for a trace."""
        pass

    @abstractmethod
    def delete_trace(self, trace_id: str) -> int:
        """Delete all spans for a trace. Returns count deleted."""
        pass

    @abstractmethod
    def cleanup_old_spans(self, before_timestamp: float) -> int:
        """Delete spans older than timestamp. Returns count deleted."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage connection."""
        pass


class SQLiteSpanStorage(SpanStorage):
    """SQLite-based span storage.

    Thread-safe implementation using connection pooling per thread.
    """

    def __init__(self, config: SpanStorageConfig | None = None):
        self.config = config or SpanStorageConfig()
        self.db_path = Path(self.config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for cursor with auto-commit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_id TEXT,
                    name TEXT NOT NULL,
                    span_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ok',
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL,

                    -- AI-native fields
                    node_id TEXT,
                    model TEXT,
                    provider TEXT,
                    tokens_input INTEGER,
                    tokens_output INTEGER,
                    tokens_total INTEGER,
                    cost_usd REAL,
                    cache_hit INTEGER,
                    tool_name TEXT,

                    -- Parallel execution
                    group_id TEXT,
                    lane_id INTEGER,
                    seq INTEGER,

                    -- Checkpoint lineage
                    parent_trace_id TEXT,
                    restore_checkpoint_id TEXT,
                    replay_mode INTEGER DEFAULT 0,

                    -- Generic
                    attributes TEXT,
                    events TEXT,

                    -- Metadata
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spans_trace_id
                ON spans(trace_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spans_node_id
                ON spans(node_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spans_start_time
                ON spans(start_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spans_span_type
                ON spans(span_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spans_parent_id
                ON spans(parent_id)
            """)

    def _span_to_row(self, span: SpanSchema) -> dict[str, Any]:
        """Convert SpanSchema to database row."""
        duration_ms = None
        if span.end_time and span.start_time:
            duration_ms = (span.end_time - span.start_time) * 1000

        tokens_input = None
        tokens_output = None
        tokens_total = None
        if span.tokens:
            tokens_input = span.tokens.input
            tokens_output = span.tokens.output
            tokens_total = span.tokens.total

        cost_usd = None
        if span.cost:
            cost_usd = span.cost.usd

        return {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "parent_id": span.parent_id,
            "name": span.name,
            "span_type": span.span_type.value
            if hasattr(span.span_type, "value")
            else str(span.span_type),
            "status": span.status,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": duration_ms,
            "node_id": span.node_id,
            "model": span.model,
            "provider": span.provider,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tokens_total": tokens_total,
            "cost_usd": cost_usd,
            "cache_hit": 1 if span.cache_hit else 0 if span.cache_hit is not None else None,
            "tool_name": span.tool_name,
            "group_id": span.group_id,
            "lane_id": span.lane_id,
            "seq": span.seq,
            "parent_trace_id": span.parent_trace_id,
            "restore_checkpoint_id": span.restore_checkpoint_id,
            "replay_mode": 1 if span.replay_mode else 0,
            "attributes": json.dumps(span.attributes, default=str) if span.attributes else None,
            "events": (
                json.dumps([e.model_dump(mode="json") for e in span.events], default=str)
                if span.events
                else None
            ),
        }

    def _row_to_span(self, row: sqlite3.Row) -> SpanSchema:
        """Convert database row to SpanSchema."""
        from houyi.infrastructure.observability.types import CostInfo, SpanEvent, TokenUsage

        tokens = None
        if row["tokens_input"] is not None or row["tokens_output"] is not None:
            tokens = TokenUsage(
                input=row["tokens_input"] or 0,
                output=row["tokens_output"] or 0,
                total=row["tokens_total"] or 0,
            )

        cost = None
        if row["cost_usd"] is not None:
            cost = CostInfo(usd=row["cost_usd"])

        cache_hit = None
        if row["cache_hit"] is not None:
            cache_hit = bool(row["cache_hit"])

        attributes = {}
        if row["attributes"]:
            attributes = json.loads(row["attributes"])

        events = []
        if row["events"]:
            events_data = json.loads(row["events"])
            events = [SpanEvent(**e) for e in events_data]

        return SpanSchema(
            span_id=row["span_id"],
            trace_id=row["trace_id"],
            parent_id=row["parent_id"],
            name=row["name"],
            span_type=SpanType(row["span_type"]),
            status=row["status"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            node_id=row["node_id"],
            model=row["model"],
            provider=row["provider"],
            tokens=tokens,
            cost=cost,
            cache_hit=cache_hit,
            tool_name=row["tool_name"],
            group_id=row["group_id"],
            lane_id=row["lane_id"],
            seq=row["seq"],
            parent_trace_id=row["parent_trace_id"],
            restore_checkpoint_id=row["restore_checkpoint_id"],
            replay_mode=bool(row["replay_mode"]),
            attributes=attributes,
            events=events,
        )

    def save(self, span: SpanSchema) -> None:
        """Save a span to storage."""
        row = self._span_to_row(span)
        columns = ", ".join(row.keys())
        placeholders = ", ".join(["?" for _ in row])
        values = list(row.values())

        with self._cursor() as cursor:
            cursor.execute(
                f"INSERT OR REPLACE INTO spans ({columns}) VALUES ({placeholders})",
                values,
            )

    def save_batch(self, spans: list[SpanSchema]) -> None:
        """Save multiple spans in a batch."""
        if not spans:
            return

        rows = [self._span_to_row(span) for span in spans]
        columns = ", ".join(rows[0].keys())
        placeholders = ", ".join(["?" for _ in rows[0]])

        with self._cursor() as cursor:
            cursor.executemany(
                f"INSERT OR REPLACE INTO spans ({columns}) VALUES ({placeholders})",
                [list(row.values()) for row in rows],
            )

    def get(self, span_id: str) -> SpanSchema | None:
        """Get a span by ID."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM spans WHERE span_id = ?", (span_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_span(row)
            return None

    def query(self, filter: SpanFilter) -> list[SpanSchema]:
        """Query spans with filter criteria."""
        conditions, params = self._build_query_filters(filter)
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM spans
            WHERE {where_clause}
            ORDER BY start_time ASC
            LIMIT ? OFFSET ?
        """
        params.extend([filter.limit, filter.offset])

        with self._cursor() as cursor:
            cursor.execute(query, params)
            return [self._row_to_span(row) for row in cursor.fetchall()]

    @staticmethod
    def _build_query_filters(filter: SpanFilter) -> tuple[list[str], list[Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        filter_pairs = [
            (filter.trace_id, "trace_id = ?"),
            (filter.span_id, "span_id = ?"),
            (filter.parent_span_id, "parent_id = ?"),
            (filter.node_id, "node_id = ?"),
            (filter.span_type.value if filter.span_type else None, "span_type = ?"),
            (filter.status, "status = ?"),
            (filter.start_time_gte, "start_time >= ?"),
            (filter.start_time_lte, "start_time <= ?"),
            (filter.end_time_gte, "end_time >= ?"),
            (filter.end_time_lte, "end_time <= ?"),
        ]
        for value, condition in filter_pairs:
            if value is None:
                continue
            conditions.append(condition)
            params.append(value)
        return conditions, params

    def get_trace(self, trace_id: str) -> list[SpanSchema]:
        """Get all spans for a trace."""
        return self.query(SpanFilter(trace_id=trace_id, limit=self.config.max_spans_per_trace))

    def delete_trace(self, trace_id: str) -> int:
        """Delete all spans for a trace."""
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM spans WHERE trace_id = ?", (trace_id,))
            return cursor.rowcount

    def cleanup_old_spans(self, before_timestamp: float) -> int:
        """Delete spans older than timestamp."""
        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM spans WHERE start_time < ?",
                (before_timestamp,),
            )
            count = cursor.rowcount

            if self.config.auto_vacuum:
                cursor.execute("PRAGMA optimize")

            return count

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics."""
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM spans")
            total_spans = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(DISTINCT trace_id) as count FROM spans")
            total_traces = cursor.fetchone()["count"]

            cursor.execute("""
                SELECT span_type, COUNT(*) as count
                FROM spans
                GROUP BY span_type
            """)
            by_type = {row["span_type"]: row["count"] for row in cursor.fetchall()}

            cursor.execute("""
                SELECT MIN(start_time) as min_time, MAX(start_time) as max_time
                FROM spans
            """)
            time_range = cursor.fetchone()

            return {
                "total_spans": total_spans,
                "total_traces": total_traces,
                "spans_by_type": by_type,
                "time_range": {
                    "min": time_range["min_time"],
                    "max": time_range["max_time"],
                },
                "db_path": str(self.db_path),
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            }

    def close(self) -> None:
        """Close storage connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


# Global storage instance
_storage: SpanStorage | None = None


def get_storage() -> SpanStorage:
    """Get global span storage instance."""
    global _storage
    if _storage is None:
        _storage = SQLiteSpanStorage()
    return _storage


def set_storage(storage: SpanStorage) -> None:
    """Set global span storage instance."""
    global _storage
    _storage = storage


def reset_storage() -> None:
    """Reset global storage instance."""
    global _storage
    if _storage:
        _storage.close()
    _storage = None


atexit.register(reset_storage)
