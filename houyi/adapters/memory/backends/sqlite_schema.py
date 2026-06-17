"""SQLite schema management for memory backend.

Handles schema initialization, migrations, and table creation.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 4


class SQLiteSchemaManager:
    """Manages SQLite schema initialization and migrations."""

    def __init__(self, conn_manager):
        self._conn_manager = conn_manager

    def ensure_column(self, cur: sqlite3.Cursor, table: str, name: str, ddl: str) -> None:
        """Add table.name if absent. Idempotent across restarts."""
        existing = {row["name"] for row in cur.execute(f"PRAGMA table_info({table})")}
        if name not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    def ensure_vec_table(self, dim: int) -> bool:
        """Lazily create the memories_vec vec0 virtual table.

        Returns True iff the table is now available for writes.
        """
        if not self._conn_manager.vec_available:
            return False
        if self._conn_manager.vec_dim == dim:
            return True
        if self._conn_manager.vec_dim is not None and self._conn_manager.vec_dim != dim:
            logger.warning(
                "embedding dim mismatch: stored=%s new=%s; skipping memories_vec write",
                self._conn_manager.vec_dim,
                dim,
            )
            return False
        conn = self._conn_manager.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0("
                f" embedding float[{dim}]"
                f")"
            )
            cur.execute(
                "INSERT INTO memories_vec_meta(key, value) VALUES('dim', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(dim),),
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("failed to create memories_vec(dim=%s): %s", dim, exc)
            conn.rollback()
            return False
        finally:
            cur.close()
        self._conn_manager.vec_dim = dim
        return True

    def init_schema(self) -> None:
        """Initialize database schema with all required tables."""
        conn = self._conn_manager.get_connection()
        cur = conn.cursor()
        try:
            self._create_memories_table(cur)
            self._create_memories_fts(cur)
            self._create_embedding_cache(cur)
            self._create_entity_state(cur)
            self._create_events(cur)
            self._create_vague_candidates(cur)
            self._create_raw_turn_log(cur)
            self._create_extract_queue(cur)
            self._create_vec_meta(cur)
            self._create_memory_edges(cur)
            self._create_community_labels(cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _create_memories_table(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
            record_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            key TEXT NOT NULL,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'fact',
            tags TEXT DEFAULT '[]',
            confidence REAL DEFAULT 1.0,
            decay REAL DEFAULT 1.0,
            provenance TEXT,
            metadata TEXT DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            ttl REAL,
            valid_from REAL,
            valid_to REAL,
            embedding BLOB,
            UNIQUE(scope, key)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_updated ON memories(updated_at DESC)")

        self.ensure_column(
            cur,
            "memories",
            "embedding_pending",
            "INTEGER NOT NULL DEFAULT 1",
        )
        self.ensure_column(
            cur,
            "memories",
            "embedding_provider",
            "TEXT",
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_emb_pending "
            "ON memories(embedding_pending) WHERE embedding_pending = 1"
        )

    def _create_memories_fts(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
 CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
 key, content, tags,
 content='memories',
 content_rowid='rowid',
 tokenize='unicode61'
 )
 """)
        cur.execute("""
 CREATE TRIGGER IF NOT EXISTS mem_fts_ai AFTER INSERT ON memories BEGIN
 INSERT INTO memories_fts(rowid, key, content, tags)
 VALUES (new.rowid, new.key, new.content, new.tags);
 END
 """)
        cur.execute("""
 CREATE TRIGGER IF NOT EXISTS mem_fts_ad AFTER DELETE ON memories BEGIN
 INSERT INTO memories_fts(memories_fts, rowid, key, content, tags)
 VALUES ('delete', old.rowid, old.key, old.content, old.tags);
 END
 """)
        cur.execute("""
 CREATE TRIGGER IF NOT EXISTS mem_fts_au AFTER UPDATE ON memories BEGIN
 INSERT INTO memories_fts(memories_fts, rowid, key, content, tags)
 VALUES ('delete', old.rowid, old.key, old.content, old.tags);
 INSERT INTO memories_fts(rowid, key, content, tags)
 VALUES (new.rowid, new.key, new.content, new.tags);
 END
 """)

    def _create_embedding_cache(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
 CREATE TABLE IF NOT EXISTS embedding_cache (
 record_id TEXT NOT NULL,
 provider TEXT NOT NULL,
 model TEXT NOT NULL,
 embedding BLOB NOT NULL,
 created_at REAL NOT NULL,
 PRIMARY KEY (record_id, provider, model),
 FOREIGN KEY (record_id) REFERENCES memories(record_id) ON DELETE CASCADE
 )
 """)

    def _create_entity_state(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
 CREATE TABLE IF NOT EXISTS entity_state (
 state_id TEXT PRIMARY KEY,
 namespace TEXT NOT NULL,
 entity TEXT NOT NULL,
 attribute TEXT NOT NULL,
 value TEXT NOT NULL,
 certainty TEXT NOT NULL DEFAULT 'certain',
 qualifiers TEXT,
 valid_from REAL NOT NULL,
 valid_to REAL,
 source_unit_id TEXT,
 created_at REAL NOT NULL,
 UNIQUE(namespace, entity, attribute, valid_from)
 )
 """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_state_active "
            "ON entity_state(namespace, entity, attribute) "
            "WHERE valid_to IS NULL"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_state_temporal "
            "ON entity_state(namespace, entity, attribute, valid_from DESC)"
        )

    def _create_events(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                subject TEXT NOT NULL,
                action TEXT NOT NULL,
                object TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '',
                certainty TEXT NOT NULL DEFAULT 'certain',
                qualifiers TEXT,
                source_anchor TEXT NOT NULL,
                valid_from REAL NOT NULL,
                valid_to REAL,
                created_at REAL NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_subject ON events(namespace, subject, valid_to)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_subject_action "
            "ON events(namespace, subject, action, valid_to)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_subject_timestamp "
            "ON events(namespace, subject, timestamp)"
        )

    def _create_vague_candidates(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
 CREATE TABLE IF NOT EXISTS vague_candidates (
 candidate_id TEXT PRIMARY KEY,
 namespace TEXT NOT NULL,
 entity TEXT NOT NULL,
 attribute TEXT NOT NULL,
 value TEXT NOT NULL,
 qualifiers TEXT,
 source_unit_id TEXT,
 fact_payload TEXT NOT NULL,
 reason TEXT NOT NULL DEFAULT 'vague',
 created_at REAL NOT NULL
 )
 """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vague_candidates_lookup "
            "ON vague_candidates(namespace, entity, attribute, created_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vague_candidates_reason "
            "ON vague_candidates(namespace, reason, created_at DESC)"
        )

    def _create_raw_turn_log(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
 CREATE TABLE IF NOT EXISTS raw_turn_log (
 turn_id TEXT PRIMARY KEY,
 namespace TEXT NOT NULL,
 session_id TEXT NOT NULL,
 turn_index INTEGER NOT NULL,
 role TEXT NOT NULL,
 content TEXT NOT NULL,
 metadata TEXT,
 created_at REAL NOT NULL,
 UNIQUE(namespace, session_id, turn_index)
 )
 """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_turn_session "
            "ON raw_turn_log(namespace, session_id, turn_index)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_turn_created "
            "ON raw_turn_log(namespace, created_at DESC)"
        )

    def _create_extract_queue(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
 CREATE TABLE IF NOT EXISTS extract_queue (
 queue_id TEXT PRIMARY KEY,
 turn_id TEXT NOT NULL,
 namespace TEXT NOT NULL,
 session_id TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending',
 attempts INTEGER NOT NULL DEFAULT 0,
 enqueued_at REAL NOT NULL,
 claimed_at REAL,
 last_error TEXT,
 UNIQUE(turn_id)
 )
 """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_extract_queue_pending "
            "ON extract_queue(state, enqueued_at) WHERE state = 'pending'"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_extract_queue_inflight "
            "ON extract_queue(state, claimed_at) WHERE state = 'in_progress'"
        )

    def _create_vec_meta(self, cur: sqlite3.Cursor) -> None:
        cur.execute(
            """
 CREATE TABLE IF NOT EXISTS memories_vec_meta (
 key TEXT PRIMARY KEY,
 value TEXT NOT NULL
 )
 """
        )
        row = cur.execute("SELECT value FROM memories_vec_meta WHERE key='dim'").fetchone()
        if row is not None:
            with contextlib.suppress(ValueError, TypeError):
                self._conn_manager.vec_dim = int(row["value"])
        if self._conn_manager.vec_available and self._conn_manager.vec_dim is not None:
            cur.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0("
                f" embedding float[{self._conn_manager.vec_dim}]"
                f")"
            )

    def _create_memory_edges(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_edges (
                edge_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                source_unit_id TEXT NOT NULL,
                target_unit_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                valid_from REAL NOT NULL,
                valid_to REAL,
                created_at REAL NOT NULL,
                provenance TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_edges_forward
            ON memory_edges (namespace, source_unit_id, source_type, valid_to, target_unit_id, relation)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_edges_backward
            ON memory_edges (namespace, target_unit_id, target_type, valid_to, source_unit_id, relation)
        """)

    def _create_community_labels(self, cur: sqlite3.Cursor) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_community_labels (
                namespace TEXT NOT NULL,
                node_type TEXT NOT NULL,
                node_id TEXT NOT NULL,
                community_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (namespace, node_type, node_id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_community_labels_lookup
            ON memory_community_labels (namespace, community_id, node_type)
        """)
