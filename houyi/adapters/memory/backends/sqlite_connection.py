"""SQLite connection management for memory backend.

Handles thread-local connections, extension loading, and connection pooling.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Process-level latch so the "sqlite-vec unavailable" notice is emitted at
# most once per process even if many SQLiteConnectionManager
# instances are constructed (bench, studio, tests). The flag is set when
# the first manager observes the unavailable status; subsequent managers
# stay quiet.
_VEC_STATUS_LOGGED = False


class SQLiteConnectionManager:
    """Manages SQLite connections with thread-local storage and extension loading.

    Follows the same thread-safety pattern as SQLiteSpanStorage:
    WAL mode, check_same_thread=False, thread-local connections.
    """

    def __init__(self, db_path: str | Path | None = None, data_dir: str | Path | None = None):
        if db_path:
            self._db_path = Path(db_path)
        elif data_dir:
            self._db_path = Path(data_dir) / ".houyi" / "memory.db"
        else:
            self._db_path = Path(".houyi") / "memory.db"

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._lock = threading.Lock()
        self._vec_available: bool = False
        self._vec_dim: int | None = None
        self._vec_module_available: bool = self._check_vec_module_available()
        # Surface the fallback status exactly once per process so bench /
        # studio operators do not silently consume O(N) scan results while
        # believing the production ANN path is engaged.
        self._maybe_log_vec_status()

    @staticmethod
    def _check_vec_module_available() -> bool:
        """Check if sqlite-vec module can be imported (cached at init)."""
        try:
            import sqlite_vec  # noqa: F401

            return True
        except ImportError:
            return False

    def _maybe_log_vec_status(self) -> None:
        """Emit a one-shot INFO line when the sqlite-vec extension is missing.

        Without this, callers cannot tell from the logs whether vector
        recall is running on the native vec0 path or the Python
        full-table scan fallback. The check fires only when the
        sqlite_vec module itself is missing; if the module imports but
        the runtime extension load fails on a connection, that is
        reported per-connection in try_load_vec_extension instead.
        """
        global _VEC_STATUS_LOGGED
        if _VEC_STATUS_LOGGED:
            return
        if not self._vec_module_available:
            logger.info(
                "sqlite-vec extension unavailable; vector search will use "
                "O(N) scan fallback. Install houyi[memory] for ANN."
            )
            _VEC_STATUS_LOGGED = True

    def try_load_vec_extension(self, conn: sqlite3.Connection) -> bool:
        """Attempt to load the sqlite-vec extension on conn.

        Returns True if the extension is now active. Guarded so the backend
        keeps working (degraded to Python cosine) when the extension is not
        installed, or when the host build of SQLite disables extension
        loading (e.g. distro-shipped Python without --enable-load-extension).
        """
        if not self._vec_module_available:
            return False
        try:
            import sqlite_vec
        except ImportError:
            return False
        try:
            conn.enable_load_extension(True)
            try:
                sqlite_vec.load(conn)
            finally:
                conn.enable_load_extension(False)
            return True
        except (sqlite3.OperationalError, AttributeError) as exc:
            logger.debug("sqlite-vec extension unavailable: %s", exc)
            return False

    def get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local connection.

        If close_all() was called from another thread, the cached
        thread-local reference may be stale (pointing to a closed
        connection).  Detect this by checking membership in the
        live _connections set and self-heal by creating a fresh
        connection.
        """
        if hasattr(self._local, "conn") and self._local.conn is not None:
            # Fast path: cached connection is still live.
            with self._lock:
                is_live = self._local.conn in self._connections
            if is_live:
                return self._local.conn
            # Stale — close_all() cleared the set; discard and rebuild.
            with contextlib.suppress(Exception):
                self._local.conn.close()
            self._local.conn = None

        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        vec_ok = self.try_load_vec_extension(conn)
        with self._lock:
            if vec_ok:
                self._vec_available = True
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._local.conn = conn
        with self._lock:
            self._connections.add(conn)
        return self._local.conn

    def close_all(self) -> None:
        """Close all connections."""
        with self._lock:
            for conn in self._connections:
                with contextlib.suppress(Exception):
                    conn.close()
            self._connections.clear()
            self._local.conn = None

    @property
    def vec_available(self) -> bool:
        """Whether sqlite-vec extension is available."""
        return self._vec_available

    @property
    def db_path(self) -> Path:
        """Return the database file path."""
        return self._db_path

    @property
    def vec_dim(self) -> int | None:
        """Current vector dimension from metadata."""
        return self._vec_dim

    @vec_dim.setter
    def vec_dim(self, value: int | None) -> None:
        """Set vector dimension."""
        self._vec_dim = value
