"""
SQLite connection manager for the MRPL Sovereign AI Workbench.

Provides a centralized, thread-safe connection factory with WAL mode,
foreign key enforcement, and a context-manager pattern for transactions.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# Default database location: <project_root>/data/mrpl_sovereign.db
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "mrpl_sovereign.db"

# Thread-local storage for connections (one connection per thread).
_thread_local = threading.local()

# Module-level configuration — set once at startup via `configure()`.
_db_path: Path = _DEFAULT_DB_PATH
_initialized: bool = False


def configure(db_path: Path | str | None = None) -> Path:
    """
    Set the database path for all subsequent connections.
    Must be called before any `get_connection()` or `transaction()` calls.
    Returns the resolved database path.
    """
    global _db_path, _initialized

    if db_path is not None:
        _db_path = Path(db_path).resolve()
    else:
        _db_path = _DEFAULT_DB_PATH.resolve()

    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _initialized = True
    logger.info("Database configured at: %s", _db_path)
    return _db_path


def get_db_path() -> Path:
    """Return the currently configured database path."""
    return _db_path


def get_connection() -> sqlite3.Connection:
    """
    Return a thread-local SQLite connection.

    The connection is created once per thread and reused. It is configured
    with WAL journal mode, foreign key enforcement, and row factory set to
    sqlite3.Row for dict-like access.
    """
    conn = getattr(_thread_local, "connection", None)

    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            # Connection was closed externally; recreate.
            _thread_local.connection = None

    if not _initialized:
        configure()

    _db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(_db_path),
        timeout=30,
        check_same_thread=False,
    )

    # Enable WAL mode for better concurrent read performance.
    conn.execute("PRAGMA journal_mode = WAL")
    # Enforce foreign key constraints.
    conn.execute("PRAGMA foreign_keys = ON")
    # Use Row factory for dict-like access on result rows.
    conn.row_factory = sqlite3.Row

    _thread_local.connection = conn
    logger.debug("Created new SQLite connection on thread %s", threading.current_thread().name)
    return conn


@contextmanager
def transaction() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for a database transaction.

    Commits on successful exit, rolls back on exception.
    Usage:
        with transaction() as conn:
            conn.execute("INSERT INTO ...")
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def execute_schema(schema_path: Path | None = None) -> None:
    """
    Execute the DDL schema file to create all tables.
    Idempotent — uses CREATE IF NOT EXISTS throughout.
    """
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"

    schema_sql = schema_path.read_text(encoding="utf-8")

    conn = get_connection()
    conn.executescript(schema_sql)
    logger.info("Database schema applied from: %s", schema_path)


def close_connection() -> None:
    """Close the thread-local connection if it exists."""
    conn = getattr(_thread_local, "connection", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.ProgrammingError:
            pass
        finally:
            _thread_local.connection = None
            logger.debug("Closed SQLite connection on thread %s", threading.current_thread().name)
