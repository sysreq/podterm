"""SQLite connection lifecycle."""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from .schema import SCHEMA, migrate

DB_DIR = os.path.expanduser("~/.local/share/podterm")
DB_PATH = os.path.join(DB_DIR, "runs.db")

# The connection is shared across FastAPI worker threads, the asyncio drain
# loop, and snapshot worker threads (`check_same_thread=False`). That flag only
# disables SQLite's own thread check; it does NOT make concurrent use of a
# single connection safe. Serialize every access through this reentrant lock so
# concurrent threads can't interleave statements / corrupt transaction state.
# It must be reentrant because some db functions call others while holding it.
_LOCK = threading.RLock()

_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(DB_DIR, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        # Wait (rather than immediately raising "database is locked") when a
        # write transaction is held elsewhere.
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.executescript(SCHEMA)
        migrate(_conn)
    return _conn


@contextmanager
def with_conn() -> Iterator[sqlite3.Connection]:
    """Yield the shared connection while holding the serialization lock.

    All public db functions that touch the connection run their body inside
    this context manager so concurrent threads can't corrupt shared state.
    The lock is reentrant, so nested db calls are safe.
    """
    with _LOCK:
        yield get_conn()


def close() -> None:
    global _conn
    with _LOCK:
        if _conn is not None:
            _conn.close()
            _conn = None
