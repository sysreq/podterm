"""SQLite connection lifecycle."""

from __future__ import annotations

import os
import sqlite3

from .schema import SCHEMA, migrate

DB_DIR = os.path.expanduser("~/.local/share/podterm")
DB_PATH = os.path.join(DB_DIR, "runs.db")

_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(DB_DIR, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.executescript(SCHEMA)
        migrate(_conn)
    return _conn


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
