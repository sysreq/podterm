"""SQLite schema and migrations."""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    pod_name            TEXT NOT NULL,
    branch              TEXT,
    commit_hash         TEXT,
    commit_msg          TEXT,
    gpu_type            TEXT,
    gpu_count           INTEGER DEFAULT 1,
    driver_version      TEXT,
    cuda_version        TEXT,
    seed                INTEGER,
    seq_len             INTEGER,
    batch_tokens        INTEGER,
    datacenter          TEXT,
    data_variant        TEXT,
    vocab_size          INTEGER,
    model_params        INTEGER,
    cost_per_hr         REAL,
    total_cost          REAL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    duration_seconds    INTEGER,
    exit_code           INTEGER,
    best_val_bpb        REAL,
    peak_memory_mib     INTEGER,
    reserved_memory_mib INTEGER,
    total_steps         INTEGER,
    config_json         TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    total_steps   INTEGER,
    train_loss    REAL,
    val_loss      REAL,
    val_bpb       REAL,
    train_time_ms INTEGER,
    step_avg_ms   REAL,
    PRIMARY KEY (run_id, step)
);

CREATE TABLE IF NOT EXISTS run_diagnostics (
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step       INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    status     TEXT,
    diag_json  TEXT,
    PRIMARY KEY (run_id, step)
);

CREATE INDEX IF NOT EXISTS idx_runs_branch  ON runs(branch);
CREATE INDEX IF NOT EXISTS idx_runs_gpu     ON runs(gpu_type);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
"""

RUN_UPDATE_COLUMNS = {
    "pod_name", "branch", "commit_hash", "commit_msg", "gpu_type", "gpu_count",
    "driver_version", "cuda_version", "seed", "seq_len", "batch_tokens",
    "datacenter", "data_variant", "vocab_size", "model_params", "cost_per_hr",
    "total_cost", "started_at", "finished_at", "duration_seconds", "exit_code",
    "best_val_bpb", "peak_memory_mib", "reserved_memory_mib", "total_steps",
    "config_json",
}


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column adds for DBs created before a schema bump."""
    have = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
    for col, decl in (
        ("driver_version", "TEXT"),
        ("cuda_version", "TEXT"),
        ("seed", "INTEGER"),
        ("seq_len", "INTEGER"),
        ("batch_tokens", "INTEGER"),
    ):
        if col not in have:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")
    conn.commit()
