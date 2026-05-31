"""SQLite-backed job store.

All job state lives here so jobs survive a server restart and so the worker
thread and the (async) request handlers share one source of truth without an
in-memory pub/sub. A fresh connection is opened per operation (WAL mode) which
is more than fast enough for this low-volume, single-worker workload.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.models.model_config import OUTPUT_ROOT

DB_PATH = Path(OUTPUT_ROOT) / "jobs.db"

# Job lifecycle: queued -> running -> (succeeded | failed | cancelled)
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL_STATUSES = {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id              TEXT PRIMARY KEY,
                topic           TEXT NOT NULL,
                audience_mode   TEXT NOT NULL,
                execution_mode  TEXT NOT NULL,
                as_of           TEXT,
                image_mode      TEXT,
                status          TEXT NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                run_id          TEXT,
                error           TEXT,
                progress        TEXT,
                output_path     TEXT,
                word_count      INTEGER,
                quality_overall INTEGER,
                created_at      TEXT NOT NULL,
                started_at      TEXT,
                finished_at     TEXT
            )
            """
        )


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["cancel_requested"] = bool(data.get("cancel_requested"))
    raw_progress = data.get("progress")
    try:
        data["progress"] = json.loads(raw_progress) if raw_progress else {}
    except (TypeError, json.JSONDecodeError):
        data["progress"] = {}
    return data


def create_job(job_id: str, *, topic: str, audience_mode: str, execution_mode: str, as_of: Optional[str], image_mode: str) -> dict[str, Any]:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, topic, audience_mode, execution_mode, as_of, image_mode, status, progress, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, topic, audience_mode, execution_mode, as_of, image_mode, STATUS_QUEUED, json.dumps({"completed_nodes": [], "current_node": None}), now_iso()),
        )
    return get_job(job_id)  # type: ignore[return-value]


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    if "progress" in fields and not isinstance(fields["progress"], str):
        fields["progress"] = json.dumps(fields["progress"])
    columns = ", ".join(f"{key} = ?" for key in fields)
    with _connect() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", (*fields.values(), job_id))


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_dict(row) for row in rows]


def request_cancel(job_id: str) -> bool:
    job = get_job(job_id)
    if not job or job["status"] in TERMINAL_STATUSES:
        return False
    update_job(job_id, cancel_requested=1)
    return True


def is_cancel_requested(job_id: str) -> bool:
    job = get_job(job_id)
    return bool(job and job["cancel_requested"])


def reconcile_on_startup() -> list[str]:
    """A server restart loses in-flight work. Mark anything 'running' as failed
    and return the ids of 'queued' jobs so the manager can re-enqueue them."""
    requeue: list[str] = []
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE status = ?",
            (STATUS_FAILED, "Interrupted by server restart.", now_iso(), STATUS_RUNNING),
        )
        rows = conn.execute("SELECT id FROM jobs WHERE status = ?", (STATUS_QUEUED,)).fetchall()
        requeue = [row["id"] for row in rows]
    return requeue
