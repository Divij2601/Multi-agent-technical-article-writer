"""Backend smoke test — verifies the job queue / DB / progress / cancel plumbing
with a FAKE fast engine, so no LLM calls or long waits are involved.

Run:  python -m tests.smoke_backend
Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from src.graph.main_graph import RunCancelled


def _wait_until(predicate, timeout=8.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="blogapi_smoke_"))

    # Point the DB at a throwaway file BEFORE importing anything that uses it.
    from api import db
    db.DB_PATH = tmp / "jobs.db"

    from api.jobs import JobManager, progress_view

    # --- fake engines -----------------------------------------------------
    def fast_engine(topic, as_of, audience_mode, execution_mode, image_mode, on_event=None, should_cancel=None):
        on_event({"type": "started", "run_id": "run_fast", "run_dir": str(tmp)})
        for node in ["router", "orchestrator", "worker_pipeline", "export", "dashboard"]:
            if should_cancel and should_cancel():
                raise RunCancelled("run_fast")
            on_event({"type": "node_completed", "node": node})
        on_event({"type": "finished", "run_id": "run_fast", "output_path": str(tmp / "x.md")})
        return {"run_id": "run_fast", "output_path": str(tmp / "x.md"), "final": "# Title\n\n" + "word " * 120, "quality_score": {"overall": 8}}

    def slow_cancelable_engine(topic, as_of, audience_mode, execution_mode, image_mode, on_event=None, should_cancel=None):
        on_event({"type": "started", "run_id": "run_slow", "run_dir": str(tmp)})
        for node in ["router", "audience_adapter", "research", "orchestrator", "worker_pipeline", "export"]:
            time.sleep(0.3)
            if should_cancel and should_cancel():
                raise RunCancelled("run_slow")
            on_event({"type": "node_completed", "node": node})
        return {"run_id": "run_slow", "output_path": "", "final": "x", "quality_score": {}}

    # --- test 1: successful job ------------------------------------------
    mgr = JobManager(engine=fast_engine)
    mgr.start()
    job_id = mgr.submit(topic="RAG in production", audience_mode="engineer", execution_mode="balanced", as_of=None, image_mode="off")
    assert _wait_until(lambda: (db.get_job(job_id) or {}).get("status") == db.STATUS_SUCCEEDED), "job did not succeed in time"
    job = db.get_job(job_id)
    assert job["word_count"] and job["word_count"] > 50, f"bad word_count: {job['word_count']}"
    assert job["quality_overall"] == 8, f"bad quality: {job['quality_overall']}"
    assert job["run_id"] == "run_fast", f"bad run_id: {job['run_id']}"
    prog = progress_view(job)
    assert prog["percent"] == 100, f"expected 100%, got {prog['percent']}"
    assert "export" in prog["completed_nodes"], "missing completed nodes"
    print(f"[1] success lifecycle OK  (words={job['word_count']}, quality={job['quality_overall']}, nodes={len(prog['completed_nodes'])})")

    # --- test 2: cancellation --------------------------------------------
    mgr2 = JobManager(engine=slow_cancelable_engine)
    mgr2.start()
    job_id2 = mgr2.submit(topic="cancel me", audience_mode="engineer", execution_mode="balanced", as_of=None, image_mode="off")
    assert _wait_until(lambda: (db.get_job(job_id2) or {}).get("status") == db.STATUS_RUNNING), "slow job never started"
    assert db.request_cancel(job_id2), "cancel request rejected"
    assert _wait_until(lambda: (db.get_job(job_id2) or {}).get("status") == db.STATUS_CANCELLED), "job did not cancel"
    print("[2] cancellation OK")

    # --- test 3: failure path --------------------------------------------
    def boom_engine(topic, as_of, audience_mode, execution_mode, image_mode, on_event=None, should_cancel=None):
        on_event({"type": "started", "run_id": "run_boom", "run_dir": str(tmp)})
        raise RuntimeError("All configured text models failed (simulated).")

    mgr3 = JobManager(engine=boom_engine)
    mgr3.start()
    job_id3 = mgr3.submit(topic="will fail", audience_mode="engineer", execution_mode="balanced", as_of=None, image_mode="off")
    assert _wait_until(lambda: (db.get_job(job_id3) or {}).get("status") == db.STATUS_FAILED), "job did not fail"
    assert "simulated" in (db.get_job(job_id3).get("error") or ""), "error not recorded"
    print("[3] failure path OK")

    # --- test 4: listing + restart reconciliation ------------------------
    assert len(db.list_jobs()) >= 3, "list_jobs missing rows"
    # Simulate an orphaned running job, then reconcile.
    orphan = db.create_job("orphan1", topic="orphan", audience_mode="engineer", execution_mode="balanced", as_of=None, image_mode="off")
    db.update_job("orphan1", status=db.STATUS_RUNNING)
    db.reconcile_on_startup()
    assert db.get_job("orphan1")["status"] == db.STATUS_FAILED, "orphan not reconciled"
    print("[4] listing + restart reconciliation OK")

    print("\nALL BACKEND SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
