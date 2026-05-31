"""Single-worker job queue.

One background thread runs blogs sequentially. Serial execution is intentional:
free-tier providers throttle on tokens-per-minute, so two concurrent generations
would compete for the same budget and both fail (see CLAUDE.md). All durable
state lives in the SQLite store; this module only owns the queue and the worker.

The engine is injectable so the queue/DB/SSE plumbing can be tested with a fast
fake engine instead of an 80-minute real run.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from typing import Any, Callable, Optional

from api import db
from src.graph.main_graph import PIPELINE_NODES, RunCancelled, run_streaming
from src.utils.markdown_utils import count_words

logger = logging.getLogger("api.jobs")

EngineFn = Callable[..., dict]


class JobManager:
    def __init__(self, engine: Optional[EngineFn] = None) -> None:
        self._engine: EngineFn = engine or run_streaming
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        db.init_db()
        for job_id in db.reconcile_on_startup():
            self._queue.put(job_id)
        self._worker = threading.Thread(target=self._run_loop, name="blog-job-worker", daemon=True)
        self._worker.start()
        self._started = True

    def submit(self, *, topic: str, audience_mode: str, execution_mode: str, as_of: Optional[str], image_mode: str) -> str:
        job_id = uuid.uuid4().hex
        db.create_job(job_id, topic=topic, audience_mode=audience_mode, execution_mode=execution_mode, as_of=as_of, image_mode=image_mode)
        self._queue.put(job_id)
        logger.info("queued job %s (%s)", job_id, topic)
        return job_id

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # -- worker internals --------------------------------------------------

    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            except Exception:  # never let the worker thread die
                logger.exception("job %s crashed unexpectedly", job_id)
                db.update_job(job_id, status=db.STATUS_FAILED, error="Unexpected worker error.", finished_at=db.now_iso())
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        job = db.get_job(job_id)
        if not job or job["status"] in db.TERMINAL_STATUSES:
            return
        if db.is_cancel_requested(job_id):
            db.update_job(job_id, status=db.STATUS_CANCELLED, finished_at=db.now_iso())
            return

        db.update_job(job_id, status=db.STATUS_RUNNING, started_at=db.now_iso())
        completed: list[str] = []

        def on_event(event: dict) -> None:
            etype = event.get("type")
            if etype == "started":
                db.update_job(job_id, run_id=event.get("run_id"))
            elif etype == "node_completed":
                node = event.get("node")
                if node and node not in completed:
                    completed.append(node)
                db.update_job(job_id, progress={"completed_nodes": list(completed), "current_node": node})

        def should_cancel() -> bool:
            return db.is_cancel_requested(job_id)

        try:
            final = self._engine(
                topic=job["topic"],
                as_of=job["as_of"],
                audience_mode=job["audience_mode"],
                execution_mode=job["execution_mode"],
                image_mode=job["image_mode"],
                on_event=on_event,
                should_cancel=should_cancel,
            )
        except RunCancelled:
            logger.info("job %s cancelled", job_id)
            db.update_job(job_id, status=db.STATUS_CANCELLED, finished_at=db.now_iso())
            return
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            db.update_job(job_id, status=db.STATUS_FAILED, error=str(exc)[:2000], finished_at=db.now_iso())
            return

        markdown = final.get("final") or final.get("humanized_md") or final.get("merged_md") or ""
        quality = (final.get("quality_score") or {}).get("overall")
        db.update_job(
            job_id,
            status=db.STATUS_SUCCEEDED,
            run_id=final.get("run_id") or job.get("run_id"),
            output_path=final.get("output_path"),
            word_count=count_words(markdown),
            quality_overall=quality,
            finished_at=db.now_iso(),
            progress={"completed_nodes": list(completed), "current_node": None},
        )
        logger.info("job %s succeeded (%d words)", job_id, count_words(markdown))


def progress_view(job: dict[str, Any]) -> dict[str, Any]:
    """Shape a job's raw progress into the JobProgress response model fields."""
    progress = job.get("progress") or {}
    completed = progress.get("completed_nodes") or []
    total = len(PIPELINE_NODES)
    percent = 100 if job.get("status") == db.STATUS_SUCCEEDED else min(99, int(len(completed) / total * 100)) if total else 0
    return {
        "completed_nodes": completed,
        "current_node": progress.get("current_node"),
        "total_nodes": total,
        "percent": percent,
    }


# Module-level singleton used by the FastAPI app.
manager = JobManager()
