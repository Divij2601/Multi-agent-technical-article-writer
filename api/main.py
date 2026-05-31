"""FastAPI application for the Blog Writing Agent.

Run with:  python -m uvicorn api.main:app --reload --port 8000
Then open: http://localhost:8000/
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api import db
from api.jobs import manager, progress_view
from api.schemas import BlogResult, CreateBlogRequest, JobDetail, JobSummary
from src.graph.main_graph import PIPELINE_NODES
from src.models.model_config import OUTPUT_ROOT, RUNS_DIRNAME
from src.utils.markdown_utils import count_words

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
RUNS_DIR = Path(OUTPUT_ROOT) / RUNS_DIRNAME

app = FastAPI(title="Blog Writing Agent API", version="1.0.0")

# Permissive CORS for local development (frontend may be served from a file
# server on a different port during UI work).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    manager.start()


# --- helpers --------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _find_markdown(run_dir: Path, output_path: Optional[str]) -> Optional[Path]:
    if output_path:
        candidate = Path(output_path)
        if candidate.exists():
            return candidate
    matches = sorted(run_dir.glob("*.md"))
    return matches[0] if matches else None


def _job_detail(job: dict[str, Any]) -> JobDetail:
    return JobDetail(
        id=job["id"],
        topic=job["topic"],
        audience_mode=job["audience_mode"],
        execution_mode=job["execution_mode"],
        status=job["status"],
        run_id=job.get("run_id"),
        word_count=job.get("word_count"),
        quality_overall=job.get("quality_overall"),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        as_of=job.get("as_of"),
        image_mode=job.get("image_mode"),
        cancel_requested=job.get("cancel_requested", False),
        error=job.get("error"),
        output_path=job.get("output_path"),
        progress=progress_view(job),
    )


# --- meta -----------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "queue_depth": manager.queue_depth}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "audience_modes": ["beginner", "practitioner", "engineer", "researcher", "executive"],
        "execution_modes": ["budget", "balanced", "quality"],
        "image_modes": ["off", "diagram", "auto", "direct"],
        "pipeline_nodes": PIPELINE_NODES,
        "note": "Generation is paced by provider rate limits and can take many minutes; one blog runs at a time.",
    }


# --- jobs -----------------------------------------------------------------

@app.post("/api/blogs", response_model=JobDetail, status_code=201)
def create_blog(req: CreateBlogRequest) -> JobDetail:
    job_id = manager.submit(
        topic=req.topic.strip(),
        audience_mode=req.audience_mode,
        execution_mode=req.execution_mode,
        as_of=req.as_of,
        image_mode=req.image_mode,
    )
    job = db.get_job(job_id)
    assert job is not None
    return _job_detail(job)


@app.get("/api/blogs", response_model=list[JobSummary])
def list_blogs(limit: int = 50) -> list[JobSummary]:
    return [JobSummary(**{k: job.get(k) for k in JobSummary.model_fields}) for job in db.list_jobs(limit)]


@app.get("/api/blogs/{job_id}", response_model=JobDetail)
def get_blog(job_id: str) -> JobDetail:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_detail(job)


@app.post("/api/blogs/{job_id}/cancel", response_model=JobDetail)
def cancel_blog(job_id: str) -> JobDetail:
    if not db.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    db.request_cancel(job_id)
    job = db.get_job(job_id)
    assert job is not None
    return _job_detail(job)


@app.get("/api/blogs/{job_id}/events")
async def blog_events(job_id: str) -> StreamingResponse:
    if not await asyncio.to_thread(db.get_job, job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        last_payload: Optional[str] = None
        ticks = 0
        while True:
            job = await asyncio.to_thread(db.get_job, job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'detail': 'Job not found'})}\n\n"
                return
            payload = json.dumps({"status": job["status"], "progress": progress_view(job), "error": job.get("error"), "run_id": job.get("run_id")})
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if job["status"] in db.TERMINAL_STATUSES:
                yield f"event: done\ndata: {payload}\n\n"
                return
            ticks += 1
            if ticks % 15 == 0:  # keep-alive comment
                yield ": ping\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/api/blogs/{job_id}/result", response_model=BlogResult)
def get_result(job_id: str) -> BlogResult:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != db.STATUS_SUCCEEDED:
        raise HTTPException(status_code=409, detail=f"Job is '{job['status']}', not ready.")
    run_id = job.get("run_id")
    if not run_id:
        raise HTTPException(status_code=404, detail="No run_id recorded for this job.")
    run_dir = _run_dir(run_id)
    md_path = _find_markdown(run_dir, job.get("output_path"))
    if not md_path:
        raise HTTPException(status_code=404, detail="Markdown artifact not found.")
    markdown = md_path.read_text(encoding="utf-8")
    run_log = _read_json(run_dir / "run_log.json")
    seo = _read_json(run_dir / "seo_metadata.json") or run_log.get("seo_metadata", {})
    images = [f"/api/runs/{run_id}/images/{p.name}" for p in sorted((run_dir / "images").glob("*")) if p.is_file()]
    return BlogResult(
        job_id=job_id,
        run_id=run_id,
        topic=job["topic"],
        markdown=markdown,
        word_count=job.get("word_count") or count_words(markdown),
        seo_metadata=seo,
        quality_score=run_log.get("quality_score", {}),
        images=images,
    )


# --- raw artifacts --------------------------------------------------------

@app.get("/api/runs/{run_id}/markdown")
def run_markdown(run_id: str) -> PlainTextResponse:
    md_path = _find_markdown(_run_dir(run_id), None)
    if not md_path:
        raise HTTPException(status_code=404, detail="Markdown not found")
    return PlainTextResponse(md_path.read_text(encoding="utf-8"), media_type="text/markdown")


@app.get("/api/runs/{run_id}/dashboard")
def run_dashboard(run_id: str) -> FileResponse:
    path = _run_dir(run_id) / "dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(path, media_type="text/html")


@app.get("/api/runs/{run_id}/images/{filename}")
def run_image(run_id: str, filename: str) -> FileResponse:
    # Guard against path traversal: only serve a bare filename from images/.
    safe = Path(filename).name
    path = _run_dir(run_id) / "images" / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


# --- frontend (mounted last so /api/* wins) -------------------------------

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:
    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "Blog Writing Agent API. Frontend not built yet; see /docs for the API."}
