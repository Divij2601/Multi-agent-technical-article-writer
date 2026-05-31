"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

AudienceMode = Literal["beginner", "practitioner", "engineer", "researcher", "executive"]
ExecutionMode = Literal["budget", "balanced", "quality"]
ImageMode = Literal["off", "diagram", "auto", "direct"]


class CreateBlogRequest(BaseModel):
    topic: str = Field(..., min_length=4, max_length=300)
    audience_mode: AudienceMode = "engineer"
    execution_mode: ExecutionMode = "balanced"
    as_of: Optional[str] = Field(default=None, description="ISO date the post is written 'as of'. Defaults to today.")
    image_mode: ImageMode = Field(default="off", description="'off' keeps it text-only (recommended while image quota is limited).")


class JobProgress(BaseModel):
    completed_nodes: list[str] = Field(default_factory=list)
    current_node: Optional[str] = None
    total_nodes: int = 0
    percent: int = 0


class JobSummary(BaseModel):
    id: str
    topic: str
    audience_mode: str
    execution_mode: str
    status: str
    run_id: Optional[str] = None
    word_count: Optional[int] = None
    quality_overall: Optional[int] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class JobDetail(JobSummary):
    as_of: Optional[str] = None
    image_mode: Optional[str] = None
    cancel_requested: bool = False
    error: Optional[str] = None
    output_path: Optional[str] = None
    progress: JobProgress = Field(default_factory=JobProgress)


class BlogResult(BaseModel):
    job_id: str
    run_id: str
    topic: str
    markdown: str
    word_count: int
    seo_metadata: dict[str, Any] = Field(default_factory=dict)
    quality_score: dict[str, Any] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)
