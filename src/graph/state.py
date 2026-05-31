from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="What the reader should understand or be able to do after this section.")
    bullets: list[str] = Field(..., min_length=3, max_length=7)
    target_words: int = Field(..., description="Target word count for this section (350-750). Aim high; sections must be substantial, not summaries.")
    tags: list[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: list[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1536x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: list[ImageSpec] = Field(default_factory=list)


class DiagramBlueprint(BaseModel):
    title: str
    subtitle: str = ""
    steps: list[str] = Field(default_factory=list, min_length=3, max_length=7)
    callouts: list[str] = Field(default_factory=list, max_length=3)
    footer: str = ""


class ClaimItem(BaseModel):
    claim_id: str
    task_id: int
    claim_text: str
    claim_type: Literal["current_event", "external_fact", "quantitative", "code_claim", "conceptual", "other"]


class ClaimExtraction(BaseModel):
    claims: list[ClaimItem] = Field(default_factory=list)


class ClaimReport(BaseModel):
    claim_id: str
    task_id: int
    claim_text: str
    verdict: Literal["verified", "weakly_supported", "unsupported"]
    matched_urls: list[str] = Field(default_factory=list)
    rationale: str


class SectionVerification(BaseModel):
    task_id: int
    verification_status: Literal["verified", "weakly_supported", "unsupported", "unavailable"] = "verified"
    revision_required: bool = False
    claim_reports: list[ClaimReport] = Field(default_factory=list)


class CitationEntry(BaseModel):
    claim_id: str
    span_text: str
    url: str
    label: str = "Source"


class SectionCitations(BaseModel):
    task_id: int
    citations: list[CitationEntry] = Field(default_factory=list)


class QualityScore(BaseModel):
    clarity: int = Field(..., ge=1, le=10)
    hallucination_risk: int = Field(..., ge=1, le=10)
    technical_depth: int = Field(..., ge=1, le=10)
    seo_readiness: int = Field(..., ge=1, le=10)
    redundancy: int = Field(..., ge=1, le=10)
    overall: int = Field(..., ge=1, le=10)
    summary: str
    needs_revision: bool = False


class RetryRecord(BaseModel):
    node: str
    scope: str
    attempt_count: int
    succeeded: bool
    last_error: Optional[str] = None
    fallback_used: Optional[str] = None


class AudienceProfile(BaseModel):
    mode: Literal["beginner", "practitioner", "engineer", "researcher", "executive"] = "engineer"
    tone: str
    reading_level: str
    code_density: Literal["none", "light", "medium", "heavy"] = "medium"
    depth_budget: Literal["low", "medium", "high"] = "high"
    example_style: str
    constraints: list[str] = Field(default_factory=list)


class PersonaPerspective(BaseModel):
    persona: Literal["optimist", "critic", "neutral"]
    summary: str
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    section_hints: list[str] = Field(default_factory=list)


class PersonaBundle(BaseModel):
    perspectives: list[PersonaPerspective] = Field(default_factory=list)
    synthesis_brief: str = ""


class SeoMetadata(BaseModel):
    meta_title: str
    meta_description: str
    slug: str
    keywords: list[str] = Field(default_factory=list)
    faq_block: str = ""


class RunArtifact(BaseModel):
    run_id: str
    markdown_path: str
    image_paths: list[str] = Field(default_factory=list)
    seo_metadata_path: Optional[str] = None
    trace_path: Optional[str] = None
    dashboard_path: Optional[str] = None


class TraceEvent(BaseModel):
    node: str
    phase: Literal["start", "end"]
    timestamp: str
    elapsed_ms: Optional[int] = None
    status: Literal["ok", "fallback", "error"] = "ok"
    details: dict[str, Any] = Field(default_factory=dict)


class BlogState(TypedDict):
    topic: str
    as_of: str
    audience_mode: str
    execution_mode: str
    run_id: str
    run_dir: str
    mode: str
    needs_research: bool
    queries: list[str]
    evidence: list[EvidenceItem]
    source_registry: dict[str, dict[str, Any]]
    section_evidence_map: dict[str, list[dict[str, Any]]]
    audience_profile: dict[str, Any]
    persona_bundle: dict[str, Any]
    plan: Optional[Plan]
    sections: Annotated[list[tuple[int, str]], operator.add]
    section_citations: Annotated[list[dict[str, Any]], operator.add]
    verification_reports: Annotated[list[dict[str, Any]], operator.add]
    revision_required_sections: Annotated[list[int], operator.add]
    retry_records: Annotated[list[dict[str, Any]], operator.add]
    fallback_reasons: Annotated[list[dict[str, Any]], operator.add]
    trace_events: Annotated[list[dict[str, Any]], operator.add]
    merged_md: str
    humanized_md: str
    seo_metadata: dict[str, Any]
    md_with_placeholders: str
    image_specs: list[dict[str, Any]]
    final: str
    output_path: str
    image_paths: list[str]
    quality_score: dict[str, Any]
    metrics_summary: dict[str, Any]
    artifact_manifest: dict[str, Any]
