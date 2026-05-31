from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from src.agents.audience_adapter_agent import audience_adapter_node
from src.agents.humanizer_agent import humanizer_node
from src.agents.persona_agents import persona_node
from src.agents.planner_agent import orchestrator_node
from src.agents.quality_scorer_agent import quality_scoring_node
from src.agents.research_agent import citation_enricher_node, research_node
from src.agents.revision_agent import worker_pipeline_node
from src.agents.router_agent import router_node
from src.agents.seo_optimizer_agent import seo_optimizer_node
from src.graph.reducer_graph import image_pipeline_node, reducer_pipeline_node
from src.graph.state import BlogState, RunArtifact
from src.graph.worker_graph import fanout
from src.models.model_config import DEFAULT_AUDIENCE, DEFAULT_EXECUTION_MODE
from src.observability.dashboard import write_dashboard
from src.observability.metrics import summarize_metrics
from src.observability.run_logger import write_run_log
from src.observability.tracing import trace_node
from src.outputs.artifact_store import create_run_dir
from src.outputs.writer import write_markdown, write_seo_metadata
from src.utils.logging_utils import dump_json


def _route_after_audience(state: BlogState) -> str:
    return "research" if state["needs_research"] else "citation_enricher"


def _trace(name: str, fn):
    def wrapper(state: BlogState):
        return trace_node(name, lambda: fn(state))

    return wrapper


def export_node(state: BlogState) -> dict[str, Any]:
    run_dir = Path(state["run_dir"])
    plan = state["plan"]
    assert plan is not None
    markdown_path = write_markdown(run_dir, plan.blog_title, state["final"])
    seo_metadata_path = write_seo_metadata(run_dir, state.get("seo_metadata", {}))
    metrics_summary = summarize_metrics(state.get("trace_events", []), state.get("retry_records", []), state.get("fallback_reasons", []))
    manifest = RunArtifact(
        run_id=state["run_id"],
        markdown_path=str(markdown_path),
        image_paths=state.get("image_paths", []),
        seo_metadata_path=str(seo_metadata_path),
    ).model_dump()
    return {
        "output_path": str(markdown_path),
        "seo_metadata": state.get("seo_metadata", {}),
        "metrics_summary": metrics_summary,
        "artifact_manifest": manifest,
    }


def dashboard_node(state: BlogState) -> dict[str, Any]:
    run_dir = Path(state["run_dir"])
    payload = {
        "run_id": state["run_id"],
        "topic": state["topic"],
        "output_path": state.get("output_path", ""),
        "quality_score": state.get("quality_score", {}),
        "retry_records": state.get("retry_records", []),
        "fallback_reasons": state.get("fallback_reasons", []),
        "metrics_summary": state.get("metrics_summary", {}),
        "trace_events": state.get("trace_events", []),
        "seo_metadata": state.get("seo_metadata", {}),
        "verification_reports": state.get("verification_reports", []),
        "artifact_manifest": state.get("artifact_manifest", {}),
    }
    trace_path = write_run_log(run_dir, payload)
    dashboard_path = write_dashboard(run_dir, payload)
    manifest = dict(state.get("artifact_manifest", {}))
    manifest["trace_path"] = str(trace_path)
    manifest["dashboard_path"] = str(dashboard_path)
    return {"artifact_manifest": manifest}


def build_graph():
    graph = StateGraph(BlogState)
    graph.add_node("router", _trace("router", router_node))
    graph.add_node("audience_adapter", _trace("audience_adapter", audience_adapter_node))
    graph.add_node("research", _trace("research", research_node))
    graph.add_node("citation_enricher", _trace("citation_enricher", citation_enricher_node))
    graph.add_node("orchestrator", _trace("orchestrator", orchestrator_node))
    graph.add_node("persona", _trace("persona", persona_node))
    graph.add_node("worker_pipeline", _trace("worker_pipeline", worker_pipeline_node))
    graph.add_node("reducer", _trace("reducer", reducer_pipeline_node))
    graph.add_node("humanizer", _trace("humanizer", humanizer_node))
    graph.add_node("seo_optimizer", _trace("seo_optimizer", seo_optimizer_node))
    graph.add_node("image_pipeline", _trace("image_pipeline", image_pipeline_node))
    graph.add_node("quality_scoring", _trace("quality_scoring", quality_scoring_node))
    graph.add_node("export", _trace("export", export_node))
    graph.add_node("dashboard", _trace("dashboard", dashboard_node))

    graph.add_edge(START, "router")
    graph.add_edge("router", "audience_adapter")
    graph.add_conditional_edges("audience_adapter", _route_after_audience, {"research": "research", "citation_enricher": "citation_enricher"})
    graph.add_edge("research", "citation_enricher")
    graph.add_edge("citation_enricher", "orchestrator")
    graph.add_edge("orchestrator", "persona")
    graph.add_conditional_edges("persona", fanout, ["worker_pipeline"])
    graph.add_edge("worker_pipeline", "reducer")
    graph.add_edge("reducer", "humanizer")
    graph.add_edge("humanizer", "seo_optimizer")
    graph.add_edge("seo_optimizer", "image_pipeline")
    graph.add_edge("image_pipeline", "quality_scoring")
    graph.add_edge("quality_scoring", "export")
    graph.add_edge("export", "dashboard")
    graph.add_edge("dashboard", END)
    return graph.compile()


app = build_graph()


def run(topic: str, as_of: Optional[str] = None, audience_mode: str = DEFAULT_AUDIENCE, execution_mode: str = DEFAULT_EXECUTION_MODE) -> dict[str, Any]:
    if as_of is None:
        as_of = date.today().isoformat()
    run_id, run_dir = create_run_dir()
    # Serialize the section fan-out by default. Free-tier providers throttle on
    # tokens-per-minute (Groq) and requests-per-day (Gemini); running 7-9 writer
    # branches in parallel bursts past those limits instantly. max_concurrency=1
    # paces the pipeline so backoff in the LLM factory can ride out the windows.
    # Override with BLOG_MAX_CONCURRENCY if you have higher rate limits.
    try:
        max_concurrency = max(1, int(os.getenv("BLOG_MAX_CONCURRENCY", "1")))
    except ValueError:
        max_concurrency = 1
    return app.invoke(
        {
            "topic": topic,
            "as_of": as_of,
            "audience_mode": audience_mode,
            "execution_mode": execution_mode,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "source_registry": {},
            "section_evidence_map": {},
            "audience_profile": {},
            "persona_bundle": {},
            "plan": None,
            "sections": [],
            "section_citations": [],
            "verification_reports": [],
            "revision_required_sections": [],
            "retry_records": [],
            "fallback_reasons": [],
            "trace_events": [],
            "merged_md": "",
            "humanized_md": "",
            "seo_metadata": {},
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
            "output_path": "",
            "image_paths": [],
            "quality_score": {},
            "metrics_summary": {},
            "artifact_manifest": {},
        },
        config={"max_concurrency": max_concurrency},
    )
