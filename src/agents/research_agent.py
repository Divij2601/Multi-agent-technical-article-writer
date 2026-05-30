from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from src.graph.state import BlogState, EvidenceItem, EvidencePack
from src.models.llm_factory import invoke_structured
from src.utils.evidence_utils import match_evidence_for_task
from src.utils.retry_utils import add_fallback_reason, add_retry_record


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant and authoritative sources.
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
- Keep snippets short.
- Deduplicate by URL.
"""


def _tavily_search(query: str, max_results: int = 4) -> list[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    normalized: list[dict] = []
    for result in results or []:
        normalized.append(
            {
                "title": result.get("title") or "",
                "url": result.get("url") or "",
                "snippet": result.get("content") or result.get("snippet") or "",
                "published_at": result.get("published_date") or result.get("published_at"),
                "source": result.get("source"),
            }
        )
    return normalized


def research_node(state: BlogState) -> dict:
    queries = (state.get("queries", []) or [])[:5]
    raw_results: list[dict] = []
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []

    for query in queries:
        last_error: Optional[str] = None
        for attempt in range(2):
            try:
                raw_results.extend(_tavily_search(query, max_results=4))
                add_retry_record(retry_records, node="tavily_search", scope=query[:48], attempt_count=attempt + 1, succeeded=True)
                break
            except Exception as exc:
                last_error = str(exc)
                add_retry_record(retry_records, node="tavily_search", scope=query[:48], attempt_count=attempt + 1, succeeded=False, last_error=last_error)
        else:
            add_fallback_reason(fallback_reasons, node="research", scope=query[:48], reason=f"Tavily failed; continuing with empty results: {last_error}")

    if not raw_results:
        if state.get("needs_research"):
            add_fallback_reason(fallback_reasons, node="research", scope="global", reason="No evidence found; continuing with low-evidence mode.")
        return {"evidence": [], "retry_records": retry_records, "fallback_reasons": fallback_reasons}

    compact_results: list[dict] = []
    seen_urls: set[str] = set()
    for result in raw_results:
        url = result.get("url") or ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        compact_results.append(
            {
                "title": (result.get("title") or "")[:160],
                "url": url,
                "snippet": (result.get("snippet") or "")[:320],
                "published_at": result.get("published_at"),
                "source": result.get("source"),
            }
        )
        if len(compact_results) >= 12:
            break

    pack, llm_records = invoke_structured(
        EvidencePack,
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{compact_results}"),
        ],
        task_kind="research_synthesizer",
        execution_mode=state["execution_mode"],
    )
    for record in llm_records:
        add_retry_record(retry_records, node="research_synthesizer", scope=record["provider"], attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))

    dedup: dict[str, EvidenceItem] = {}
    for item in pack.evidence:
        if item.url:
            dedup[item.url] = item
    return {"evidence": list(dedup.values()), "retry_records": retry_records, "fallback_reasons": fallback_reasons}


def citation_enricher_node(state: BlogState) -> dict:
    source_registry = {item.url: item.model_dump() for item in state.get("evidence", []) or []}
    fallback_reasons: list[dict] = []
    if not source_registry and state.get("needs_research"):
        add_fallback_reason(fallback_reasons, node="citation_enricher", scope="global", reason="No evidence available for source registry.")
    return {"source_registry": source_registry, "fallback_reasons": fallback_reasons}


def build_section_evidence_map(tasks, evidence: list[EvidenceItem]) -> dict[str, list[dict[str, Any]]]:
    return {str(task.id): [item.model_dump() for item in match_evidence_for_task(task, evidence)] for task in tasks}
