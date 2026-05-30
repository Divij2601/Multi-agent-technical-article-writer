from __future__ import annotations

from src.graph.state import CitationEntry, ClaimReport, SectionCitations
from src.utils.markdown_utils import dedupe_citations


def status_rank(status: str) -> int:
    return {"verified": 0, "weakly_supported": 1, "unsupported": 2, "unavailable": 3}.get(status, 3)


def build_citation_entries(task_id: int, claim_reports: list[ClaimReport]) -> SectionCitations:
    citations: list[CitationEntry] = []
    for report in claim_reports:
        if report.verdict == "unsupported":
            continue
        for url in report.matched_urls[:1]:
            citations.append(
                CitationEntry(
                    claim_id=report.claim_id,
                    span_text=report.claim_text[:160],
                    url=url,
                )
            )
    return SectionCitations(task_id=task_id, citations=dedupe_citations(citations))
