from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import ClaimExtraction, EvidenceItem, SectionCitations, SectionVerification, Task
from src.models.llm_factory import invoke_structured
from src.prompts.verification_prompts import CLAIM_EXTRACTION_SYSTEM, FACT_CHECK_SYSTEM
from src.utils.claim_utils import build_citation_entries, status_rank
from src.utils.retry_utils import add_fallback_reason, add_retry_record


def fact_checker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    mode = payload.get("mode", "closed_book")
    evidence = [EvidenceItem(**item) for item in payload.get("evidence", [])]
    section_md = payload["section_md"]
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []

    if not evidence and mode != "open_book":
        verification = SectionVerification(task_id=task.id, verification_status="verified", revision_required=False)
        empty_citations = SectionCitations(task_id=task.id)
        return {
            "verification": verification.model_dump(),
            "section_citations_item": empty_citations.model_dump(),
            "verification_reports": [verification.model_dump()],
            "section_citations": [empty_citations.model_dump()],
        }

    try:
        extraction, extract_records = invoke_structured(
            ClaimExtraction,
            [
                SystemMessage(content=CLAIM_EXTRACTION_SYSTEM),
                HumanMessage(content=f"Task id: {task.id}\nMode: {mode}\nSection markdown:\n{section_md}"),
            ],
            task_kind="fact_checker",
            execution_mode=payload.get("execution_mode", "balanced"),
        )
        for record in extract_records:
            add_retry_record(retry_records, node="claim_extractor", scope=f"task:{task.id}:{record['provider']}", attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
    except Exception as exc:
        verification = SectionVerification(task_id=task.id, verification_status="unavailable", revision_required=False)
        empty_citations = SectionCitations(task_id=task.id)
        add_fallback_reason(fallback_reasons, node="fact_checker", scope=f"task:{task.id}", reason=f"Claim extraction failed; keeping original section: {exc}")
        return {
            "verification": verification.model_dump(),
            "section_citations_item": empty_citations.model_dump(),
            "verification_reports": [verification.model_dump()],
            "section_citations": [empty_citations.model_dump()],
            "retry_records": retry_records,
            "fallback_reasons": fallback_reasons,
            "revision_required_sections": [],
            "revision_required": False,
        }

    if not extraction.claims:
        verification = SectionVerification(task_id=task.id, verification_status="verified", revision_required=False)
        empty_citations = SectionCitations(task_id=task.id)
        return {
            "verification": verification.model_dump(),
            "section_citations_item": empty_citations.model_dump(),
            "verification_reports": [verification.model_dump()],
            "section_citations": [empty_citations.model_dump()],
            "retry_records": retry_records,
            "fallback_reasons": fallback_reasons,
            "revision_required_sections": [],
            "revision_required": False,
        }

    evidence_text = "\n".join(f"- {item.title} | {item.url} | {item.snippet or ''} | {item.published_at or 'date:unknown'}" for item in evidence)
    claim_lines = "\n".join(f"- {claim.claim_id} | {claim.claim_type} | {claim.claim_text}" for claim in extraction.claims)
    try:
        verification, fact_records = invoke_structured(
            SectionVerification,
            [
                SystemMessage(content=FACT_CHECK_SYSTEM + "\nReturn a SectionVerification object containing claim_reports for every extracted claim."),
                HumanMessage(content=f"Task id: {task.id}\nClaims:\n{claim_lines}\n\nAllowed evidence:\n{evidence_text}"),
            ],
            task_kind="fact_checker",
            execution_mode=payload.get("execution_mode", "balanced"),
        )
        for record in fact_records:
            add_retry_record(retry_records, node="fact_checker", scope=f"task:{task.id}:{record['provider']}", attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
    except Exception as exc:
        verification = SectionVerification(task_id=task.id, verification_status="unavailable", revision_required=False)
        empty_citations = SectionCitations(task_id=task.id)
        add_fallback_reason(fallback_reasons, node="fact_checker", scope=f"task:{task.id}", reason=f"Fact checking failed; keeping original section: {exc}")
        return {
            "verification": verification.model_dump(),
            "section_citations_item": empty_citations.model_dump(),
            "verification_reports": [verification.model_dump()],
            "section_citations": [empty_citations.model_dump()],
            "retry_records": retry_records,
            "fallback_reasons": fallback_reasons,
            "revision_required_sections": [],
            "revision_required": False,
        }

    if verification.claim_reports:
        worst = max((report.verdict for report in verification.claim_reports), key=status_rank)
        verification.verification_status = worst
        verification.revision_required = worst in {"weakly_supported", "unsupported"}
    section_citations = build_citation_entries(task.id, verification.claim_reports)
    revision_required_sections = [task.id] if verification.revision_required else []
    return {
        "verification": verification.model_dump(),
        "section_citations_item": section_citations.model_dump(),
        "verification_reports": [verification.model_dump()],
        "section_citations": [section_citations.model_dump()],
        "retry_records": retry_records,
        "fallback_reasons": fallback_reasons,
        "revision_required_sections": revision_required_sections,
        "revision_required": verification.revision_required,
    }
