from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import EvidenceItem, SectionCitations, SectionVerification, Task
from src.models.llm_factory import invoke_text
from src.prompts.verification_prompts import REVISION_SYSTEM
from src.utils.markdown_utils import inject_citations_into_section
from src.utils.retry_utils import add_fallback_reason, add_retry_record


def revision_agent_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    evidence = [EvidenceItem(**item) for item in payload.get("evidence", [])]
    verification = SectionVerification(**payload.get("verification", {"task_id": task.id}))
    section_citations = SectionCitations(**payload.get("section_citations_item", {"task_id": task.id, "citations": []}))
    section_md = payload["section_md"]
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []

    if not verification.revision_required:
        final_section = inject_citations_into_section(section_md, section_citations)
        return {"sections": [(task.id, final_section)]}

    evidence_text = "\n".join(f"- {item.title} | {item.url} | {item.snippet or ''}" for item in evidence)
    report_text = "\n".join(
        f"- {report.claim_text} => {report.verdict} | {', '.join(report.matched_urls) or 'no URLs'} | {report.rationale}"
        for report in verification.claim_reports
    )
    try:
        revised_section, llm_records = invoke_text(
            [
                SystemMessage(content=REVISION_SYSTEM),
                HumanMessage(content=f"Task id: {task.id}\nOriginal section:\n{section_md}\n\nClaim reports:\n{report_text}\n\nAllowed evidence:\n{evidence_text}"),
            ],
            task_kind="revision",
            execution_mode=payload.get("execution_mode", "balanced"),
            attempts_per_client=1,
        )
        for record in llm_records:
            add_retry_record(retry_records, node="revision_agent", scope=f"task:{task.id}:{record['provider']}", attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
    except Exception as exc:
        revised_section = section_md
        add_fallback_reason(fallback_reasons, node="revision_agent", scope=f"task:{task.id}", reason=f"Revision failed; keeping original section: {exc}")

    final_section = inject_citations_into_section(revised_section, section_citations)
    return {"sections": [(task.id, final_section)], "retry_records": retry_records, "fallback_reasons": fallback_reasons}


def worker_pipeline_node(payload: dict) -> dict:
    from src.agents.section_writer_agent import worker_draft_node
    from src.agents.fact_checker_agent import fact_checker_node

    state = dict(payload)
    for step in (worker_draft_node, fact_checker_node, revision_agent_node):
        state.update(step(state))
    return {
        "sections": state.get("sections", []),
        "verification_reports": state.get("verification_reports", []),
        "section_citations": state.get("section_citations", []),
        "revision_required_sections": state.get("revision_required_sections", []),
        "retry_records": state.get("retry_records", []),
        "fallback_reasons": state.get("fallback_reasons", []),
    }
