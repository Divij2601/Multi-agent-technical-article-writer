from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import BlogState, EvidenceItem, Plan, Task
from src.models.llm_factory import invoke_text
from src.prompts.writer_prompts import WRITER_SYSTEM
from src.utils.markdown_utils import clean_links, count_words
from src.utils.retry_utils import add_retry_record


def generate_section(task: Task, plan: Plan, topic: str, mode: str, evidence: list[EvidenceItem], *, audience_profile: dict, synthesis_brief: str, execution_mode: str) -> tuple[str, list[dict]]:
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {item.title} | {item.url} | {item.published_at or 'date:unknown'}"
        for item in evidence[:20]
    )
    base_human_message = (
        f"Blog title: {plan.blog_title}\n"
        f"Audience: {plan.audience}\n"
        f"Audience profile: {audience_profile}\n"
        f"Tone: {plan.tone}\n"
        f"Blog kind: {plan.blog_kind}\n"
        f"Constraints: {plan.constraints}\n"
        f"Topic: {topic}\n"
        f"Mode: {mode}\n"
        f"Persona synthesis brief: {synthesis_brief}\n\n"
        f"Section title: {task.title}\n"
        f"Goal: {task.goal}\n"
        f"Target words: {task.target_words}\n"
        f"Tags: {task.tags}\n"
        f"requires_research: {task.requires_research}\n"
        f"requires_citations: {task.requires_citations}\n"
        f"requires_code: {task.requires_code}\n"
        f"Bullets:{bullets_text}\n\n"
        f"Evidence (only use these URLs when citing):\n{evidence_text}\n"
    )
    retry_records: list[dict] = []
    section_md, llm_records = invoke_text(
        [SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=base_human_message)],
        task_kind="writer",
        execution_mode=execution_mode,
    )
    for record in llm_records:
        add_retry_record(retry_records, node="worker_draft", scope=f"task:{task.id}:{record['provider']}", attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))

    min_words = max(int(task.target_words * 0.8), 150)
    allowed_urls = {item.url for item in evidence if item.url}
    allow_links = mode == "open_book" or task.requires_citations
    for _ in range(2):
        section_md = clean_links(section_md, allowed_urls, allow_links)
        issues: list[str] = []
        if not section_md.startswith("## "):
            issues.append("Start the section with a level-2 Markdown heading.")
        if count_words(section_md) < min_words:
            issues.append(f"Expand the draft to at least {min_words} words.")
        if task.requires_code and "```" not in section_md:
            issues.append("Include one minimal, correct code block because requires_code=true.")
        if not issues:
            break
        section_md, revise_records = invoke_text(
            [
                SystemMessage(content=WRITER_SYSTEM),
                HumanMessage(content=f"{base_human_message}\nRevise this draft to fix:\n" + "\n".join(f"- {issue}" for issue in issues) + f"\n\nCurrent draft:\n{section_md}"),
            ],
            task_kind="revision",
            execution_mode=execution_mode,
        )
        for record in revise_records:
            add_retry_record(retry_records, node="worker_revision", scope=f"task:{task.id}:{record['provider']}", attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))

    return clean_links(section_md, allowed_urls, allow_links), retry_records


def worker_draft_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**item) for item in payload.get("evidence", [])]
    persona_bundle = payload.get("persona_bundle", {})
    section_md, retry_records = generate_section(
        task,
        plan,
        payload["topic"],
        payload.get("mode", "closed_book"),
        evidence,
        audience_profile=payload.get("audience_profile", {}),
        synthesis_brief=persona_bundle.get("synthesis_brief", ""),
        execution_mode=payload.get("execution_mode", "balanced"),
    )
    return {"section_md": section_md, "retry_records": retry_records}
