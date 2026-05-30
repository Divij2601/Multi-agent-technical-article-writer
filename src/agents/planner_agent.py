from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.research_agent import build_section_evidence_map
from src.graph.state import BlogState, Plan
from src.models.llm_factory import invoke_structured
from src.models.model_config import MAX_SECTION_COUNT, MAX_TOTAL_TARGET_WORDS, MIN_SECTION_COUNT, MIN_TOTAL_TARGET_WORDS
from src.prompts.planner_prompts import ORCHESTRATOR_SYSTEM
from src.utils.retry_utils import add_retry_record


def _plan_issues(plan: Plan) -> list[str]:
    issues: list[str] = []
    if not (MIN_SECTION_COUNT <= len(plan.tasks) <= MAX_SECTION_COUNT):
        issues.append(f"The plan must contain {MIN_SECTION_COUNT}-{MAX_SECTION_COUNT} sections.")
    total_target_words = sum(task.target_words for task in plan.tasks)
    if total_target_words < MIN_TOTAL_TARGET_WORDS:
        issues.append(f"Total target words are too low ({total_target_words}).")
    if total_target_words > MAX_TOTAL_TARGET_WORDS:
        issues.append(f"Total target words are too high ({total_target_words}).")
    if not any(task.requires_code for task in plan.tasks):
        issues.append("At least one section must set requires_code=true.")
    return issues


def orchestrator_node(state: BlogState) -> dict:
    audience_profile = state["audience_profile"]
    evidence = state.get("evidence", [])
    retry_records: list[dict] = []
    base_prompt = (
        f"Topic: {state['topic']}\n"
        f"Mode: {state['mode']}\n"
        f"As of: {state['as_of']}\n"
        f"Audience profile: {audience_profile}\n\n"
        f"Evidence:\n{[item.model_dump() for item in evidence][:16]}"
    )
    messages = [SystemMessage(content=ORCHESTRATOR_SYSTEM), HumanMessage(content=base_prompt)]
    plan: Plan | None = None
    for _ in range(3):
        plan, llm_records = invoke_structured(
            Plan,
            messages,
            task_kind="orchestrator",
            execution_mode=state["execution_mode"],
        )
        for record in llm_records:
            add_retry_record(retry_records, node="orchestrator", scope=record["provider"], attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
        issues = _plan_issues(plan)
        if not issues:
            break
        messages.append(HumanMessage(content="Fix these issues and return a valid Plan:\n" + "\n".join(f"- {issue}" for issue in issues)))
    assert plan is not None
    return {
        "plan": plan,
        "section_evidence_map": build_section_evidence_map(plan.tasks, evidence),
        "retry_records": retry_records,
    }
