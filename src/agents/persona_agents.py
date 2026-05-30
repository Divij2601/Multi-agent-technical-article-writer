from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import BlogState, PersonaBundle, PersonaPerspective
from src.models.llm_factory import invoke_structured
from src.prompts.planner_prompts import PERSONA_SYSTEM
from src.utils.retry_utils import add_fallback_reason, add_retry_record


def persona_node(state: BlogState) -> dict:
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []
    plan = state["plan"]
    assert plan is not None
    perspectives: list[PersonaPerspective] = []

    for persona in ("optimist", "critic", "neutral"):
        try:
            perspective, llm_records = invoke_structured(
                PersonaPerspective,
                [
                    SystemMessage(content=PERSONA_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Persona: {persona}\n"
                            f"Topic: {state['topic']}\n"
                            f"Audience profile: {state['audience_profile']}\n"
                            f"Plan: {plan.model_dump()}\n"
                            f"Evidence: {[item.model_dump() for item in state.get('evidence', [])][:8]}"
                        )
                    ),
                ],
                task_kind="persona",
                execution_mode=state["execution_mode"],
            )
            for record in llm_records:
                add_retry_record(retry_records, node=f"persona_{persona}", scope=record["provider"], attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
            perspectives.append(perspective)
        except Exception as exc:
            add_fallback_reason(fallback_reasons, node=f"persona_{persona}", scope="global", reason=f"{persona} perspective failed: {exc}")

    synthesis = "\n".join(
        f"{p.persona.title()}: {p.summary}\nRisks: {', '.join(p.risks) or 'none'}\nOpportunities: {', '.join(p.opportunities) or 'none'}"
        for p in perspectives
    ) or "Write a balanced, practical, technically grounded blog."

    bundle = PersonaBundle(perspectives=perspectives, synthesis_brief=synthesis)
    return {"persona_bundle": bundle.model_dump(), "retry_records": retry_records, "fallback_reasons": fallback_reasons}
