from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import BlogState, RouterDecision
from src.models.llm_factory import invoke_structured
from src.prompts.router_prompts import ROUTER_SYSTEM
from src.utils.retry_utils import add_fallback_reason, add_retry_record


def router_node(state: BlogState) -> dict:
    retry_records: list[dict] = []
    try:
        decision, llm_records = invoke_structured(
            RouterDecision,
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=f"Topic: {state['topic']}\nAs of: {state['as_of']}"),
            ],
            task_kind="router",
            execution_mode=state["execution_mode"],
        )
        for record in llm_records:
            add_retry_record(
                retry_records,
                node="router",
                scope=record["provider"],
                attempt_count=record["attempt"],
                succeeded=record["succeeded"],
                last_error=record.get("error"),
            )
        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": decision.queries,
            "retry_records": retry_records,
        }
    except Exception as exc:
        fallbacks: list[dict] = []
        add_fallback_reason(fallbacks, node="router", scope="global", reason=f"Router failed; defaulting to hybrid: {exc}")
        return {
            "needs_research": True,
            "mode": "hybrid",
            "queries": [state["topic"]],
            "retry_records": retry_records,
            "fallback_reasons": fallbacks,
        }
