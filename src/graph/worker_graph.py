from __future__ import annotations

from langgraph.types import Send

from src.graph.state import BlogState


def fanout(state: BlogState):
    plan = state["plan"]
    assert plan is not None
    evidence = state.get("evidence", []) or []
    section_evidence_map = state.get("section_evidence_map", {})
    return [
        Send(
            "worker_pipeline",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": plan.model_dump(),
                "evidence": section_evidence_map.get(str(task.id), [item.model_dump() for item in evidence]),
                "source_registry": state.get("source_registry", {}),
                "audience_profile": state.get("audience_profile", {}),
                "persona_bundle": state.get("persona_bundle", {}),
                "execution_mode": state["execution_mode"],
            },
        )
        for task in plan.tasks
    ]
