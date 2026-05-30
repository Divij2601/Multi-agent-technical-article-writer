from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import BlogState, QualityScore
from src.models.llm_factory import invoke_structured
from src.prompts.scoring_prompts import QUALITY_SCORING_SYSTEM
from src.utils.retry_utils import add_fallback_reason, add_retry_record


def quality_scoring_node(state: BlogState) -> dict:
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []
    try:
        score, llm_records = invoke_structured(
            QualityScore,
            [
                SystemMessage(content=QUALITY_SCORING_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"Audience profile: {state['audience_profile']}\n"
                        f"Verification reports: {state.get('verification_reports', [])}\n"
                        f"SEO metadata: {state.get('seo_metadata', {})}\n\n"
                        f"Final markdown:\n{state['final']}"
                    )
                ),
            ],
            task_kind="quality_scoring",
            execution_mode=state["execution_mode"],
        )
        for record in llm_records:
            add_retry_record(retry_records, node="quality_scoring", scope=record["provider"], attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
        return {"quality_score": score.model_dump(), "retry_records": retry_records, "fallback_reasons": fallback_reasons}
    except Exception as exc:
        add_fallback_reason(fallback_reasons, node="quality_scoring", scope="global", reason=f"Quality scoring failed; using advisory fallback: {exc}")
        fallback = QualityScore(
            clarity=6,
            hallucination_risk=5,
            technical_depth=6,
            seo_readiness=6,
            redundancy=6,
            overall=6,
            summary="Fallback score used because the evaluator failed.",
            needs_revision=False,
        )
        return {"quality_score": fallback.model_dump(), "retry_records": retry_records, "fallback_reasons": fallback_reasons}
