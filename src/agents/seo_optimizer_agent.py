from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import BlogState, SeoMetadata
from src.models.llm_factory import invoke_structured
from src.prompts.seo_prompts import SEO_SYSTEM
from src.utils.retry_utils import add_fallback_reason, add_retry_record


def seo_optimizer_node(state: BlogState) -> dict:
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []
    try:
        seo_metadata, llm_records = invoke_structured(
            SeoMetadata,
            [
                SystemMessage(content=SEO_SYSTEM),
                HumanMessage(content=f"Topic: {state['topic']}\nAudience profile: {state['audience_profile']}\n\nBlog markdown:\n{state['humanized_md'] or state['merged_md']}"),
            ],
            task_kind="seo",
            execution_mode=state["execution_mode"],
        )
        for record in llm_records:
            add_retry_record(retry_records, node="seo_optimizer", scope=record["provider"], attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
        return {"seo_metadata": seo_metadata.model_dump(), "retry_records": retry_records, "fallback_reasons": fallback_reasons}
    except Exception as exc:
        add_fallback_reason(fallback_reasons, node="seo_optimizer", scope="global", reason=f"SEO optimization failed; using fallback metadata: {exc}")
        fallback = SeoMetadata(
            meta_title=state["topic"][:60],
            meta_description=f"Technical guide to {state['topic']}.",
            slug=state["topic"].lower().replace(" ", "-"),
            keywords=[state["topic"]],
            faq_block="",
        )
        return {"seo_metadata": fallback.model_dump(), "retry_records": retry_records, "fallback_reasons": fallback_reasons}
