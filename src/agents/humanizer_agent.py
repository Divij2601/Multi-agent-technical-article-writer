from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import BlogState
from src.models.llm_factory import invoke_text
from src.prompts.writer_prompts import HUMANIZER_SYSTEM
from src.utils.retry_utils import add_fallback_reason, add_retry_record


def humanizer_node(state: BlogState) -> dict:
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []
    merged_md = state["merged_md"]
    try:
        humanized_text, llm_records = invoke_text(
            [
                SystemMessage(content=HUMANIZER_SYSTEM),
                HumanMessage(content=f"Audience profile: {state['audience_profile']}\n\nMarkdown draft:\n{merged_md}"),
            ],
            task_kind="humanizer",
            execution_mode=state["execution_mode"],
        )
        for record in llm_records:
            add_retry_record(retry_records, node="humanizer", scope=record["provider"], attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
        humanized_md = humanized_text
    except Exception as exc:
        humanized_md = merged_md
        add_fallback_reason(fallback_reasons, node="humanizer", scope="global", reason=f"Humanizer failed; using merged markdown: {exc}")
    return {"humanized_md": humanized_md, "retry_records": retry_records, "fallback_reasons": fallback_reasons}
