from __future__ import annotations

from typing import Any

from src.agents.image_agent import decide_images, generate_and_place_images, merge_content
from src.graph.state import BlogState


def reducer_pipeline_node(state: BlogState) -> dict[str, Any]:
    working_state: dict[str, Any] = dict(state)
    working_state.update(merge_content(state))
    return {"merged_md": working_state["merged_md"]}


def image_pipeline_node(state: BlogState) -> dict[str, Any]:
    working_state: dict[str, Any] = dict(state)
    working_state.update(decide_images(state))
    working_state.update(generate_and_place_images(working_state))  # type: ignore[arg-type]
    return {
        "md_with_placeholders": working_state.get("md_with_placeholders", ""),
        "image_specs": working_state.get("image_specs", []),
        "final": working_state.get("final", ""),
        "image_paths": working_state.get("image_paths", []),
        "retry_records": working_state.get("retry_records", []),
        "fallback_reasons": working_state.get("fallback_reasons", []),
    }
