from __future__ import annotations

from src.graph.state import AudienceProfile, BlogState
from src.models.model_config import DEFAULT_AUDIENCE


AUDIENCE_PRESETS = {
    "beginner": AudienceProfile(
        mode="beginner",
        tone="clear, patient, and approachable",
        reading_level="introductory",
        code_density="light",
        depth_budget="medium",
        example_style="small, intuitive examples",
        constraints=["Define jargon before using it", "Prefer intuition before implementation"],
    ),
    "practitioner": AudienceProfile(
        mode="practitioner",
        tone="practical and grounded",
        reading_level="intermediate",
        code_density="medium",
        depth_budget="medium",
        example_style="applied examples and checklists",
        constraints=["Emphasize operational trade-offs", "Prefer implementation patterns over theory"],
    ),
    "engineer": AudienceProfile(
        mode="engineer",
        tone="technical and implementation-oriented",
        reading_level="advanced",
        code_density="heavy",
        depth_budget="high",
        example_style="minimal working examples and system details",
        constraints=["Include APIs, failure modes, and debugging", "Be precise and concise"],
    ),
    "researcher": AudienceProfile(
        mode="researcher",
        tone="analytical and rigorous",
        reading_level="advanced",
        code_density="medium",
        depth_budget="high",
        example_style="comparative and evidence-aware",
        constraints=["Discuss assumptions and limitations", "Prefer formal comparisons and caveats"],
    ),
    "executive": AudienceProfile(
        mode="executive",
        tone="concise and strategic",
        reading_level="broad",
        code_density="none",
        depth_budget="low",
        example_style="decision-oriented summaries",
        constraints=["Minimize code", "Focus on implications, risk, and cost"],
    ),
}


def audience_adapter_node(state: BlogState) -> dict:
    mode = (state.get("audience_mode") or DEFAULT_AUDIENCE).strip().lower()
    profile = AUDIENCE_PRESETS.get(mode, AUDIENCE_PRESETS[DEFAULT_AUDIENCE])
    return {"audience_profile": profile.model_dump()}
