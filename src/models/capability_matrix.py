from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    model: str
    quality_rank: int
    cost_rank: int
    speed_rank: int


TASK_CAPABILITIES = {
    "router": ("fast_eval", "balanced_eval"),
    "research_synthesizer": ("fast_eval", "balanced_eval"),
    "orchestrator": ("balanced_synthesis", "quality_synthesis"),
    "persona": ("balanced_synthesis", "quality_synthesis"),
    "writer": ("balanced_synthesis", "quality_synthesis"),
    "revision": ("balanced_synthesis", "quality_synthesis"),
    "fact_checker": ("fast_eval", "balanced_eval"),
    "quality_scoring": ("fast_eval", "balanced_eval"),
    "humanizer": ("balanced_eval", "balanced_synthesis"),
    "seo": ("fast_eval", "balanced_eval"),
    "diagram_blueprint": ("fast_eval", "balanced_eval"),
}
