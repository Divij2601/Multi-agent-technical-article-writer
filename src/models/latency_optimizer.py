from __future__ import annotations

from .capability_matrix import TASK_CAPABILITIES


def choose_profile(task_kind: str, execution_mode: str) -> str:
    capabilities = TASK_CAPABILITIES.get(task_kind, ("balanced_synthesis",))
    if execution_mode == "budget":
        return capabilities[0]
    if execution_mode == "quality":
        return capabilities[-1]
    return capabilities[min(1, len(capabilities) - 1)]
