from __future__ import annotations

from datetime import datetime, timezone

from src.graph.state import TraceEvent


def make_event(node: str, phase: str, *, status: str = "ok", elapsed_ms: int | None = None, details: dict | None = None) -> dict:
    return TraceEvent(
        node=node,
        phase=phase,  # type: ignore[arg-type]
        timestamp=datetime.now(timezone.utc).isoformat(),
        elapsed_ms=elapsed_ms,
        status=status,  # type: ignore[arg-type]
        details=details or {},
    ).model_dump()
