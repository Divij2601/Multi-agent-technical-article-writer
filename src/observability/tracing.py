from __future__ import annotations

import time
from typing import Any, Callable

from .events import make_event


def trace_node(node: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    start = time.perf_counter()
    start_event = make_event(node, "start")
    status = "ok"
    try:
        output = fn()
    except Exception:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    end_event = make_event(node, "end", status=status, elapsed_ms=elapsed_ms)
    trace_events = [start_event, end_event]
    existing = output.get("trace_events", [])
    output["trace_events"] = existing + trace_events
    return output
