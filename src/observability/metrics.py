from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_metrics(trace_events: list[dict[str, Any]], retry_records: list[dict[str, Any]], fallback_reasons: list[dict[str, Any]]) -> dict[str, Any]:
    node_timings: dict[str, list[int]] = defaultdict(list)
    for event in trace_events:
        if event.get("phase") == "end" and event.get("elapsed_ms") is not None:
            node_timings[event["node"]].append(int(event["elapsed_ms"]))
    return {
        "node_timings_ms": {node: values for node, values in node_timings.items()},
        "retry_count": len(retry_records),
        "fallback_count": len(fallback_reasons),
    }
