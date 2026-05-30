from __future__ import annotations

from typing import Any, Optional

from src.graph.state import RetryRecord


def add_retry_record(
    retry_records: Optional[list[dict[str, Any]]],
    *,
    node: str,
    scope: str,
    attempt_count: int,
    succeeded: bool,
    last_error: str | None = None,
    fallback_used: str | None = None,
) -> None:
    if retry_records is None:
        return
    retry_records.append(
        RetryRecord(
            node=node,
            scope=scope,
            attempt_count=attempt_count,
            succeeded=succeeded,
            last_error=last_error,
            fallback_used=fallback_used,
        ).model_dump()
    )


def add_fallback_reason(
    fallback_reasons: Optional[list[dict[str, Any]]],
    *,
    node: str,
    scope: str,
    reason: str,
) -> None:
    if fallback_reasons is None:
        return
    fallback_reasons.append({"node": node, "scope": scope, "reason": reason})
