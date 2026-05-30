from __future__ import annotations

import re

from src.graph.state import EvidenceItem, Task


def tokenize_for_match(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in {"with", "from", "that", "this", "into", "they", "their", "have", "about"}
    }


def match_evidence_for_task(task: Task, evidence: list[EvidenceItem], *, top_k: int = 4) -> list[EvidenceItem]:
    query_tokens = tokenize_for_match(" ".join([task.title, task.goal, *task.bullets, " ".join(task.tags)]))
    scored: list[tuple[int, EvidenceItem]] = []
    for item in evidence:
        evidence_tokens = tokenize_for_match(" ".join([item.title, item.snippet or "", item.source or ""]))
        overlap = len(query_tokens & evidence_tokens)
        freshness_bonus = 1 if item.published_at else 0
        score = overlap * 3 + freshness_bonus + (1 if task.requires_citations and item.url else 0)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    matched = [item for _, item in scored[:top_k]]
    return matched or evidence[:top_k]
