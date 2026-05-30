from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from src.graph.state import CitationEntry, SectionCitations
from src.models.model_config import OUTPUT_ROOT


def safe_stem(value: str, default: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or default


def safe_markdown_path(run_dir: Path, title: str) -> Path:
    stem = safe_stem(title, f"blog_{date.today().isoformat()}")
    return run_dir / f"{stem}.md"


def safe_image_filename(filename: str, index: int) -> str:
    candidate = Path(filename or f"image_{index}.png")
    stem = safe_stem(candidate.stem, f"image_{index}")
    suffix = candidate.suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    return f"{stem}{suffix}"


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def clean_links(markdown_text: str, allowed_urls: set[str], allow_links: bool) -> str:
    def replace_markdown_link(match: re.Match[str]) -> str:
        label = match.group(1)
        url = match.group(2)
        return match.group(0) if allow_links and url in allowed_urls else label

    markdown_text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", replace_markdown_link, markdown_text)
    if allow_links:
        return markdown_text
    return re.sub(r"https?://\S+", "", markdown_text)


def dedupe_citations(entries: list[CitationEntry]) -> list[CitationEntry]:
    deduped: dict[tuple[str, str], CitationEntry] = {}
    for entry in entries:
        deduped[(entry.claim_id, entry.url)] = entry
    return list(deduped.values())


def inject_citations_into_section(markdown_text: str, section_citations: SectionCitations) -> str:
    updated = markdown_text
    for citation in section_citations.citations:
        marker = f"([{citation.label}]({citation.url}))"
        if citation.url in updated:
            continue
        span = citation.span_text.strip()
        if span and span in updated:
            updated = updated.replace(span, f"{span} {marker}", 1)
            continue
        sentence_match = re.search(r"([^.?!\n]*" + re.escape(span[:32]) + r"[^.?!\n]*[.?!])", updated, flags=re.IGNORECASE)
        if sentence_match:
            sentence = sentence_match.group(1)
            updated = updated.replace(sentence, f"{sentence} {marker}", 1)
        else:
            updated = updated.rstrip() + f"\n\n{marker}"
    return updated


def split_markdown_code_blocks(markdown_text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(```.*?```)", markdown_text, flags=re.DOTALL)
    segments: list[tuple[str, str]] = []
    for part in parts:
        if not part:
            continue
        segments.append(("code", part) if part.startswith("```") else ("text", part))
    return segments
