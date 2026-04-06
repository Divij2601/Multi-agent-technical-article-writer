from __future__ import annotations

import operator
import os
import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.getenv("BLOG_OUTPUT_DIR", PROJECT_ROOT))
IMAGES_DIR = OUTPUT_DIR / "images"

MIN_SECTION_COUNT = 7
MAX_SECTION_COUNT = 8
MIN_TOTAL_TARGET_WORDS = 1800
MAX_TOTAL_TARGET_WORDS = 3200
TARGET_IMAGE_COUNT = 3

DEFAULT_GEMINI_TEXT_MODEL = os.getenv("BLOG_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GROQ_TEXT_MODEL = os.getenv("BLOG_GROQ_MODEL", "llama-3.3-70b-versatile")
DIRECT_IMAGE_MODELS = [
    item.strip()
    for item in os.getenv(
        "BLOG_GEMINI_IMAGE_MODELS",
        "gemini-2.5-flash-image,gemini-3.1-pro-image-preview",
    ).split(",")
    if item.strip()
]

TRUE_VALUES = {"1", "true", "yes", "on"}
IMAGE_MODE = os.getenv("BLOG_IMAGE_MODE", "diagram").strip().lower()
OVERWRITE_ASSETS = os.getenv("BLOG_OVERWRITE_ASSETS", "1").strip().lower() in TRUE_VALUES

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should understand or be able to do after this section.",
    )
    bullets: list[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3-6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (160-500).")
    tags: list[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: list[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. transformer_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt or diagram intent for image creation.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1536x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: list[ImageSpec] = Field(default_factory=list)


class DiagramBlueprint(BaseModel):
    title: str
    subtitle: str = ""
    steps: list[str] = Field(default_factory=list, min_length=3, max_length=5)
    callouts: list[str] = Field(default_factory=list, max_length=3)
    footer: str = ""


class State(TypedDict):
    topic: str
    as_of: str
    recency_days: int
    mode: str
    needs_research: bool
    queries: list[str]
    evidence: list[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[list[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: list[dict]
    final: str
    output_path: str
    image_paths: list[str]


def _build_text_llms() -> list:
    provider = os.getenv("BLOG_LLM_PROVIDER", "auto").strip().lower()
    temperature = float(os.getenv("BLOG_LLM_TEMPERATURE", "0.3"))
    google_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    clients: list = []

    if provider == "gemini" and google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        clients.append(
            ChatGoogleGenerativeAI(
                model=DEFAULT_GEMINI_TEXT_MODEL,
                temperature=temperature,
                google_api_key=google_api_key,
            )
        )
    elif provider == "groq" and groq_api_key:
        from langchain_groq import ChatGroq

        clients.append(
            ChatGroq(
                model=DEFAULT_GROQ_TEXT_MODEL,
                temperature=temperature,
            )
        )
    else:
        if groq_api_key:
            from langchain_groq import ChatGroq

            clients.append(
                ChatGroq(
                    model=DEFAULT_GROQ_TEXT_MODEL,
                    temperature=temperature,
                )
            )
        if google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            clients.append(
                ChatGoogleGenerativeAI(
                    model=DEFAULT_GEMINI_TEXT_MODEL,
                    temperature=temperature,
                    google_api_key=google_api_key,
                )
            )

    if not clients:
        raise RuntimeError(
            "No supported LLM credentials found. Set GOOGLE_API_KEY for Gemini free-tier text models "
            "or GROQ_API_KEY for Groq."
        )

    return clients


LLM_CANDIDATES = _build_text_llms()
llm = LLM_CANDIDATES[0]


def _message_to_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunk.strip() for chunk in chunks if chunk).strip()
    return str(content).strip()


def _invoke_text_model(messages: list, *, attempts_per_client: int = 2) -> str:
    errors: list[str] = []
    for client in LLM_CANDIDATES:
        for attempt in range(attempts_per_client):
            try:
                return _message_to_text(client.invoke(messages))
            except Exception as exc:
                errors.append(f"{client.__class__.__name__} attempt {attempt + 1}: {exc}")
    raise RuntimeError("All configured text models failed. " + " | ".join(errors[-4:]))


def _invoke_structured_model(schema, messages: list, *, attempts_per_client: int = 2):
    errors: list[str] = []
    for client in LLM_CANDIDATES:
        runnable = client.with_structured_output(schema)
        for attempt in range(attempts_per_client):
            try:
                return runnable.invoke(messages)
            except Exception as exc:
                errors.append(f"{client.__class__.__name__} attempt {attempt + 1}: {exc}")
    raise RuntimeError("All configured structured-output models failed. " + " | ".join(errors[-4:]))


def _safe_stem(value: str, default: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or default


def _safe_markdown_path(title: str) -> Path:
    stem = _safe_stem(title, f"blog_{date.today().isoformat()}")
    return OUTPUT_DIR / f"{stem}.md"


def _safe_image_filename(filename: str, index: int) -> str:
    candidate = Path(filename or f"image_{index}.png")
    stem = _safe_stem(candidate.stem, f"image_{index}")
    suffix = candidate.suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    return f"{stem}{suffix}"


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _clean_links(markdown_text: str, allowed_urls: set[str], allow_links: bool) -> str:
    def replace_markdown_link(match: re.Match[str]) -> str:
        label = match.group(1)
        url = match.group(2)
        return match.group(0) if allow_links and url in allowed_urls else label

    markdown_text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", replace_markdown_link, markdown_text)

    if allow_links:
        return markdown_text

    return re.sub(r"https?://\S+", "", markdown_text)


def _image_dimensions(size: str) -> tuple[int, int]:
    width, height = size.split("x")
    return int(width), int(height)


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_multiline(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    spacing: int,
) -> int:
    lines = _wrap_text(draw, text, font, max_width)
    x, y = position
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + spacing
    return y


def _plan_issues(plan: Plan) -> list[str]:
    issues: list[str] = []

    if not (MIN_SECTION_COUNT <= len(plan.tasks) <= MAX_SECTION_COUNT):
        issues.append(f"The plan must contain {MIN_SECTION_COUNT}-{MAX_SECTION_COUNT} sections.")

    total_target_words = sum(task.target_words for task in plan.tasks)
    if total_target_words < MIN_TOTAL_TARGET_WORDS:
        issues.append(
            f"Total target words are too low ({total_target_words}). Make the plan substantial, at least {MIN_TOTAL_TARGET_WORDS}."
        )
    if total_target_words > MAX_TOTAL_TARGET_WORDS:
        issues.append(
            f"Total target words are too high ({total_target_words}). Keep the total under {MAX_TOTAL_TARGET_WORDS}."
        )

    if not any(task.requires_code for task in plan.tasks):
        issues.append("At least one section must set requires_code=true.")

    debug_or_failure = any(
        any(keyword in bullet.lower() for keyword in ("debug", "failure", "edge case", "observability", "pitfall"))
        for task in plan.tasks
        for bullet in task.bullets
    )
    if not debug_or_failure:
        issues.append("Include at least one section covering debugging, edge cases, or failure modes.")

    performance_or_cost = any(
        any(keyword in bullet.lower() for keyword in ("performance", "latency", "cost", "trade-off", "throughput"))
        for task in plan.tasks
        for bullet in task.bullets
    )
    if not performance_or_cost:
        issues.append("Include at least one section covering performance, cost, or trade-offs.")

    for task in plan.tasks:
        if len(task.bullets) < 3:
            issues.append(f"Section '{task.title}' needs at least 3 bullets.")
        if task.target_words < 160:
            issues.append(f"Section '{task.title}' is too short; keep target_words >= 160.")
        if task.target_words > 500:
            issues.append(f"Section '{task.title}' is too long; keep target_words <= 500.")

    return issues


ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed before planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts.
- hybrid (needs_research=true):
  Mostly evergreen but benefits from recent examples, tools, releases, or model references.
- open_book (needs_research=true):
  Mostly volatile: rankings, pricing, weekly news, policy, "latest" or "as of now" topics.

If needs_research=true:
- Output 3-8 high-signal queries.
- Queries must be specific and scoped.
- If the topic includes timing language like latest, current, this week, or 2026, reflect that in the queries.
"""


def router_node(state: State) -> dict:
    topic = state["topic"]
    decision = _invoke_structured_model(
        RouterDecision,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}\nAs of: {state['as_of']}"),
        ],
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


def _tavily_search(query: str, max_results: int = 4) -> list[dict]:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return []

    try:
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
    except Exception:
        return []

    normalized: list[dict] = []
    for result in results or []:
        normalized.append(
            {
                "title": result.get("title") or "",
                "url": result.get("url") or "",
                "snippet": result.get("content") or result.get("snippet") or "",
                "published_at": result.get("published_date") or result.get("published_at"),
                "source": result.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant and authoritative sources.
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do not guess.
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])[:5]
    raw_results: list[dict] = []

    for query in queries:
        raw_results.extend(_tavily_search(query, max_results=4))

    if not raw_results:
        return {"evidence": []}

    compact_results: list[dict] = []
    seen_urls: set[str] = set()
    for result in raw_results:
        url = result.get("url") or ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        compact_results.append(
            {
                "title": (result.get("title") or "")[:160],
                "url": url,
                "snippet": (result.get("snippet") or "")[:320],
                "published_at": result.get("published_at"),
                "source": result.get("source"),
            }
        )
        if len(compact_results) >= 12:
            break

    pack = _invoke_structured_model(
        EvidencePack,
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{compact_results}"),
        ],
    )

    dedup: dict[str, EvidenceItem] = {}
    for item in pack.evidence:
        if item.url:
            dedup[item.url] = item

    return {"evidence": list(dedup.values())}


ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create exactly 7 or 8 sections.
- The total target_words across all sections should land between 1800 and 3200.
- Every section must include:
  1) goal (1 sentence)
  2) 3-6 bullets that are concrete, specific, and non-overlapping
  3) target word count (160-500)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build, compare, measure, inspect, verify, debug, or optimize.
- At least one section must include a minimal code sketch or MWE (set requires_code=true).
- Include at least one section on edge cases, debugging, or observability.
- Include at least one section on performance, cost, or trade-offs.
- End with a practical checklist, synthesis, or next-steps section.

Grounding rules:
- Mode closed_book: keep it evergreen and concept-first.
- Mode hybrid:
  - Use evidence only for fresh examples, tools, models, or releases.
  - Mark sections using fresh info as requires_research=true and requires_citations=true.
- Mode open_book:
  - Set blog_kind="news_roundup".
  - Every section must summarize events plus implications.
  - If evidence is weak, be transparent and avoid unsupported claims.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    base_prompt = (
        f"Topic: {state['topic']}\n"
        f"Mode: {mode}\n"
        f"As of: {state['as_of']}\n\n"
        "Evidence (only use for fresh claims; may be empty):\n"
        f"{[item.model_dump() for item in evidence][:16]}"
    )

    messages = [
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=base_prompt),
    ]

    plan: Optional[Plan] = None
    for _ in range(3):
        plan = _invoke_structured_model(Plan, messages)
        issues = _plan_issues(plan)
        if not issues:
            return {"plan": plan}

        messages.append(
            HumanMessage(
                content=(
                    "Revise the outline and fix all of these issues:\n"
                    + "\n".join(f"- {issue}" for issue in issues)
                    + "\n\nReturn only a valid Plan object."
                )
            )
        )

    assert plan is not None
    return {"plan": plan}


def fanout(state: State):
    plan = state["plan"]
    assert plan is not None

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": plan.model_dump(),
                "evidence": [item.model_dump() for item in state.get("evidence", [])],
            },
        )
        for task in plan.tasks
    ]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write one section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided goal and cover all bullets in order.
- Stay close to target_words (+/- 15%).
- Output only the section content in Markdown.
- Start with a '## <Section Title>' heading.

Substance requirements:
- Make the section genuinely explanatory, not outline-like filler.
- Each bullet must receive at least one meaningful paragraph or a compact bullet block with explanation.
- Prefer 3-5 dense paragraphs, or paragraphs plus supporting bullets.
- Use examples, implementation details, trade-offs, and failure modes where relevant.
- If requires_code=true, include at least one minimal, correct code block.
- Any code block must include the imports or setup it depends on.

Grounding policy:
- If mode == open_book:
  - Do not introduce specific event, company, model, funding, or policy claims unless they are supported by the provided evidence URLs.
  - For supported outside-world claims, cite the URL as ([Source](URL)).
- If requires_citations == true:
  - Cite provided evidence URLs for outside-world claims.
- Evergreen reasoning is fine without citations unless requires_citations is true.
- If citations are not required, do not include raw URLs or external links.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff and avoid marketing language.
- Keep it technically useful for a developer audience.
"""


def _generate_section(task: Task, plan: Plan, topic: str, mode: str, evidence: list[EvidenceItem]) -> str:
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {item.title} | {item.url} | {item.published_at or 'date:unknown'}"
        for item in evidence[:20]
    )

    base_human_message = (
        f"Blog title: {plan.blog_title}\n"
        f"Audience: {plan.audience}\n"
        f"Tone: {plan.tone}\n"
        f"Blog kind: {plan.blog_kind}\n"
        f"Constraints: {plan.constraints}\n"
        f"Topic: {topic}\n"
        f"Mode: {mode}\n\n"
        f"Section title: {task.title}\n"
        f"Goal: {task.goal}\n"
        f"Target words: {task.target_words}\n"
        f"Tags: {task.tags}\n"
        f"requires_research: {task.requires_research}\n"
        f"requires_citations: {task.requires_citations}\n"
        f"requires_code: {task.requires_code}\n"
        f"Bullets:{bullets_text}\n\n"
        f"Evidence (only use these URLs when citing):\n{evidence_text}\n"
    )

    messages = [
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=base_human_message),
    ]

    section_md = _invoke_text_model(messages)
    min_words = max(int(task.target_words * 0.8), 150)
    allowed_urls = {item.url for item in evidence if item.url}
    allow_links = mode == "open_book" or task.requires_citations

    for _ in range(2):
        issues: list[str] = []
        section_md = _clean_links(section_md, allowed_urls, allow_links)
        actual_words = _count_words(section_md)

        if not section_md.startswith("## "):
            issues.append("Start the section with a level-2 Markdown heading.")
        if actual_words < min_words:
            issues.append(
                f"The draft is too short at about {actual_words} words. Expand it to at least {min_words} words."
            )
        if task.requires_code and "```" not in section_md:
            issues.append("Include one minimal, correct code block because requires_code=true.")

        if not issues:
            break

        section_md = _invoke_text_model(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        f"{base_human_message}\n"
                        "Revise this draft so it satisfies the constraints.\n\n"
                        f"Current draft:\n{section_md}\n\n"
                        "Fix these issues:\n"
                        + "\n".join(f"- {issue}" for issue in issues)
                    )
                ),
            ]
        )

    return _clean_links(section_md, allowed_urls, allow_links)


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**item) for item in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    section_md = _generate_section(task, plan, topic, mode, evidence)
    return {"sections": [(task.id, section_md)]}


def merge_content(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda item: item[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide where visuals would materially improve this blog.

Rules:
- Return exactly 3 technical visuals for broad explainers, tutorials, comparisons, or system-design posts.
- Use 2 visuals only if the topic is unusually narrow.
- Never exceed 3 visuals.
- Every visual must be a diagram, flow, architecture map, comparison chart, or checklist-style figure.
- Avoid decorative or photorealistic prompts.
- Filenames must end in .png.
- Insert placeholders exactly as [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- Spread the visuals across the article instead of clustering them together.
- If you genuinely cannot justify visuals, keep md_with_placeholders equal to the input and return images=[].

Return strictly GlobalImagePlan.
"""


def _heading_matches(markdown_text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"^## .+$", markdown_text, flags=re.MULTILINE))


def _insert_placeholders(markdown_text: str, placeholders: list[str]) -> str:
    headings = _heading_matches(markdown_text)
    if not headings:
        return markdown_text.rstrip() + "\n\n" + "\n\n".join(placeholders) + "\n"

    total = len(headings)
    candidate_indexes = sorted({max(0, total // 5), max(0, total // 2), max(0, total - 2)})
    selected_indexes = candidate_indexes[: len(placeholders)]

    offset = 0
    for placeholder, index in zip(placeholders, selected_indexes):
        match = headings[index]
        insert_at = match.end() + offset
        markdown_text = markdown_text[:insert_at] + f"\n\n{placeholder}\n" + markdown_text[insert_at:]
        offset += len(f"\n\n{placeholder}\n")

    return markdown_text


def _default_image_specs(topic: str, plan: Plan) -> list[dict]:
    titles = [task.title for task in plan.tasks]
    middle_title = titles[len(titles) // 2] if titles else "Core Flow"
    end_title = titles[-1] if titles else "Practical Checklist"

    return [
        {
            "placeholder": "[[IMAGE_1]]",
            "filename": "01_topic_overview.png",
            "alt": f"Overview diagram for {topic}",
            "caption": f"High-level overview of the key stages behind {topic}.",
            "prompt": f"Technical overview diagram of {topic} with 4 labeled stages and short explanatory labels.",
            "size": "1536x1024",
            "quality": "medium",
        },
        {
            "placeholder": "[[IMAGE_2]]",
            "filename": "02_core_workflow.png",
            "alt": f"Core workflow for {middle_title}",
            "caption": f"Core workflow or execution path behind '{middle_title}'.",
            "prompt": f"Step-by-step technical workflow for {middle_title} within {topic}, with arrows, labels, and implementation notes.",
            "size": "1536x1024",
            "quality": "medium",
        },
        {
            "placeholder": "[[IMAGE_3]]",
            "filename": "03_checklist_and_pitfalls.png",
            "alt": f"Checklist and pitfalls for {end_title}",
            "caption": f"Practical checklist and common pitfalls related to '{end_title}'.",
            "prompt": f"Checklist-style technical diagram for {end_title} in {topic}, showing best practices, common mistakes, and quick fixes.",
            "size": "1536x1024",
            "quality": "medium",
        },
    ]


def _normalize_image_specs(image_specs: list[dict], topic: str, plan: Plan) -> list[dict]:
    specs = image_specs[:TARGET_IMAGE_COUNT]
    if len(specs) < 2:
        specs = _default_image_specs(topic, plan)

    normalized: list[dict] = []
    for index, raw in enumerate(specs[:TARGET_IMAGE_COUNT], start=1):
        normalized.append(
            {
                "placeholder": f"[[IMAGE_{index}]]",
                "filename": _safe_image_filename(str(raw.get("filename", f"image_{index}.png")), index),
                "alt": str(raw.get("alt") or f"{topic} technical diagram {index}"),
                "caption": str(raw.get("caption") or f"Technical figure {index} for {topic}."),
                "prompt": str(raw.get("prompt") or f"Technical diagram for {topic}, figure {index}."),
                "size": raw.get("size") if raw.get("size") in {"1024x1024", "1024x1536", "1536x1024"} else "1536x1024",
                "quality": raw.get("quality") if raw.get("quality") in {"low", "medium", "high"} else "medium",
            }
        )
    return normalized


def decide_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    merged_md = state["merged_md"]

    try:
        image_plan = _invoke_structured_model(
            GlobalImagePlan,
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Topic: {state['topic']}\n\n"
                        "Insert placeholders and propose technical visual prompts.\n\n"
                        f"{merged_md}"
                    )
                ),
            ],
        )
        md_with_placeholders = image_plan.md_with_placeholders or merged_md
        image_specs = [item.model_dump() for item in image_plan.images]
    except Exception:
        md_with_placeholders = merged_md
        image_specs = []

    normalized_specs = _normalize_image_specs(image_specs, state["topic"], plan)
    placeholders = [spec["placeholder"] for spec in normalized_specs]

    for placeholder in placeholders:
        if placeholder not in md_with_placeholders:
            md_with_placeholders = _insert_placeholders(merged_md, placeholders)
            break

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": normalized_specs,
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    last_error: Optional[Exception] = None
    client = genai.Client(api_key=api_key)

    for model_name in DIRECT_IMAGE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
        except Exception as exc:
            last_error = exc
            continue

        parts = getattr(response, "parts", None)
        if not parts and getattr(response, "candidates", None):
            try:
                parts = response.candidates[0].content.parts
            except Exception:
                parts = None

        if not parts:
            continue

        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data

    if last_error:
        raise RuntimeError(f"No direct Gemini image could be generated: {last_error}") from last_error
    raise RuntimeError("No inline image bytes were returned by the configured Gemini image models.")


def _build_diagram_blueprint(spec: dict, topic: str, plan: Plan) -> DiagramBlueprint:
    prompt = (
        "Create a compact blueprint for a technical diagram.\n"
        "Keep text short enough to fit in a clean PNG.\n"
        "Steps should be 2-7 words each and reflect a real sequence or comparison.\n\n"
        f"Topic: {topic}\n"
        f"Blog title: {plan.blog_title}\n"
        f"Image alt: {spec['alt']}\n"
        f"Caption: {spec['caption']}\n"
        f"Visual intent: {spec['prompt']}"
    )

    try:
        return _invoke_structured_model(
            DiagramBlueprint,
            [
                SystemMessage(
                    content=(
                        "You design concise blueprints for technical diagrams. "
                        "Return only a DiagramBlueprint object."
                    )
                ),
                HumanMessage(content=prompt),
            ],
        )
    except Exception:
        seed_text = " ".join([spec["alt"], spec["caption"], spec["prompt"]])
        phrases = [
            cleaned.strip().title()
            for cleaned in re.split(r"[,\n;]|->", seed_text)
            if 3 <= len(cleaned.strip()) <= 48
        ]
        steps = list(dict.fromkeys(phrases))[:4]
        while len(steps) < 3:
            steps.append(f"Key Step {len(steps) + 1}")

        return DiagramBlueprint(
            title=spec["alt"][:72],
            subtitle=spec["caption"][:96],
            steps=steps[:5],
            callouts=[task.title for task in plan.tasks[:2]],
            footer="Free-tier fallback diagram generated locally.",
        )


def _save_image_bytes_as_png(image_bytes: bytes, output_path: Path) -> None:
    image = Image.open(BytesIO(image_bytes))
    image.save(output_path, format="PNG")


def _render_local_diagram(spec: dict, topic: str, plan: Plan, output_path: Path) -> None:
    blueprint = _build_diagram_blueprint(spec, topic, plan)
    width, height = _image_dimensions(spec["size"])

    image = Image.new("RGB", (width, height), "#F7FAFC")
    draw = ImageDraw.Draw(image)

    top_rgb = (247, 250, 252)
    bottom_rgb = (226, 236, 248)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        row_color = tuple(
            int(top_rgb[index] + (bottom_rgb[index] - top_rgb[index]) * ratio)
            for index in range(3)
        )
        draw.line([(0, y), (width, y)], fill=row_color)

    dark = (22, 28, 36)
    muted = (84, 95, 109)
    border = (203, 213, 225)
    accent = (36, 99, 235)
    accent_soft = (219, 234, 254)

    title_font = _load_font(46, bold=True)
    subtitle_font = _load_font(24)
    step_font = _load_font(27, bold=True)
    body_font = _load_font(22)
    small_font = _load_font(18)

    margin_x = 64
    header_rect = (margin_x, 40, width - margin_x, 180)
    draw.rounded_rectangle(header_rect, radius=28, fill="white", outline=border, width=2)
    current_y = _draw_multiline(
        draw,
        blueprint.title,
        (header_rect[0] + 28, 62),
        font=title_font,
        fill=dark,
        max_width=header_rect[2] - header_rect[0] - 56,
        spacing=6,
    )
    if blueprint.subtitle:
        _draw_multiline(
            draw,
            blueprint.subtitle,
            (header_rect[0] + 28, current_y + 4),
            font=subtitle_font,
            fill=muted,
            max_width=header_rect[2] - header_rect[0] - 56,
            spacing=4,
        )

    steps = blueprint.steps[:5]
    box_count = max(3, len(steps))
    top = 240
    bottom = height - 180
    gap = 26
    box_height = int((bottom - top - (gap * (box_count - 1))) / box_count)
    left = margin_x + 40
    right = width - margin_x - 40

    for index, step in enumerate(steps[:box_count]):
        box_top = top + index * (box_height + gap)
        box_bottom = box_top + box_height
        fill_color = accent_soft if index % 2 == 0 else "white"
        draw.rounded_rectangle(
            (left, box_top, right, box_bottom),
            radius=26,
            fill=fill_color,
            outline=accent if index % 2 == 0 else border,
            width=3,
        )

        badge_center_y = box_top + 34
        draw.ellipse((left + 18, badge_center_y - 18, left + 54, badge_center_y + 18), fill=accent)
        draw.text((left + 29, badge_center_y - 13), str(index + 1), font=small_font, fill="white")

        _draw_multiline(
            draw,
            step,
            (left + 74, box_top + 24),
            font=step_font,
            fill=dark,
            max_width=right - left - 96,
            spacing=4,
        )

        if index < box_count - 1:
            mid_x = (left + right) // 2
            arrow_top = box_bottom + 5
            arrow_bottom = box_bottom + gap - 5
            draw.line((mid_x, arrow_top, mid_x, arrow_bottom), fill=accent, width=4)
            draw.polygon(
                [
                    (mid_x, arrow_bottom + 8),
                    (mid_x - 10, arrow_bottom - 4),
                    (mid_x + 10, arrow_bottom - 4),
                ],
                fill=accent,
            )

    if blueprint.callouts:
        badge_y = height - 110
        current_x = margin_x
        for callout in blueprint.callouts[:3]:
            text_width = draw.textbbox((0, 0), callout, font=body_font)[2]
            badge_width = text_width + 34
            draw.rounded_rectangle(
                (current_x, badge_y, current_x + badge_width, badge_y + 42),
                radius=20,
                fill="white",
                outline=border,
                width=2,
            )
            draw.text((current_x + 17, badge_y + 10), callout, font=body_font, fill=muted)
            current_x += badge_width + 18

    footer = blueprint.footer or f"Figure derived from the blog plan for {topic}."
    _draw_multiline(
        draw,
        footer,
        (margin_x, height - 56),
        font=small_font,
        fill=muted,
        max_width=width - (margin_x * 2),
        spacing=2,
    )

    image.save(output_path, format="PNG")


def _create_visual_asset(spec: dict, topic: str, plan: Plan, output_path: Path) -> Path:
    if output_path.exists() and not OVERWRITE_ASSETS:
        return output_path

    if IMAGE_MODE == "direct":
        image_bytes = _gemini_generate_image_bytes(spec["prompt"])
        _save_image_bytes_as_png(image_bytes, output_path)
        return output_path

    if IMAGE_MODE == "auto":
        try:
            image_bytes = _gemini_generate_image_bytes(spec["prompt"])
            _save_image_bytes_as_png(image_bytes, output_path)
            return output_path
        except Exception:
            pass

    _render_local_diagram(spec, topic, plan, output_path)
    return output_path


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    output_path = _safe_markdown_path(plan.blog_title)

    if not image_specs:
        output_path.write_text(md, encoding="utf-8")
        return {"final": md, "output_path": str(output_path), "image_paths": []}

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []

    for index, spec in enumerate(image_specs, start=1):
        placeholder = spec["placeholder"]
        filename = _safe_image_filename(spec["filename"], index)
        image_path = IMAGES_DIR / filename

        try:
            _create_visual_asset(spec, state["topic"], plan, image_path)
            image_markdown = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            md = md.replace(placeholder, image_markdown)
            image_paths.append(str(image_path))
        except Exception as exc:
            fallback_block = (
                f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                f"> **Alt:** {spec.get('alt', '')}\n>\n"
                f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                f"> **Error:** {exc}\n"
            )
            md = md.replace(placeholder, fallback_block)

    output_path.write_text(md, encoding="utf-8")
    return {"final": md, "output_path": str(output_path), "image_paths": image_paths}


reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("worker", worker_node)
graph.add_node("reducer", reducer_subgraph)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
graph.add_edge("research", "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

app = graph.compile()


def run(topic: str, as_of: Optional[str] = None) -> dict:
    if as_of is None:
        as_of = date.today().isoformat()

    return app.invoke(
        {
            "topic": topic,
            "as_of": as_of,
            "recency_days": 7,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
            "output_path": "",
            "image_paths": [],
        }
    )


if __name__ == "__main__":
    result = run("Inside a Transformer: From Attention to Output Tokens")
    print(result.get("output_path", ""))
