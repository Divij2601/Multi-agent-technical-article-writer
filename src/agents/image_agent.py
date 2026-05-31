from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image, ImageDraw

from src.graph.state import BlogState, DiagramBlueprint, GlobalImagePlan, ImageSpec, Plan
from src.models.llm_factory import invoke_structured
from src.models.model_config import DEFAULT_IMAGE_MODE, DEFAULT_IMAGE_MODELS, OVERWRITE_ASSETS, TARGET_IMAGE_COUNT
from src.utils.image_utils import draw_multiline, image_dimensions, load_font, save_image_bytes_as_png
from src.utils.markdown_utils import safe_image_filename
from src.utils.retry_utils import add_fallback_reason, add_retry_record


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
"""


def merge_content(state: BlogState) -> dict:
    plan = state["plan"]
    assert plan is not None
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda item: item[0])]
    merged_md = f"# {plan.blog_title}\n\n" + "\n\n".join(ordered_sections).strip() + "\n"
    return {"merged_md": merged_md}


def _heading_matches(markdown_text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"^## .+$", markdown_text, flags=re.MULTILINE))


def _insert_placeholders(markdown_text: str, placeholders: list[str]) -> str:
    headings = _heading_matches(markdown_text)
    if not headings:
        return markdown_text.rstrip() + "\n\n" + "\n\n".join(placeholders) + "\n"
    total = len(headings)
    candidate_indexes = sorted({max(0, total // 5), max(0, total // 2), max(0, total - 2)})
    offset = 0
    for placeholder, index in zip(placeholders, candidate_indexes[: len(placeholders)]):
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
        {"placeholder": "[[IMAGE_1]]", "filename": "01_topic_overview.png", "alt": f"Overview diagram for {topic}", "caption": f"High-level overview of the key stages behind {topic}.", "prompt": f"Technical overview diagram of {topic} with 4 labeled stages.", "size": "1536x1024", "quality": "medium"},
        {"placeholder": "[[IMAGE_2]]", "filename": "02_core_workflow.png", "alt": f"Core workflow for {middle_title}", "caption": f"Core workflow or execution path behind '{middle_title}'.", "prompt": f"Step-by-step technical workflow for {middle_title} within {topic}.", "size": "1536x1024", "quality": "medium"},
        {"placeholder": "[[IMAGE_3]]", "filename": "03_checklist_and_pitfalls.png", "alt": f"Checklist and pitfalls for {end_title}", "caption": f"Practical checklist and common pitfalls related to '{end_title}'.", "prompt": f"Checklist-style technical diagram for {end_title} in {topic}.", "size": "1536x1024", "quality": "medium"},
    ]


def _normalize_image_specs(image_specs: list[dict], topic: str, plan: Plan) -> list[dict]:
    specs = image_specs[:TARGET_IMAGE_COUNT] or _default_image_specs(topic, plan)
    if len(specs) < 2:
        specs = _default_image_specs(topic, plan)
    normalized: list[dict] = []
    for index, raw in enumerate(specs[:TARGET_IMAGE_COUNT], start=1):
        normalized.append(
            {
                "placeholder": f"[[IMAGE_{index}]]",
                "filename": safe_image_filename(str(raw.get("filename", f"image_{index}.png")), index),
                "alt": str(raw.get("alt") or f"{topic} technical diagram {index}"),
                "caption": str(raw.get("caption") or f"Technical figure {index} for {topic}."),
                "prompt": str(raw.get("prompt") or f"Technical diagram for {topic}, figure {index}."),
                "size": raw.get("size") if raw.get("size") in {"1024x1024", "1024x1536", "1536x1024"} else "1536x1024",
                "quality": raw.get("quality") if raw.get("quality") in {"low", "medium", "high"} else "medium",
            }
        )
    return normalized


# Image modes that skip generation entirely and keep the text output clean.
_IMAGE_OFF_MODES = {"off", "none", "skip", "disabled", "text_only"}


def decide_images(state: BlogState) -> dict:
    plan = state["plan"]
    assert plan is not None
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []
    merged_md = state["humanized_md"] or state["merged_md"]
    image_mode = state.get("image_mode") or DEFAULT_IMAGE_MODE
    if image_mode in _IMAGE_OFF_MODES:
        # Text-only mode: do not plan or insert any image placeholders.
        return {"md_with_placeholders": merged_md, "image_specs": [], "retry_records": retry_records, "fallback_reasons": fallback_reasons}
    try:
        image_plan, llm_records = invoke_structured(
            GlobalImagePlan,
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(content=f"Blog kind: {plan.blog_kind}\nTopic: {state['topic']}\n\n{merged_md}"),
            ],
            task_kind="diagram_blueprint",
            execution_mode=state["execution_mode"],
        )
        for record in llm_records:
            add_retry_record(retry_records, node="decide_images", scope=record["provider"], attempt_count=record["attempt"], succeeded=record["succeeded"], last_error=record.get("error"))
        md_with_placeholders = image_plan.md_with_placeholders or merged_md
        image_specs = [item.model_dump() for item in image_plan.images]
    except Exception as exc:
        md_with_placeholders = merged_md
        image_specs = []
        add_fallback_reason(fallback_reasons, node="decide_images", scope="global", reason=f"Image planning failed: {exc}")

    normalized_specs = _normalize_image_specs(image_specs, state["topic"], plan)
    placeholders = [spec["placeholder"] for spec in normalized_specs]
    if any(placeholder not in md_with_placeholders for placeholder in placeholders):
        md_with_placeholders = _insert_placeholders(merged_md, placeholders)
    return {"md_with_placeholders": md_with_placeholders, "image_specs": normalized_specs, "retry_records": retry_records, "fallback_reasons": fallback_reasons}


def _gemini_generate_image_bytes(prompt: str, *, scope: str, retry_records: Optional[list[dict]] = None) -> bytes:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    client = genai.Client(api_key=api_key)
    last_error: Optional[str] = None
    for model_name in DEFAULT_IMAGE_MODELS:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt, config=types.GenerateContentConfig(response_modalities=["IMAGE"]))
        except Exception as exc:
            last_error = str(exc)
            add_retry_record(retry_records, node="gemini_image", scope=f"{scope}:{model_name}", attempt_count=1, succeeded=False, last_error=last_error)
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
                add_retry_record(retry_records, node="gemini_image", scope=f"{scope}:{model_name}", attempt_count=1, succeeded=True)
                return inline.data
    raise RuntimeError(last_error or "No inline image bytes were returned by Gemini.")


def _build_diagram_blueprint(spec: dict, topic: str, plan: Plan, *, execution_mode: str) -> DiagramBlueprint:
    prompt = (
        "Create a compact blueprint for a technical diagram.\n"
        "Keep text short enough to fit in a clean PNG.\n"
        "Steps should be 2-7 words each and reflect a real sequence or comparison.\n\n"
        f"Topic: {topic}\nBlog title: {plan.blog_title}\nImage alt: {spec['alt']}\nCaption: {spec['caption']}\nVisual intent: {spec['prompt']}"
    )
    blueprint, _ = invoke_structured(
        DiagramBlueprint,
        [SystemMessage(content="You design concise blueprints for technical diagrams."), HumanMessage(content=prompt)],
        task_kind="diagram_blueprint",
        execution_mode=execution_mode,
    )
    return blueprint


def _render_local_diagram(spec: dict, topic: str, plan: Plan, output_path: Path, *, execution_mode: str) -> None:
    blueprint = _build_diagram_blueprint(spec, topic, plan, execution_mode=execution_mode)
    width, height = image_dimensions(spec["size"])
    image = Image.new("RGB", (width, height), "#F7FAFC")
    draw = ImageDraw.Draw(image)
    dark = (22, 28, 36)
    muted = (84, 95, 109)
    border = (203, 213, 225)
    accent = (36, 99, 235)
    accent_soft = (219, 234, 254)
    title_font = load_font(46, bold=True)
    subtitle_font = load_font(24)
    step_font = load_font(27, bold=True)
    small_font = load_font(18)
    margin_x = 64
    header_rect = (margin_x, 40, width - margin_x, 180)
    draw.rounded_rectangle(header_rect, radius=28, fill="white", outline=border, width=2)
    current_y = draw_multiline(draw, blueprint.title, (header_rect[0] + 28, 62), font=title_font, fill=dark, max_width=header_rect[2] - header_rect[0] - 56, spacing=6)
    if blueprint.subtitle:
        draw_multiline(draw, blueprint.subtitle, (header_rect[0] + 28, current_y + 4), font=subtitle_font, fill=muted, max_width=header_rect[2] - header_rect[0] - 56, spacing=4)
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
        draw.rounded_rectangle((left, box_top, right, box_bottom), radius=26, fill=accent_soft if index % 2 == 0 else "white", outline=accent if index % 2 == 0 else border, width=3)
        draw_multiline(draw, step, (left + 74, box_top + 24), font=step_font, fill=dark, max_width=right - left - 96, spacing=4)
        draw.text((left + 28, box_top + 20), str(index + 1), font=small_font, fill=accent)
    footer = blueprint.footer or f"Figure derived from the blog plan for {topic}."
    draw_multiline(draw, footer, (margin_x, height - 56), font=small_font, fill=muted, max_width=width - (margin_x * 2), spacing=2)
    image.save(output_path, format="PNG")


def _create_visual_asset(spec: dict, topic: str, plan: Plan, output_path: Path, *, execution_mode: str, image_mode: str = DEFAULT_IMAGE_MODE, retry_records: Optional[list[dict]] = None, fallback_reasons: Optional[list[dict]] = None) -> Path:
    if output_path.exists() and not OVERWRITE_ASSETS:
        return output_path
    if image_mode == "direct":
        save_image_bytes_as_png(_gemini_generate_image_bytes(spec["prompt"], scope=spec["filename"], retry_records=retry_records), output_path)
        return output_path
    if image_mode == "auto":
        try:
            save_image_bytes_as_png(_gemini_generate_image_bytes(spec["prompt"], scope=spec["filename"], retry_records=retry_records), output_path)
            return output_path
        except Exception as exc:
            add_fallback_reason(fallback_reasons, node="image_generation", scope=spec["filename"], reason=f"Direct Gemini image generation failed; using local diagram fallback: {exc}")
    _render_local_diagram(spec, topic, plan, output_path, execution_mode=execution_mode)
    return output_path


def generate_and_place_images(state: BlogState) -> dict:
    plan = state["plan"]
    assert plan is not None
    run_dir = Path(state["run_dir"])
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    md = state["md_with_placeholders"] or state["humanized_md"] or state["merged_md"]
    image_mode = state.get("image_mode") or DEFAULT_IMAGE_MODE
    retry_records: list[dict] = []
    fallback_reasons: list[dict] = []
    image_paths: list[str] = []
    for index, spec in enumerate(state.get("image_specs", []) or [], start=1):
        filename = safe_image_filename(spec["filename"], index)
        image_path = images_dir / filename
        try:
            _create_visual_asset(spec, state["topic"], plan, image_path, execution_mode=state["execution_mode"], image_mode=image_mode, retry_records=retry_records, fallback_reasons=fallback_reasons)
            md = md.replace(spec["placeholder"], f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*")
            image_paths.append(str(image_path))
        except Exception as exc:
            # Image generation is non-critical. On failure, remove the placeholder
            # cleanly so the TEXT output is never polluted by error dumps. The
            # failure is recorded in telemetry (run log / dashboard) instead.
            add_fallback_reason(fallback_reasons, node="generate_and_place_images", scope=filename, reason=f"Image generation failed; placeholder removed to keep text clean: {exc}")
            md = md.replace(spec["placeholder"], "")
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return {"final": md, "image_paths": image_paths, "retry_records": retry_records, "fallback_reasons": fallback_reasons}
