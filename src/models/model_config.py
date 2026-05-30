from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.getenv("BLOG_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
IMAGES_DIRNAME = "images"
RUNS_DIRNAME = "runs"

TRUE_VALUES = {"1", "true", "yes", "on"}

MIN_SECTION_COUNT = 7
MAX_SECTION_COUNT = 8
MIN_TOTAL_TARGET_WORDS = 1800
MAX_TOTAL_TARGET_WORDS = 3200
TARGET_IMAGE_COUNT = 3

DEFAULT_GEMINI_TEXT_MODEL = os.getenv("BLOG_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GROQ_TEXT_MODEL = os.getenv("BLOG_GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_FAST_GROQ_MODEL = os.getenv("BLOG_FAST_GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_GEMINI_EVAL_MODEL = os.getenv("BLOG_GEMINI_EVAL_MODEL", DEFAULT_GEMINI_TEXT_MODEL)
DEFAULT_IMAGE_MODELS = [
    item.strip()
    for item in os.getenv(
        "BLOG_GEMINI_IMAGE_MODELS",
        "gemini-2.5-flash-image,gemini-3.1-pro-image-preview",
    ).split(",")
    if item.strip()
]

DEFAULT_EXECUTION_MODE = os.getenv("BLOG_EXECUTION_MODE", "balanced").strip().lower()
DEFAULT_IMAGE_MODE = os.getenv("BLOG_IMAGE_MODE", "diagram").strip().lower()
OVERWRITE_ASSETS = os.getenv("BLOG_OVERWRITE_ASSETS", "1").strip().lower() in TRUE_VALUES
DEFAULT_AUDIENCE = os.getenv("BLOG_DEFAULT_AUDIENCE", "engineer").strip().lower()
